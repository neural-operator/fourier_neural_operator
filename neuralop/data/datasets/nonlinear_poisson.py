import torch
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import DataLoader
from .dict_dataset import DictDataset

from neuralop.data.transforms.data_processors import DefaultDataProcessor


path = Path(__file__).resolve().parent.joinpath('data')

def generate_latent_queries(query_res, pad=0, domain_lims=[[-1.4,1.4],[-1.4,1.4]]):
    oneDMeshes = []
    for lower,upper in domain_lims:
        oneDMesh = np.linspace(lower,upper,query_res)
        if pad > 0:
            start = np.linspace(lower - pad/query_res, lower, pad+1)
            stop = np.linspace(upper, upper + pad/query_res, pad+1)
            oneDMesh = np.concatenate([start,oneDMesh,stop])
        oneDMeshes.append(oneDMesh)
    grid = np.stack(np.meshgrid(*oneDMeshes,indexing='xy')) # c, x, y, z(?)
    grid = torch.from_numpy(grid.astype(np.float32))
    latent_queries = grid.permute(*list(range(1,len(domain_lims)+1)), 0)
    return latent_queries

def generate_output_grid(grid_res, coefs, domain_lims=[[-1.4,1.4],[-1.4,1.4]], tol=1e-7):
    xi = domain_lims[0][0] + (domain_lims[0][1] - domain_lims[0][0]) * torch.rand(grid_res)
    yi = domain_lims[1][0] + (domain_lims[1][1] - domain_lims[1][0]) * torch.rand(grid_res)

    c1, c2 = coefs['c1'].item(), coefs['c2'].item()
    
    # Rejection sampling
    theta = torch.arctan2(xi, yi)
    length = torch.sqrt(xi**2 + yi**2)
    r0 = 1.0 + c1 * torch.cos(4 * theta) + c2 * torch.cos(8 * theta)
    mask = r0 > length - tol

    X_filtered = xi[mask]
    Y_filtered = yi[mask]

    XY_filtered = torch.stack((X_filtered, Y_filtered), dim=-1)

    return XY_filtered

def source(x, beta, mu_1, mu_2):
    # Calculate the squared differences between x and the means
    diff = (x[:, 0].unsqueeze(1) - torch.tensor(mu_1)) ** 2 + (x[:, 1].unsqueeze(1) - torch.tensor(mu_2)) ** 2
    exponent = torch.exp(-diff)
    
    # Multiply by beta and sum along the appropriate axis
    result = torch.tensor(beta) * exponent
    source_terms = result.sum(dim=1)

    return source_terms


def load_nonlinear_poisson_pt(
        data_path, 
        query_res=48, 
        domain_padding=0, 
        encode=True, 
        val_on_same_instance=False,
        n_train=1, 
        n_test=1,
        n_in=6000,
        n_out=6000,
        n_eval=6000,
        n_bound = 1024,
        input_min_sample_points=None,
        input_max_sample_points=None,
        input_subsample_level=None,
        output_subsample_level=None,
        train_out_res=None
        ):
    try:
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
            print("Dictionary loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file was not found.")
    random.shuffle(data)

    train_end = int(0.7*len(data))
    
    if n_train > train_end:
        n_train = train_end
        print('WARNING: Max n_train is 0.7 of the length of the data file. Overriding.')
    print(f"{n_train=}")
    if n_test > len(data) - train_end:
        n_test =  len(data) - train_end
    print(f"{n_test=}")

    data_list = []

    for idx, instance in enumerate(data):
        f_f = torch.tensor(instance['train_source_terms_domain'][:n_in], dtype=torch.float32)
        f_g = torch.tensor(instance['train_bc_domain'][:n_in], dtype=torch.float32)
        f_dist = torch.tensor(instance['train_distances_domain'][:n_in], dtype=torch.float32)

        input_geom = torch.tensor(instance['train_points_domain'][:n_in], dtype=torch.float32)

        if idx < n_train:
            if train_out_res:
                # Using uniform output mesh
                out_p_domain = generate_output_grid(train_out_res, instance['coefs'])
                out_source_domain = source(out_p_domain, instance['coefs']['beta'], instance['coefs']['mu_1'], instance['coefs']['mu_2'])
                y_domain = torch.ones(out_p_domain.shape[0])
                out_p_domain.requires_grad = True
            else:
                # Using randomly sampled Fenics mesh
                out_p_domain = torch.tensor(instance['val_points_domain'][:n_out], dtype=torch.float32)
                out_source_domain = torch.tensor(instance['val_source_terms_domain'][:n_out], dtype=torch.float32)
                y_domain = torch.tensor(instance['val_values_domain'][:n_out], dtype=torch.float32)
                out_p_domain.requires_grad = True

            # Give the boundary points for the output during training
            out_p_bound = torch.tensor(instance['val_points_boundary'][:n_bound], dtype=torch.float32)
            out_source_bound = torch.tensor(instance['val_source_terms_boundary'][:n_bound], dtype=torch.float32)
            y_bound = torch.tensor(instance['val_values_boundary'][:n_bound], dtype=torch.float32)

            out_p_bound.requires_grad = True
            out_source = torch.cat((out_source_bound, out_source_domain))
            #y = torch.cat((y_bound, y_domain))
        else:
            # TODO: FIX  train/eval different points passed
            out_p_bound = None
            out_p_domain = torch.tensor(instance['val_points_domain'][:n_eval], dtype=torch.float32)
            out_source_domain = torch.tensor(instance['val_source_terms_domain'][:n_eval], dtype=torch.float32)
            y_domain = torch.tensor(instance['val_values_domain'][:n_eval], dtype=torch.float32)
            y_bound = None

        f_f = torch.cat((torch.tensor(instance['train_source_terms_boundary'][:n_in], dtype=torch.float32), f_f))
        f_g = torch.cat((torch.tensor(instance['train_bc_boundary'][:n_in], dtype=torch.float32), f_g))
        f_dist = torch.cat((torch.zeros(n_in), f_dist))
        input_geom = torch.vstack((torch.tensor(instance['train_points_boundary'][:n_in], dtype=torch.float32), input_geom))

        f_f = f_f.unsqueeze(dim=-1)
        f_g = f_g.unsqueeze(dim=-1)
        f_dist = f_dist.unsqueeze(dim=-1)


        f = torch.cat((f_f, f_g, f_dist), dim=-1)
        latent_queries = generate_latent_queries(query_res=query_res,
                                                pad=domain_padding
                                                )
        
        data_dict = {'x': f, 
                    # input coords
                    'input_geom': input_geom,
                    # latent grid
                    'latent_queries': latent_queries,
                    # domain info
                    'output_queries_domain': out_p_domain,
                    'output_source_terms_domain': out_source_domain,
                    'y_domain': y_domain,
                    # boundary info
                    #'output_queries_bound': out_p_bound,
                    #'output_source_terms_bound': out_source_bound,
                    #'y_bound': y_bound,
                    #'output_source_terms': out_source,
                    'coefs': instance['coefs'],
                    'num_boundary': n_bound,
                    'out_sub_level': output_subsample_level if output_subsample_level else 1
                    }
        
        # avoid collating None for boundary values by inserting only if the tensors exist
        if y_bound is not None:
            data_dict['y_bound'] = y_bound
            data_dict['output_queries_bound'] = out_p_bound
            data_dict['output_source_terms_bound'] = out_source_bound
        data_list.append(data_dict)
    
    train_data = data_list[:n_train]

    if val_on_same_instance:
        test_data = train_data.copy()
    else:
        test_data = data_list[train_end:train_end+n_test]


    train_dataloader = DataLoader(DictDataset(train_data)) 
    test_dataloader = DataLoader(DictDataset(test_data)) 
    
    data_processor = PoissonGINODataProcessor(
        input_min=input_min_sample_points,
        input_max=input_max_sample_points,
        input_sub_level=input_subsample_level,
        output_sub_level=output_subsample_level
    )
    return train_dataloader, test_dataloader, data_processor


class PoissonGINODataProcessor(DefaultDataProcessor):
    """PoissonGINODataProcessor does the same thing
    as a DefaultDataProcessor with the addition of randomly subsampling
    points in the model's domain and codomain. Written specifically
    for the forward call signature of neuralop.models.GINO
    """
    def __init__(
        self, device='cpu', 
        in_normalizer=None, 
        out_normalizer=None, 
        positional_encoding=None, 
        input_min=100,
        input_max=1000,
        input_sub_level=None, 
        output_sub_level=None
    ):
        """A simple processor to pre/post process data before training/inferencing a model.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        positional_encoding : Processor, optional, default is None
            class that appends a positional encoding to the input
        input_sub_level : float, optional, default is None
            level at which to subsample points in the domain (between 0 and 1)
        output_sub_level : float, optional, default is None
            level at which to subsample points in the codomain (between 0 and 1)
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.positional_encoding = positional_encoding
        self.input_sub_level = input_sub_level
        if not output_sub_level:
            output_sub_level = 1
        self.output_sub_level = output_sub_level
        self.device = device
        self.input_min = input_min
        self.input_max = input_max

    def preprocess(self, data_dict, batched=True):
        # inputs of shape (_, n_in, in_dim)
        x = data_dict["x"].to(self.device)
        input_geom = data_dict["input_geom"].to(self.device)

        if input_geom.ndim == 4:
            input_geom = input_geom.squeeze(0)
        
        if x.ndim == 4:
            x = x.squeeze(0)

        if self.input_sub_level is not None:
            # Sample set percentage
            n_in = int(input_geom.shape[1] * self.input_sub_level)
        else:
            # Sample random in between range
            n_in = random.randint(self.input_min, self.input_max)
        
        input_indices = random.sample(list(range(input_geom.shape[-2])), k=n_in)
        x = x[:, input_indices, ...]
        input_geom = input_geom[:, input_indices, ...]

        # Subsample points on the output domain
        # first
        '''y = data_dict["y"].to(self.device)
        if y.ndim == 4:
            y = y.squeeze(0)'''

        if 'y_bound' in data_dict.keys():
            y_bound = data_dict["y_bound"]
            #print(f"{y_bound.shape=}")
        else:
            y_bound = None
        
        y_domain = data_dict["y_domain"]
        #print(f"{y_domain.shape=}")

        output_queries_bound = data_dict.get('output_queries_bound', None)
        output_queries_domain = data_dict['output_queries_domain']

        ## Subsample all points defined on the output domain/boundary

        n_bound = data_dict["num_boundary"].item()
        n_bound_out = n_bound * self.output_sub_level
        n_domain_out = output_queries_domain.shape[1] * self.output_sub_level

        '''if n_domain_out < 0:
            n_bound = 0
            n_bound_out = 0
            n_domain_out = output_queries.shape[1] * self.output_sub_level'''
    
        output_indices_bound = random.sample(list(range(0, n_bound)), k=int(n_bound_out))
        output_indices_domain = random.sample(list(range(0, output_queries_domain.shape[1])), k=int(n_domain_out))
        #output_indices = output_indices_bound + output_indices_domain
        '''if output_queries_bound is not None:
            print(f"{output_queries_bound.shape=}")
        print(f"{output_queries_domain.shape=}")'''

        if y_bound is not None:
            y_bound = y_bound[:, output_indices_bound]
            output_queries_bound = output_queries_bound[:, output_indices_bound]
        
        y_domain = y_domain[:, output_indices_domain]
        output_queries_domain = output_queries_domain[:, output_indices_domain]


        '''if y.shape[-2] >= max(output_indices):
            y = y[:, output_indices, ...]
            if 'y_bound' in data_dict.keys():
                y_bound = y_bound[:, ]'''


        if "output_source_terms_domain" in data_dict.keys():
            output_source_terms_domain = data_dict["output_source_terms_domain"]
            output_source_terms_domain = output_source_terms_domain[:, output_indices_domain, ...]
        else:
            output_source_terms_domain = None
        
        if "output_source_terms_bound" in data_dict.keys():
            output_source_terms_bound = data_dict["output_source_terms_bound"]
            output_source_terms_bound = output_source_terms_bound[:, output_indices_bound, ...]
        else:
            output_source_terms_bound = None
        
        if y_bound is not None:
            #y = torch.cat((y_bound, y_domain), dim=1) # concat along the point index dimension
            y_bound = y_bound.unsqueeze(-1).to(self.device)
            y_domain = y_domain.unsqueeze(-1).to(self.device)

            if self.out_normalizer is not None and self.train:
                y_bound = self.out_normalizer.transform(y_bound)
                y_domain = self.out_normalizer.transform(y_domain)

            y = {
                'boundary': y_bound, # add feature dim
                'domain': y_domain,
            }
            # load both boundaries and interior points to device before concatenating so they exist 
            # separately in the computational graph for later use in physics
            output_queries_domain = output_queries_domain.to(self.device)
            output_queries_bound = output_queries_bound.to(self.device)
            #output_queries = torch.cat((output_queries_bound, output_queries_domain), dim=1)
            output_queries = {
                'boundary': output_queries_bound,
                'domain': output_queries_domain
            }
        else:
            y = y_domain.unsqueeze(-1).to(self.device) # add feature dim
            output_queries = output_queries_domain.to(self.device)
            if self.out_normalizer is not None and self.train:
                y = self.out_normalizer.transform(y)
                
        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        

        data_dict["x"] = x

        # In eval mode, pass whole tensors instead of dicts of queries and y
        if not self.training and isinstance(y, dict):
            y = torch.cat((y['boundary'], y['domain']), dim=1)
            output_queries = torch.cat((output_queries['boundary'], output_queries['domain']), dim=1)
        data_dict["y"] = y
        data_dict["input_geom"] = input_geom.to(self.device)
        data_dict["output_queries"] = output_queries
        #data_dict["output_queries_domain"] = output_queries_domain
        #data_dict["output_queries_bound"] = output_queries_bound
        if output_source_terms_domain is  not None:
            data_dict["output_source_terms_domain"] = output_source_terms_domain.to(self.device)
        # TODO: what are these and how to use them?
        '''if 'coefs' in data_dict.keys():
            coefs = data_dict['coefs']
            print(f"{coefs=}")
            data_dict['coefs'] = data_dict['coefs'].to(self.device).squeeze(0)'''

        data_dict['latent_queries'] = data_dict['latent_queries'].to(self.device).squeeze(0)

        return data_dict

    def postprocess(self, output, data_dict):
        y = data_dict["y"]
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["y"] = y
        return output, data_dict


if __name__ == "__main__":
    train_data = load_nonlinear_poisson_pt(str(path))
    print(np.array(train_data).shape)