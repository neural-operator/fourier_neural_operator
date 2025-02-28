from abc import ABCMeta, abstractmethod

import torch
from neuralop.training.patching import MultigridPatching2D


import torch
from neuralop.training.patching import MultigridPatching2D


class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    # default wrap method
    def wrap(self, model):
        self.model = model
        return self
    
    # default train and eval methods
    def train(self, val: bool=True):
        super().train(val)
        if self.model is not None:
            self.model.train()
    
    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.eval()

    @abstractmethod
    def forward(self, x):
        pass

class DefaultDataProcessor(DataProcessor):
    """DefaultDataProcessor is a simple processor 
    to pre/post process data before training/inferencing a model.
    """
    def __init__(
        self, in_normalizer=None, out_normalizer=None
    ):
        """
        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        """preprocess a batch of data into the format
        expected in model's forward call

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        data_dict : dict
            input data dictionary with at least
            keys 'x' (inputs) and 'y' (ground truth)
        batched : bool, optional
            whether data contains a batch dim, by default True

        Returns
        -------
        dict
            preprocessed data_dict
        """
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        """postprocess model outputs and data_dict
        into format expected by training or val loss

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        output : torch.Tensor
            raw model outputs
        data_dict : dict
            dictionary containing single batch
            of data

        Returns
        -------
        out, data_dict
            postprocessed outputs and data dict
        """
        if self.out_normalizer and not self.training:
            output = self.out_normalizer.inverse_transform(output)
        return output, data_dict

    def forward(self, **data_dict):
        """forward call wraps a model
        to perform preprocessing, forward, and post-
        processing all in one call

        Returns
        -------
        output, data_dict
            postprocessed data for use in loss
        """
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output = self.postprocess(output)
        return output, data_dict


class IncrementalDataProcessor(torch.nn.Module):
    def __init__(self, 
                 in_normalizer=None, out_normalizer=None, device = 'cpu',
                 subsampling_rates=[2, 1], dataset_resolution=16, dataset_indices=[2,3], epoch_gap=10, verbose=False):
        """An incremental processor to pre/post process data before training/inferencing a model
        In particular this processor first regularizes the input resolution based on the sub_list and dataset_indices
        in the spatial domain based on a fixed number of epochs. We incrementally increase the resolution like done 
        in curriculum learning to train the model. This is useful for training models on large datasets with high
        resolution data.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        device : str, optional, default is 'cpu'
            device 'cuda' or 'cpu' where computations are performed
        subsampling_rates : list, optional, default is [2, 1]
            list of subsampling rates to use
        dataset_resolution : int, optional, default is 16
            resolution of the input data
        dataset_indices : list, optional, default is [2, 3]
            list of indices of the dataset to slice to regularize the input resolution - Spatial Dimensions
        epoch_gap : int, optional, default is 10
            number of epochs to wait before increasing the resolution
        verbose : bool, optional, default is False
            if True, print the current resolution
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = device
        self.sub_list = subsampling_rates
        self.dataset_resolution = dataset_resolution
        self.dataset_indices = dataset_indices
        self.epoch_gap = epoch_gap
        self.verbose = verbose
        self.epoch = 0
        
        self.current_index = 0
        self.current_logged_epoch = 0
        self.current_sub = self.index_to_sub_from_table(self.current_index)
        self.current_res = int(self.dataset_resolution / self.current_sub)   
        
        print(f'Original Incre Res: change index to {self.current_index}')
        print(f'Original Incre Res: change sub to {self.current_sub}')
        print(f'Original Incre Res: change res to {self.current_res}')

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self
    
    def epoch_wise_res_increase(self, epoch):
        # Update the current_sub and current_res values based on the epoch
        if epoch % self.epoch_gap == 0 and epoch != 0 and (
                self.current_logged_epoch != epoch):
            self.current_index += 1
            self.current_sub = self.index_to_sub_from_table(self.current_index)
            self.current_res = int(self.dataset_resolution / self.current_sub)
            self.current_logged_epoch = epoch

            if self.verbose:
                print(f'Incre Res Update: change index to {self.current_index}')
                print(f'Incre Res Update: change sub to {self.current_sub}')
                print(f'Incre Res Update: change res to {self.current_res}')

    def index_to_sub_from_table(self, index):
        # Get the sub value from the sub_list based on the index
        if index >= len(self.sub_list):
            return self.sub_list[-1]
        else:
            return self.sub_list[index]

    def regularize_input_res(self, x, y):
        # Regularize the input data based on the current_sub and dataset_name
        for idx in self.dataset_indices:
            indexes = torch.arange(0, x.size(idx), self.current_sub, device=self.device)
            x = x.index_select(dim=idx, index=indexes)
            y = y.index_select(dim=idx, index=indexes)
        return x, y
    
    def step(self, loss=None, epoch=None, x=None, y=None):
        if x is not None and y is not None:
            self.epoch_wise_res_increase(epoch)
            return self.regularize_input_res(x, y)
        
    def preprocess(self, data_dict, batched=True):
        x = data_dict['x'].to(self.device)
        y = data_dict['y'].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)
        
        if self.training:
            x, y = self.step(epoch=self.epoch, x=x, y=y)
        
        data_dict['x'] = x
        data_dict['y'] = y

        return data_dict 

    def postprocess(self, output, data_dict):
        y = data_dict['y']
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict['y'] = y
        return output, data_dict
    
    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict['x'])
        output = self.postprocess(output)
        return output, data_dict
    
class MGPatchingDataProcessor(DataProcessor):
    def __init__(
        self,
        model: torch.nn.Module,
        levels: int,
        padding_fraction: float,
        stitching: float,
        device: str = "cpu",
        use_distributed: bool=False,
        in_normalizer=None,
        out_normalizer=None,
    ):
        """MGPatchingDataProcessor
        Applies multigrid patching to inputs out-of-place
        with an optional output encoder/other data transform

        Parameters
        ----------
        model: nn.Module
            model to wrap in MultigridPatching2D
        levels : int
            number of multi-grid patching levels to use
        padding_fraction : float
            fraction by which to pad inputs in multigrid-patching
        stitching : bool
            whether to stitch back the output from the multi-grid patches 
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model inputs, by default None
        in_normalizer : neuralop.datasets.transforms.Transform, optional
            OutputEncoder to decode model outputs, by default None
        device : str, optional
            device 'cuda' or 'cpu' where computations are performed
        """
        super().__init__()
        self.levels = levels
        self.padding_fraction = padding_fraction
        self.stitching = stitching
        self.patcher = MultigridPatching2D(
            model=model,
            levels=self.levels,
            padding_fraction=self.padding_fraction,
            stitching=self.stitching,
            use_distributed=use_distributed,
        )
        self.device = device

        # set normalizers to none by default
        self.in_normalizer, self.out_normalizer = None, None
        if in_normalizer:
            self.in_normalizer = in_normalizer.to(self.device)
        if out_normalizer:
            self.out_normalizer = out_normalizer.to(self.device)
        self.model = model

    def to(self, device):
        self.device = device
        if self.in_normalizer:
            self.in_normalizer = self.in_normalizer.to(self.device)
        if self.out_normalizer:
            self.out_normalizer = self.out_normalizer.to(self.device)
        return self

    def preprocess(self, data_dict, batched=True):
        """
        Preprocess data assuming that if encoder exists, it has
        encoded all data during data loading

        Params
        ------

        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        batched: bool
            whether the first dimension of 'x', 'y' represents batching
        """
        data_dict = {
            k: v.to(self.device) for k, v in data_dict.items() if torch.is_tensor(v)
        }
        x, y = data_dict["x"], data_dict["y"]
        if self.in_normalizer:
            x = self.in_normalizer.transform(x)
            y = self.out_normalizer.transform(y)
        data_dict["x"], data_dict["y"] = self.patcher.patch(x, y)

        return data_dict

    def postprocess(self, out, data_dict):
        """
        Postprocess model outputs.

        Params
        ------

        data_dict: dict
            dictionary keyed with 'x', 'y' etc
            represents one batch of data input to a model
        out: torch.Tensor
            model output predictions
        """
        y = data_dict["y"]
        out, y = self.patcher.unpatch(out, y, evaluation=not self.training)

        if self.out_normalizer:
            y = self.out_normalizer.inverse_transform(y)
            out = self.out_normalizer.inverse_transform(out)

        data_dict["y"] = y

        return out, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(**data_dict)
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict
    
