from math import ceil, floor

import torch
from torch.utils.data import DataLoader
from torch_harmonics.examples import ShallowWaterSolver

def load_spherical_swe(n_train, n_tests, batch_size, test_batch_sizes,
                  train_resolution=(256, 512), test_resolutions=[(256, 512)],
                  device=torch.device('cpu')):
    """Load the Spherical Shallow Water equations Dataloader"""

    print(f'Loading train dataloader at resolution {train_resolution} '
          f'with {n_train} samples and {batch_size=}')
    train_dataset = SphericalSWEDataset(dims=train_resolution, num_examples=n_train, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, persistent_workers=False)

    test_loaders =  dict()
    for (res, n_test, test_batch_size) in zip(test_resolutions, n_tests, test_batch_sizes):
        print(f'Loading test dataloader at resolution {res} '
              f'with {n_test} samples and {test_batch_size=}')

        test_dataset = SphericalSWEDataset(dims=res, num_examples=n_test, device=device)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0, persistent_workers=False)
        test_loaders[res] = test_loader

    return train_loader, test_loaders


class SphericalSWEDataset(torch.utils.data.Dataset):
    """Custom Dataset class for PDE training data"""
    def __init__(self, dt=3600, dims=(256, 512), initial_condition='random', num_examples=32,
                 device=torch.device('cpu'), normalize=True, stream=None):
        # Caution: this is a heuristic which can break and lead to diverging results
        dt_min = 256 / dims[0] * 150
        nsteps = int(floor(dt / dt_min))

        self.num_examples = num_examples
        self.device = device
        self.stream = stream

        self.nlat = dims[0]
        self.nlon = dims[1]

        # number of solver steps used to compute the target
        self.nsteps = nsteps
        self.normalize = normalize

        lmax = ceil(self.nlat/3)
        mmax = lmax
        dt_solver = dt / float(self.nsteps)
        self.solver = ShallowWaterSolver(self.nlat, self.nlon, dt_solver, lmax=lmax, mmax=mmax, grid='equiangular').to(self.device).float()

        self.set_initial_condition(ictype=initial_condition)

        if self.normalize:
            inp0, _ = self._get_sample()
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def __len__(self):
        length = self.num_examples if self.ictype == 'random' else 1
        return length

    def set_initial_condition(self, ictype='random'):
        self.ictype = ictype
    
    def set_num_examples(self, num_examples=32):
        self.num_examples = num_examples

    def _get_sample(self):

        if self.ictype == 'random':
            inp = self.solver.random_initial_condition(mach=0.2)
        elif self.ictype == 'galewsky':
            inp = self.solver.galewsky_initial_condition()
            
        # solve pde for n steps to return the target
        tar = self.solver.timestep(inp, self.nsteps)
        inp = self.solver.spec2grid(inp)
        tar = self.solver.spec2grid(tar)        

        return inp, tar

    def __getitem__(self, index):

        with torch.inference_mode():
            with torch.no_grad():
                inp, tar = self._get_sample()

                if self.normalize:
                    inp = (inp - self.inp_mean) / torch.sqrt(self.inp_var)
                    tar = (tar - self.inp_mean) / torch.sqrt(self.inp_var)

        return {'x': inp.clone(), 'y': tar.clone()}