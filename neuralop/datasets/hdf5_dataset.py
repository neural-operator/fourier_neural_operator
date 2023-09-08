import h5py
import torch
from torch.utils.data import Dataset


class H5pyDataset(Dataset):
    """PDE h5py dataset"""

    def __init__(
        self,
        data_path,
        resolution=128,
        transform_x=None,
        transform_y=None,
        n_samples=None,
    ):
        resolution_to_step = {128: 8, 256: 4, 512: 2, 1024: 1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(
                f"Got {resolution=}, expected one of {resolution_to_step.keys()}"
            )

        self.subsample_step = subsample_step
        self.data_path = data_path
        self._data = None
        self.transform_x = transform_x
        self.transform_y = transform_y

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            with h5py.File(str(self.data_path), "r") as f:
                self.n_samples = f["x"].shape[0]

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(str(self.data_path), "r")
        return self._data

    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            assert (
                idx < self.n_samples
            ), f"Trying to access sample {idx} of dataset with {self.n_samples} samples"
        else:
            for i in idx:
                assert (
                    i < self.n_samples
                ), f"Trying to access sample {i} " \
                   f"of dataset with {self.n_samples} samples"

        x = self.data["x"][idx, :: self.subsample_step, :: self.subsample_step]
        y = self.data["y"][idx, :: self.subsample_step, :: self.subsample_step]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {"x": x, "y": y}
