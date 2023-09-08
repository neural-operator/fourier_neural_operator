import torch
import zarr
from torch.utils.data import Dataset


class ZarrDataset(Dataset):
    """PDE h5py dataset"""

    def __init__(
        self,
        filename,
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
        self.filename = str(filename)

        self._data = None
        self.transform_x = transform_x
        self.transform_y = transform_y

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            data = zarr.open(self.filename, mode="r")
            self.n_samples = data.shape[0]
            del data

    def attrs(self, array_name, name):
        data = zarr.open(
            self.filename, mode="r", synchronizer=zarr.ThreadSynchronizer()
        )
        value = data[array_name].attrs[name]
        del data
        return value

    @property
    def data(self):
        if self._data is None:
            self._data = zarr.open(
                self.filename, mode="r", synchronizer=zarr.ThreadSynchronizer()
            )
        return self._data

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
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {"x": x, "y": y}

    def __getitems__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(
            [
                self.data["x"][i, :: self.subsample_step, :: self.subsample_step]
                for i in idx
            ],
            dtype=torch.float32,
        )
        y = torch.tensor(
            [
                self.data["y"][i, :: self.subsample_step, :: self.subsample_step]
                for i in idx
            ],
            dtype=torch.float32,
        )

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {"x": x, "y": y}
