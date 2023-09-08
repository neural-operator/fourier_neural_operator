from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader, TensorDataset

from neuralop.datasets.npz_dataset import fetch_npz


def load_burgers(
    data_home: Union[str, Path],
    file_name: str,
    n_train,
    n_test,
    downsample_rate=8,
    batch_train=32,
    batch_test=100,
    grid=(0, 1),
) -> Tuple[DataLoader, DataLoader]:
    """Loads the 1D Burgers dataset from an ``.npz`` data file.

    Target datafile will be read like a dict with initial (i.e. input) field
    values ``a(x, t=0)`` at key "a" and final (i.e. output to learn) field
    values ``u(x, t=1)`` at key "u".

    Parameters
    ----------
    data_home : str | PathLike[str]
        Fully qualified path to a directory containing the target ``.npz`` file.
    downsample_rate : int, optional (default=8)
        How much data to skip (as in a list slice) in DataLoader when training
        an operator. Ex: ``downsample_rate=1`` learns from all data, while
        ``downsample_rate=10`` learns from 10% of the original data.

    Returns
    -------
    A tuple like ``(train_loader, test_loader)`` containing two
    ``torch.utils.data.DataLoader``s
    """
    data = fetch_npz(file_name, "", data_home)  # unused zenodo_url
    x_data: torch.Tensor = torch.tensor(data.a[:, :, ::downsample_rate])
    y_data: torch.Tensor = torch.tensor(data.u[:, :, ::downsample_rate])

    x_train = x_data[0:n_train, :, :]
    x_test = x_data[n_train : (n_train + n_test), :, :]

    y_train = y_data[0:n_train, :, :]
    y_test = y_data[n_train : (n_train + n_test), :, :]

    s = x_train.size(-1)

    if grid is not None:
        grid = torch.linspace(grid[0], grid[1], s + 1)[0:-1].view(1, -1)

        grid_train = grid.repeat(n_train, 1)
        grid_test = grid.repeat(n_test, 1)

        # ``x_data`` is already unsqueezed
        x_train = torch.cat((x_train, grid_train.unsqueeze(1)), 1)
        x_test = torch.cat((x_test, grid_test.unsqueeze(1)), 1)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_train,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_test,
        shuffle=False,
    )

    return train_loader, test_loader
