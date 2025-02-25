from typing import Literal
import importlib

import torch

def segment_csr(
    src: torch.Tensor,
    neighborhood_splits: torch.Tensor,
    reduce: Literal["mean", "sum"],
    eps: float=1e-7,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    If use_scatter is set to False or torch_scatter is not
    properly built, segment_csr falls back to a naive PyTorch implementation

    Note: the native version is mainly intended for running tests on 
    CPU-only GitHub CI runners to get around a versioning issue. 
    torch_scatter should be installed and built if possible. 

    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    neighborhood_splits : torch.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    eps : float
        Tiny perturbation to prevent div by zero in scaling
        neigborhoods if a particular neighborhood is empty

    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")
    
    device = src.device

    if src.ndim == 3:
        point_dim = 1
    else:
        point_dim = 0

    # if batched, shape [b, n_out, channels]
    # otherwise shape [n_out, channels]    
    n_in = src.shape[point_dim]

    n_reps = neighborhood_splits[1:] - neighborhood_splits[:-1]

    inds = torch.arange(n_in).unsqueeze(0).to(device)
    mask = (inds >= neighborhood_splits[:-1].unsqueeze(1)) & (inds < neighborhood_splits[1:].unsqueeze(1))

    # add a batch dim to the mask if src is batched
    if src.ndim == 3:
        mask = mask.unsqueeze(0)
    out = (mask.unsqueeze(-1).to(src.dtype) * src.unsqueeze(point_dim)).sum(dim=point_dim+1)

    if reduce == 'mean':
        # scale the outputs by number of reduced neighbors, add eps to avoid div by zero
        scale = (1 / (n_reps + eps)).unsqueeze(0).T
        out = out * scale

    return out
