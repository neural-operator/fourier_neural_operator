"""
Tests fallback neighbor search on a small 2d grid
that was calculated manually
"""

import numpy as np
import torch
import pytest

from ..neighbor_search import native_neighbor_search, spatial_hashing_neighbor_search

# Manually-calculated CSR list of neighbors 
# in a 5x5 grid on [0,1] X [0,1] for radius=0.3

indices = [0, 1, 5, 0, 1, 2, 6, 1, 2, 3, 7, 2, 3, 4, 8,
           3, 4, 9, 0, 5, 6, 10, 1, 5, 6, 7, 11, 2, 6, 7,
           8, 12, 3, 7, 8, 9, 13, 4, 8, 9, 14, 5, 10, 11,
           15, 6, 10, 11, 12, 16, 7, 11, 12, 13, 17, 8, 12,
           13, 14, 18, 9, 13, 14, 19, 10, 15, 16, 20, 11, 15,
           16, 17, 21, 12, 16, 17, 18, 22, 13, 17, 18, 19, 23,
           14, 18, 19, 24, 15, 20, 21, 16, 20, 21, 22, 17, 21,
           22, 23, 18, 22, 23, 24, 19, 23, 24]

splits = [0, 3, 7, 11, 15, 18, 22, 27, 32, 37, 41, 45, 50,
          55, 60, 64, 68, 73, 78, 83, 87, 90, 94, 98, 102, 105]

def test_fallback_nb_search():
    mesh_grid = np.stack(np.meshgrid(*[np.linspace(0,1,5) for _ in range(2)], indexing="ij"), axis=-1)
    coords = torch.Tensor(mesh_grid.reshape(-1,2)) # reshape into n**d x d coord points
    return_dict = native_neighbor_search(data=coords, queries=coords, radius=0.3)
    
    assert return_dict['neighbors_index'].tolist() == indices
    assert return_dict['neighbors_row_splits'].tolist() == splits

def test_hashed_nb_search():
    mesh_grid = np.stack(np.meshgrid(*[np.linspace(0,1,5) for _ in range(2)], indexing="ij"), axis=-1)
    coords = torch.Tensor(mesh_grid.reshape(-1,2)) # reshape into n**d x d coord points
    return_dict = spatial_hashing_neighbor_search(data=coords, queries=coords, radius=0.3)
    
    assert return_dict['neighbors_index'].tolist() == indices
    assert return_dict['neighbors_row_splits'].tolist() == splits