import math
import torch
from torch.testing import assert_close

from ..data_losses import LpLoss, H1Loss, MSELoss
from ..finite_diff import central_diff_1d, central_diff_2d, central_diff_3d
from neuralop.layers.embeddings import regular_grid_nd

def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reductions='mean')
    l2_2d_sum = LpLoss(d=2, p=2, reductions='sum')
    x = torch.randn(10, 4, 4)

    abs_0 = l2_2d_mean.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # L2 w/out normalizing constant
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones, h=1.)
    assert mean_abs_l2_err.item() == 4.

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones, h=1.)
    assert sum_abs_l2_err.item() == 40.

    eps = 1e-7
    # L2 with default 1d normalizing constant 
    # result should be scaled by 2pi/(geometric mean of input dims= 4)
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() - (4. * math.pi / 2) <= eps

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() - 40. * math.pi / 2  <= eps

def test_h1loss():
    h1 = H1Loss(d=2, reductions='mean')
    x = torch.randn(10, 4, 4)

    abs_0 = h1.abs(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # H1 w/out normalizing constant, 
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_h1 = h1.abs(zeros, ones, h=1.)
    assert mean_abs_h1.item() == 4.

def test_mseloss():
    mse_2d = MSELoss(reductions='sum')
    x = torch.randn(10, 4, 4)

    abs_0 = mse_2d(x,x)
    assert abs_0.item() == 0.

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # all elem-wise differences are 1., squared and averaged = 1.
    # reduced by sum across batch = 10 * 1. = 10.
    mean_abs_mse = mse_2d(zeros, ones)
    assert mean_abs_mse.item() == 10.

def test_central_diff1d():
    # assert f(x) = x
    # has derivative 1 everywhere when boundaries are fixed
    x = torch.arange(10)
    dx = central_diff_1d(x, h=1., fix_x_bnd=True)
    assert_close(dx,torch.ones_like(dx))

def test_central_diff2d():
    grid = regular_grid_nd(resolutions=[10,10], grid_boundaries=[[0,10]] * 2)
    x = torch.stack(grid, dim=0)
    dx, dy = central_diff_2d(x, h=1., fix_x_bnd=True, fix_y_bnd=True)
    # pos encoding A[:,i,j] = [xi, yj]

    # dx[:,i,j] = f(x_i, y_j) vector valued <fx, fy>
    # dfx(coords) == 1s
    
    assert_close(dx[0], torch.ones_like(dx[0]))
    assert_close(dx[1], torch.zeros_like(dx[1]))
    
    assert_close(dy[0], torch.zeros_like(dy[0]))
    assert_close(dy[1], torch.ones_like(dy[1]))

def test_central_diff3d():
    grid = regular_grid_nd(resolutions=[10,10,10], grid_boundaries=[[0,10]] * 3)
    x = torch.stack(grid, dim=0)
    # pos encoding A[:,i,j,k] = [xi, yj, zk]
    dx, dy, dz = central_diff_3d(x, h=1., fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True)
    # dx[:,i,j,k] = f(x_i, y_j, z_k) vector valued <fx, fy, fz>
    # dfx(coords) == 1s
    
    assert_close(dx[0], torch.ones_like(dx[0]))
    assert_close(dx[1], torch.zeros_like(dx[1]))
    assert_close(dx[2], torch.zeros_like(dx[1]))
    
    assert_close(dy[0], torch.zeros_like(dy[0]))
    assert_close(dy[1], torch.ones_like(dy[1]))
    assert_close(dy[2], torch.zeros_like(dy[2]))

    assert_close(dz[0], torch.zeros_like(dz[0]))
    assert_close(dz[1], torch.zeros_like(dz[1]))
    assert_close(dz[2], torch.ones_like(dz[2]))
