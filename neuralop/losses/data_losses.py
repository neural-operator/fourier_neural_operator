"""
data_losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List

import torch

from .finite_diff import central_diff_1d, central_diff_2d, central_diff_3d

#loss function with rel/abs Lp loss
class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions
    """
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        """

        Parameters
        ----------
        d : int, optional
            dimension of data on which to compute, by default 1
        p : int, optional
            order of L-norm, by default 2
            L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
        L : float or list, optional
            quadrature weights per dim, by default 2*math.pi
            either single scalar for each dim, or one per dim
        reduce_dims : int, optional
            dimensions across which to reduce for loss, by default 0
        reductions : str, optional
            whether to reduce each dimension above 
            by summing ('sum') or averaging ('mean')
        """
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            allowed_reductions = ["sum", "mean"]
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean',\
                f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions}"
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean',\
                        f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions[j]}"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_h(self, x):
        """uniform_h creates default normalization constants
        if none already exist.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        h : list
            list of normalization constants per-dim
        """
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        """
        reduce x across all dimensions in self.reduce_dims 
        according to self.reductions

        Params
        ------
        x: torch.Tensor
            inputs
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constants for reduction
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y, squared=True):
        """
        rel: squared relative LpLoss
        computes ||x-y||^2/||y||^2

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        squared : bool
            whether to compute squared relative loss
            by default True
        """

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm
        if squared: 
            diff = diff ** 2

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.abs(y_pred, y)

class H1Loss(object):
    """
    H1Loss provides the H1 Sobolev norm between
    two d-dimensional discretized functions
    """
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        """

        Parameters
        ----------
        d : int, optional
            dimension of input functions, by default 1
        L : int or list, optional
            quadrature weights (single or by dimension), by default 2*math.pi
        reduce_dims : int, optional
            dimensions across which to reduce for loss, by default 0
        reductions : str, optional
            whether to reduce each dimension above 
            by summing ('sum') or averaging ('mean')
        fix_x_bnd : bool, optional
            whether to fix finite difference derivative
            computation on the x boundary, by default False
        fix_y_bnd : bool, optional
            whether to fix finite difference derivative
            computation on the y boundary, by default False
        fix_z_bnd : bool, optional
            whether to fix finite difference derivative
            computation on the z boundary, by default False
        """
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    @property
    def name(self):
        return f"H1_{self.d}DLoss"
     
    def compute_terms(self, x, y, h):
        """compute_terms computes the necessary
        finite-difference derivative terms for computing
        the H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : int or list
            discretization size (single or per dim)

        Returns
        -------
        _type_
            _description_
        """
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_h(self, x):
        """uniform_h creates default normalization constants
        if none already exist.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        h : list
            list of normalization constants per-dim
        """
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h
    
    def reduce_all(self, x):
        """
        reduce x across all dimensions in self.reduce_dims 
        according to self.reductions

        Params
        ------
        x: torch.Tensor
            inputs
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x
        
    def abs(self, x, y, h=None):
        """absolute H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constant for reduction, by default None
        """
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, h=None, squared=True):
        """relative squared H1-norm
        ||x-y||^2/||y||^2

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constant for reduction, by default None
        squared : bool
            whether to compute squared relative loss, by default True
        """
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, h)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff)/(ynorm)
        if not squared: 
            diff = diff ** 0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, h=None, **kwargs):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constant for reduction, by default None
        """
        return self.rel(y_pred, y, h=h)

class PointwiseQuantileLoss(object):
    def __init__(self, alpha, reduce_dims = 0, reductions='sum'):
        """PointwiseQuantileLoss computes Quantile Loss described in [1]_

        Parameters
        ----------
        alpha : float
            value, between 0 and 1, of the proportion of points
            in the output domain expected to fall within predicted quantiles
        reduce_dims : int, optional
            dimensions to reduce when summing, by default 0
            This loss was formulated for functions with a co-domain in R, 
            so for now only 0 is supported
        reductions : str, optional
            how to apply reduction (sum or mean), by default 'sum'

        References
        -----------
        .. _[1]:
        Ma, Z., Azizzadenesheli, K., Anandkumar, A., (2024).
            Calibrated Uncertainty Quantification for Operator Learning via Conformal Prediction
            ArXiV preprint, https://arxiv.org/html/2402.01960v1
        """
        
        super().__init__()

        self.alpha = alpha

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            allowed_reductions = ["sum", "mean"]
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean',\
                f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions}"
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean',\
                        f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions[j]}"
                self.reductions = reductions

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def __call__(self, y_pred, y, eps=1e-7, **kwargs):
        """
        y_pred : torch.tensor
            predicted pointwise quantile widths
        y : torch.tensor
            true pointwise diffs (model pred - ytrue)
        """

        quantile = 1 - self.alpha
        y_abs = torch.abs(y)
        diff = y_abs - y_pred
        yscale, _ = torch.max(y_abs, dim=0)
        yscale = yscale + eps
        ptwise_loss = torch.max(quantile * diff, -(1-quantile) * diff)

        # scale pointwise loss: with prob 1-q it's weighed by q and prob q weighed by 1-q
        ptwise_loss_scaled = ptwise_loss / 2 / quantile / (1 - quantile) / yscale
        ptavg_loss = ptwise_loss_scaled.view(ptwise_loss_scaled.shape[0], -1).mean(1, keepdim=True)

        if self.reduce_dims is not None:
            loss_batch = self.reduce_all(ptavg_loss).squeeze()

        return loss_batch
    
class MSELoss(object):
    """
    MSELoss computes absolute mean-squared error between two tensors.
    """
    def __init__(self, reductions='sum'):
        super().__init__()
        allowed_reductions = ['sum', 'mean']
        assert reductions in allowed_reductions, f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions}"
        self.reductions = reductions

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, dim: List[int]=None, **kwargs):
        """MSE loss call 

        Parameters
        ----------
        y_pred : torch.Tensor
            tensor of predictions
        y : torch.Tensor
            ground truth, must be same shape as x
        dim : List[int], optional
            dimensions across which to compute MSE, by default None
        """
        if dim is None:
            dim = list(range(1, y_pred.ndim)) # no reduction across batch dim
        if self.reductions == 'sum':
            return torch.mean((y_pred - y) ** 2, dim=dim).sum() # sum of MSEs for each element
        elif self.reductions == 'mean':
            return torch.mean((y_pred - y) ** 2, dim=dim).mean()
