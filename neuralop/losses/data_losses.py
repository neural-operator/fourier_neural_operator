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
    r"""
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions. Note that 
    LpLoss always averages over the spatial dimensions.

    .. note :: 
        In function space, the Lp norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to LpLoss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of data on which to compute, by default 1
    p : int, optional
        order of L-norm, by default 2
        L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``LpLoss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 
    reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning:: 

            ``LpLoss`` always reduces over the spatial dimensions according to ``self.measure``.
            `reduction` only applies to the batch and channel dimensions.

    Examples
    --------

    ```
    """

    def __init__(self, d=1, p=2, measure=1., reduction='sum'):
        super().__init__()

        self.d = d
        self.p = p
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature

    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x

    def abs(self, x, y, quadrature=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature weights for integral
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        const = math.prod(quadrature)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/||y||

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

class H1Loss(object):
    """
    H1Loss provides the H1 Sobolev norm between
    two d-dimensional discretized functions.

    .. note :: 
        In function space, the Sobolev norm is an integral over the
        entire domain. To ensure the norm converges to the integral,
        we scale the matrix norm by quadrature weights along each spatial dimension.

        If no quadrature is passed at a call to H1Loss, we assume a regular 
        discretization and take ``1 / measure`` as the quadrature weights. 

    Parameters
    ----------
    d : int, optional
        dimension of input functions, by default 1
    measure : float or list, optional
        measure of the domain, by default 1.0
        either single scalar for each dim, or one per dim

        .. note::

        To perform quadrature, ``H1Loss`` scales ``measure`` by the size
        of each spatial dimension of ``x``, and multiplies them with 
        ||x-y||, such that the final norm is a scaled average over the spatial
        dimensions of ``x``. 

    reduction : str, optional
        whether to reduce across the batch and channel dimension
        by summing ('sum') or averaging ('mean')

        .. warning : 

            H1Loss always averages over the spatial dimensions. 
            `reduction` only applies to the batch and channel dimensions.
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
    def __init__(self, d=1, measure=1., reduction='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        
        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction

        if isinstance(measure, float):
            self.measure = [measure]*self.d
        else:
            self.measure = measure
    
    @property
    def name(self):
        return f"H1_{self.d}DLoss"
     
    def compute_terms(self, x, y, quadrature):
        """compute_terms computes the necessary
        finite-difference derivative terms for computing
        the H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : int or list
            quadrature weights

        """
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, quadrature[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, quadrature[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x
        
        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)
        
        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, quadrature, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)
        
        return dict_x, dict_y

    def uniform_quadrature(self, x):
        """
        uniform_quadrature creates quadrature weights
        scaled by the spatial size of ``x`` to ensure that 
        ``LpLoss`` computes the average over spatial dims. 

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        quadrature : list
            list of quadrature weights per-dim
        """
        quadrature = [0.0]*self.d
        for j in range(self.d, 0, -1):
            quadrature[-j] = self.measure[-j]/x.size(-j)
        
        return quadrature
    
    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
        return x
        
    def abs(self, x, y, quadrature=None):
        """absolute H1 norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
            
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        const = math.prod(quadrature)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = diff**0.5

        diff = self.reduce_all(diff).squeeze()
            
        return diff
        
    def rel(self, x, y, quadrature=None):
        """relative H1-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            quadrature constant for reduction along each dim, by default None
        """
        #Assume uniform mesh
        if quadrature is None:
            quadrature = self.uniform_quadrature(x)
        else:
            if isinstance(quadrature, float):
                quadrature = [quadrature]*self.d
        
        dict_x, dict_y = self.compute_terms(x, y, quadrature)

        diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
        ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
            ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2
        
        diff = (diff**0.5)/(ynorm**0.5)

        diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, quadrature=None, **kwargs):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        quadrature : float or list, optional
            normalization constant for reduction, by default None
        """
        return self.rel(y_pred, y, quadrature=quadrature)

class PointwiseQuantileLoss(object):
    """PointwiseQuantileLoss computes Quantile Loss described in [1]_

        Parameters
        ----------
        alpha : float
            value, between 0 and 1, of the proportion of points
            in the output domain expected to fall within predicted quantiles
        reduction : str, optional
        whether to reduce across the batch and channel dimensions
        by summing ('sum') or averaging ('mean')

        .. warning : 

            PointwiseQuantileLoss always averages over the spatial dimensions. 
            `reduction` only applies to the batch and channel dimensions.

        References
        -----------
        .. _[1] : Ma, Z., Pitt, D., Azizzadenesheli, K., Anandkumar, A., (2024).
            Calibrated Uncertainty Quantification for Operator Learning via Conformal Prediction
            TMLR 2024, https://openreview.net/pdf?id=cGpegxy12T
        """
    def __init__(self, alpha, reduction='sum'):
        
        super().__init__()

        self.alpha = alpha


        allowed_reductions = ["sum", "mean"]
        assert reduction in allowed_reductions,\
        f"error: expected `reduction` to be one of {allowed_reductions}, got {reduction}"
        self.reduction = reduction
    
    def reduce_all(self, x):
        """
        reduce x across the batch according to `self.reduction`

        Params
        ------
        x: torch.Tensor
            inputs
        """
        if self.reduction == 'sum':
            x = torch.sum(x)
        else:
            x = torch.mean(x)
        
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

        loss_batch = self.reduce_all(ptavg_loss).squeeze()

        return loss_batch
