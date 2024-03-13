"""
losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List

import torch


#Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h
    
    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]
        
    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

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
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
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

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum', fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
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
    
    def compute_terms(self, x, y, h):
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
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h
    
    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x
        
    def abs(self, x, y, h=None):
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
        
    def rel(self, x, y, h=None):
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
        
        diff = (diff**0.5)/(ynorm**0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff


    def __call__(self, y_pred, y, h=None, **kwargs):
        return self.rel(y_pred, y, h=h)


class DissipativeLoss(object):
    """Dissipative loss is used to avoid model blow-up and to encourage the model
    to learn dissipative dynamics (particularly in chaotic systems). The dissipative
    loss samples points uniformly and randomly in a spherical shell far from the
    data (between two user-specified radii: inner_radius and outer_radius) and 
    encourages that a learned model's predictions on this shell scale inwards by 
    some user-specified rule, such as linear scaling by some factor 0 < b < 1.

    Usage example: Given `model`, input `x`, and ground truth `y`.
        loss = DissipativeLoss(data_loss, loss_weight, inner_radius, outer_radius, out_channels)
        y_pred = model(x)
        x_diss = loss.sample_uniform_spherical_shell(y.shape)
        diss_pred = model(x_diss)
        print(loss(y_pred, y, pred_diss, x_diss))
        
    
    Introduced in "Learning Dissipative Dynamics in Chaotic Systems," NeurIPS 2022. 
    .. [MNO] Li, Z., Liu-Schiaffini, M., Kovachki, N., Azizzadenesheli, K., Liu, 
        B., Bhattacharya, K., ... & Anandkumar, A. (2022). Learning chaotic 
        dynamics in dissipative systems. Advances in Neural Information Processing 
        Systems, 35, 16768-16781.


    Parameters
    ----------
    data_loss : function of inputs and outputs
        Loss function to use for data (e.g., L2 loss). First argument is input; 
        second argument is target.
    loss_weight : float
        Weighting between the dissipative loss and data loss
    inner_radius : float
        Inner radius (in L2 function norm) of shell to sample from
    outer_radius : float
        Outer radius (in L2 function norm) of shell to sample from
    out_channels : int
        Dimension of input/output space (i.e., channels)
    diss_y_rule : function (default: None)
        Maps input x on a spherical shell to a desired dissipative y values. By 
        default this is a linear scaling by a factor of 1/2: x -> 0.5 * x.
    """
    def __init__(self, data_loss, loss_weight: float, inner_radius: float, outer_radius: float, out_channels: int, diss_y_rule = None):
        self.data_loss = data_loss
        self.reduce_dims = data_loss.reduce_dims
        self.reductions = data_loss.reductions

        self.y_rule = diss_y_rule # callable, maps x on spherical shell to desired dissipative y values
        self.loss_weight = loss_weight
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.out_channels = out_channels
        self.dissloss = LpLoss(d=out_channels, reduce_dims=self.reduce_dims, reductions=self.reductions)

        if diss_y_rule is None:
            self.diss_y_rule = lambda x : 0.5 * x
    
    def sample_uniform_spherical_shell(self, shape: tuple):
        """
        The input shape is (batch_size, dim1, ..., dimk, out_channels). For each of
        `batch_size` points, a function on domain of size dim1, ...., dimk is 
        sampled with norm uniformly within (inner_radius, outer_radius).

        `out_channels` is the dimension of the vector space over which the functions
        are defined.
        """
        batch_size, *spatial_shape, out_channels = shape

        n_elements = math.prod(spatial_shape)
        
        radius_correction_factor = math.sqrt(n_elements) # accounts for discretization of function space

        # Each element in batch is a function with a different radius
        sampled_radii = radius_correction_factor * torch.zeros(batch_size, 1).uniform_(self.inner_radius, self.outer_radius)
        vec = torch.randn(batch_size, n_elements) # ref: https://mathworld.wolfram.com/SpherePointPicking.html
        vec = torch.nn.functional.normalize(vec, p=2.0, dim=1)
        bcast_sampled_radii = torch.broadcast_to(sampled_radii, vec.shape) # broadcast for multiplication below

        return (bcast_sampled_radii * vec).reshape((batch_size, *spatial_shape))

    def __call__(self, y_pred, y, pred_diss, x_diss):
        """
        Arguments:
            y_pred is the model's prediction on data
            y is the ground truth data
            pred_diss is the model's prediction on x_diss
            x_diss is sampled via `sample_uniform_spherical_shell` by the user
        
        See usage details for more information.
        """
        data_loss = self.data_loss(y_pred, y)
        
        y_diss = self.diss_y_rule(x_diss)
        pred_diss = pred_diss.reshape(y_diss.shape)
        diss_loss = self.dissloss(pred_diss.reshape(-1, self.out_channels), y_diss.reshape(-1, self.out_channels))

        return data_loss + self.loss_weight * diss_loss

        
class IregularLpqLoss(torch.nn.Module):
    def __init__(self, p=2.0, q=2.0):
        super().__init__()

        self.p = 2.0
        self.q = 2.0
    
    #x, y are (n, c) or (n,)
    #vol_elm is (n,)

    def norm(self, x, vol_elm):
        if len(x.shape) > 1:
            s = torch.sum(torch.abs(x)**self.q, dim=1, keepdim=False)**(self.p/self.q)
        else:
            s = torch.abs(x)**self.p
        
        return torch.sum(s*vol_elm)**(1.0/self.p)

    def abs(self, x, y, vol_elm):
        return self.norm(x - y, vol_elm)
    
    #y is assumed y
    def rel(self, x, y, vol_elm):
        return self.abs(x, y, vol_elm)/self.norm(y, vol_elm)
    
    def forward(self, y_pred, y, vol_elm, **kwargs):
        return self.rel(y_pred, y, vol_elm)


def pressure_drag(pressure, vol_elm, inward_surface_normal, 
                  flow_direction_normal, flow_speed, 
                  reference_area, mass_density=1.0):
    
    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = torch.sum(inward_surface_normal*flow_direction_normal, dim=1, keepdim=False)
    
    return const*torch.sum(pressure*direction*vol_elm)

def friction_drag(wall_shear_stress, vol_elm, 
                  flow_direction_normal, flow_speed, 
                  reference_area, mass_density=1.0):
    
    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = torch.sum(wall_shear_stress*flow_direction_normal, dim=1, keepdim=False)

    x = torch.sum(direction*vol_elm)

    return const*torch.sum(direction*vol_elm)

def total_drag(pressure, wall_shear_stress, vol_elm, 
               inward_surface_normal, flow_direction_normal, 
               flow_speed, reference_area, mass_density=1.0):
    
    cp = pressure_drag(pressure, vol_elm, inward_surface_normal, 
                       flow_direction_normal, flow_speed, 
                       reference_area, mass_density)
    
    cf = friction_drag(wall_shear_stress, vol_elm, 
                       flow_direction_normal, flow_speed, 
                       reference_area, mass_density)
    
    return cp + cf 


class WeightedL2DragLoss(object):

    def __init__(self, mappings: dict, device: str = 'cuda'):
        """WeightedL2DragPlusLPQLoss calculates the l2 drag loss
            over the shear stress and pressure outputs of a model.

        Parameters
        ----------
        mappings: dict[tuple(Slice)]
            indices of an input tensor corresponding to above fields
        device : str, optional
            device on which to do tensor calculations, by default 'cuda'
        """
        # take in a dictionary of drag functions to be calculated on model output over output fields
        super().__init__()
        self.mappings = mappings
        self.device = device


    def __call__(self, y_pred, y, vol_elm, inward_normals, flow_normals, flow_speed, reference_area, **kwargs):
        c_pred = None
        c_truth = None
        loss = 0.
        
        stress_indices = self.mappings['wall_shear_stress']
        pred_stress = y_pred[stress_indices].view(-1,1)
        truth_stress = y[stress_indices]

        # friction drag takes padded input
        pred_stress_pad = torch.zeros((pred_stress.shape[0], 3), device=self.device)
        pred_stress_pad[:,0] = pred_stress.view(-1,)

        truth_stress_pad = torch.zeros((truth_stress.shape[0], 3), device=self.device)
        truth_stress_pad[:,0] = truth_stress.view(-1,)

        pressure_indices = self.mappings['pressure']
        pred_pressure = y_pred[pressure_indices].view(-1,1)
        truth_pressure = y[pressure_indices]

        c_pred = total_drag(pressure=pred_pressure,
                            wall_shear_stress=pred_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )
        c_truth = total_drag(pressure=truth_pressure,
                            wall_shear_stress=truth_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )

        loss += torch.abs(c_pred - c_truth) / torch.abs(c_truth)

        loss = (1.0/len(self.mappings) + 1)*loss

        return loss
