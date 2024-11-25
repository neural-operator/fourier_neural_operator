import torch
import torch.nn.functional as F

from .finite_diff import central_diff_2d
from .data_losses import LpLoss

class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method="finite_difference", loss=F.mse_loss, domain_length=1.0):
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def fdm(self, u):
        # remove extra channel dimensions
        u = u.squeeze(1)

        # shapes
        _, nt, nx = u.shape

        # we assume that the input is given on a regular grid
        dt = self.domain_length[0] / (nt - 1)
        dx = self.domain_length[1] / nx

        # du/dt and du/dx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # d^2u/dxx
        dudxx = (
            torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)
        ) / dx**2
        # fix boundary
        dudxx[..., 0] = (u[..., 2] - 2 * u[..., 1] + u[..., 0]) / dx**2
        dudxx[..., -1] = (u[..., -1] - 2 * u[..., -2] + u[..., -3]) / dx**2

        # right hand side
        right_hand_side = -dudx * u + self.visc * dudxx

        # compute the loss of the left and right hand sides of Burgers' equation
        return self.loss(dudt, right_hand_side)
    

    def __call__(self, y_pred, **kwargs):
        if self.method == "finite_difference":
            return self.fdm(u=y_pred)
        raise NotImplementedError()
    

class BCLoss(object):
    """
    Computes loss for boundary value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def boundary_condition_loss(self, u):
        u = u[:, 0, :, :] # squeeze 1-channel output
        size = u.shape[1]
        index_x = torch.cat([torch.tensor(range(0, size)), (size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)),
                         torch.zeros(size)], dim=0).long()
        index_y = torch.cat([(size - 1) * torch.ones(size), torch.tensor(range(size-1, 1, -1)), torch.zeros(size),
                            torch.tensor(range(0, size))], dim=0).long()

        boundary_u = u[:, index_x, index_y]
        truth_u = torch.zeros(boundary_u.shape, device=u.device)
        return self.loss(boundary_u, truth_u)

    def __call__(self, u, **kwargs):
        return self.boundary_condition_loss(u)

class DarcyEqnLoss(object):
    """
    Darcy-Flow PINO loss. Currently only finite-difference method
    is implemented (stay tuned!)
    """
    def __init__(self, method):
        """

        Parameters
        ----------
        method : str
            method to use to compute PINO loss
            "finite_difference" only for now
        """
        super().__init__()
        self.method = method
        self.__name__ = 'eqn'

    def finite_difference(self, a, u, domain_length=1):
        # remove extra channel dimensions
        a = a[:, 0, :, :]
        
        u = u[:, 0, :, :]

        # compute the left hand side of the Darcy Flow equation
        # note: here we assume that the input is a regular grid
        n = u.size(1)
        dx = domain_length / (n - 1)
        dy = dx
        ux, uy = central_diff_2d(u, [dx,dy], fix_x_bnd=False, fix_y_bnd=False)
        a_ux = a * ux
        a_uy = a * uy

        a_uxx, _ = central_diff_2d(a_ux, [dx,dy], fix_x_bnd=True, fix_y_bnd=True)
        _, a_uyy = central_diff_2d(a_uy, [dx,dy], fix_x_bnd=True, fix_y_bnd=True)
       
        left_hand_side =  -(a_uxx + a_uyy)
        left_hand_side = left_hand_side[:, 2:-2, 2:-2]

        # compute the Lp loss of the left and right hand sides of the Darcy Flow equation
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        lploss = LpLoss(d=2, L=1., reductions='sum')
        loss = lploss(left_hand_side, forcing_fn)

        del ux, uy, a_ux, a_uy, a_uxx, a_uyy
        return loss

    def __call__(self, u, x, out_p=None, **kwargs):
        """DarcyEqnLoss forward call

        Parameters
        ----------
        u : torch.Tensor
            output function representing flow through medium
        x : torch.Tensor
            input function representing permeability of medium
        out_p : torch.Tensor, optional
            output queries for more advanced PINO methods, by default None

        Returns
        -------
        torch.Tensor
            loss tensor
        """
        if self.method == 'finite_difference':
            return self.finite_difference(x, u)
        else:
            raise NotImplementedError()
    
class NavierStokes2dVorticityEqnLoss(object):
    def __init__(self, method="finite_difference"):
        super().__init__()
        self.method = method
    
    def finite_difference(self, w, v=1/40, t_interval=1.0):
        batchsize = w.size(0)
        nx = w.size(1)
        ny = w.size(2)
        nt = w.size(3)
        device = w.device
        w = w.reshape(batchsize, nx, ny, nt)

        w_h = torch.fft.fft2(w, dim=[1, 2])
        # Wavenumbers in y-direction
        k_max = nx//2
        N = nx
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                        torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
        # Negative Laplacian in Fourier space
        lap = (k_x ** 2 + k_y ** 2)
        lap[0, 0, 0, 0] = 1.0
        f_h = w_h / lap

        ux_h = 1j * k_y * f_h
        uy_h = -1j * k_x * f_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
        uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
        wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
        wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
        wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

        dt = t_interval / (nt-1)
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

        Du1 = wt + (ux*wx + uy*wy - v*wlap)[...,1:-1] #- forcing
        return Du1

    def __call__(self, u, x, out_p=None, **kwargs):
        """NavierStokes2dVorticityEqnLoss forward call

        Parameters
        ----------
        u : torch.Tensor
            output function representing vorticity at time t
        x : torch.Tensor
            input function representing vorticity at time 0
        out_p : torch.Tensor, optional
            output queries for more advanced PINO methods, by default None

        Returns
        -------
        torch.Tensor
            loss tensor
        """
        if self.method == 'finite_difference':
            return self.finite_difference(x, u)
        else:
            raise NotImplementedError()


class ICLoss(object):
    """
    Computes loss for initial value problems.
    """

    def __init__(self, loss=F.mse_loss):
        super().__init__()
        self.loss = loss

    def initial_condition_loss(self, y_pred, x):
        boundary_true = x[:, 0, 0, :]
        boundary_pred = y_pred[:, 0, 0, :]
        return self.loss(boundary_pred, boundary_true)

    def __call__(self, y_pred, x, **kwargs):
        return self.initial_condition_loss(y_pred, x)
