import torch
import torch.nn.functional as F

from torch.autograd import grad

from .data_losses import central_diff_2d


class BurgersEqnLoss(object):
    """
    Computes loss for Burgers' equation.
    """

    def __init__(self, visc=0.01, method="fdm", loss=F.mse_loss, domain_length=1.0):
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
        if self.method == "fdm":
            return self.fdm(u=y_pred)
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


class PoissonInteriorLoss(object):
    def __init__(self, method='autograd', loss=F.mse_loss, debug=False):
        super().__init__()
        self.method = method
        self.loss = loss
        self.debug = debug

        
    def finite_difference(self, output_queries_domain, u, output_source_terms_domain, num_boundary, out_sub_level, **kwargs):
        # WARNING: used when domain is a structured grid
        domain_length = 1/96

        u_x, u_y = central_diff_2d(u.reshape((out_sub_level, out_sub_level)), [domain_length, domain_length], fix_x_bnd=True, fix_y_bnd=True)

        u_xx, _ = central_diff_2d(u_x, [domain_length, domain_length], fix_x_bnd=True, fix_y_bnd=True)
        _, u_yy = central_diff_2d(u_y, [domain_length, domain_length], fix_x_bnd=True, fix_y_bnd=True)

        u_xx = u_xx.squeeze(0)
        u_yy = u_yy.squeeze(0)
        u = u.squeeze([0, -1])

        # compute LHS of the Poisson equation
        u_sq = torch.pow(u, 2)
        u_x = u_x.reshape((out_sub_level**2))
        u_y = u_y.reshape((out_sub_level**2))
        u_xx = u_xx.reshape((out_sub_level**2))
        u_yy = u_yy.reshape((out_sub_level**2))
        laplacian = (u_xx + u_yy)

        norm_grad_u = u_x ** 2 + u_y ** 2

        left_hand_side = laplacian + laplacian * 0.1 * u_sq + 0.2 * u * norm_grad_u
        output_source_terms_domain = output_source_terms_domain.squeeze(0).squeeze(-1)

        assert left_hand_side.shape == output_source_terms_domain.shape
        loss = self.loss(left_hand_side, output_source_terms_domain)
        del u_xx, u_yy, u_x, u_y, left_hand_side
        return loss
    
    def autograd(self, u, output_queries, output_source_terms_domain, **kwargs):
        '''
        Compute the nonlinear Poisson equation: ∇·((1 + 0.1u^2)∇u(x)) = f(x)
        '''
        # Make sure output queries are the right shape
        output_queries_domain = output_queries['domain']
        assert output_queries_domain.shape[-1] == 2
        assert output_queries_domain.ndim == 3
        n_domain = output_queries_domain.shape[1]
        # we only care about u defined over the interior.
        # Grab u_interior now
        u = u[:, -n_domain:, ...]
        u_prime = grad(outputs=u.sum(), inputs=output_queries_domain, create_graph=True, retain_graph=True)[0]
        # return None, norm_grad_u, None


        u_x = u_prime[:,0]
        u_y = u_prime[:,1]
        
        # compute second derivatives
        u_xx = grad(outputs=u_x.sum(), inputs=output_queries_domain, create_graph=True, retain_graph=True)[0][:,:,0]
        u_yy = grad(outputs=u_y.sum(), inputs=output_queries_domain, create_graph=True, retain_graph=True)[0][:,:,1]

        u_xx = u_xx.squeeze(0)
        u_yy = u_yy.squeeze(0)
        u_prime = u_prime.squeeze(0)
        u = u.squeeze([0, -1])

        # compute LHS of the Poisson equation
        u_sq = torch.pow(u, 2)
        laplacian = (u_xx + u_yy)
        norm_grad_u = torch.pow(u_prime, 2).sum(dim=-1)

        '''print(f"{u_sq.shape=}")
        print(f"{u_xx.shape=}")
        print(f"{u_yy.shape=}")
        print(f"{norm_grad_u.shape=}")'''
        assert u_sq.shape == u_xx.shape == u_yy.shape == norm_grad_u.shape

        left_hand_side = laplacian + laplacian * 0.1 * u_sq + 0.2 * u * norm_grad_u
        output_source_terms_domain = output_source_terms_domain.squeeze(0)

        assert left_hand_side.shape == output_source_terms_domain.shape
        loss = self.loss(left_hand_side, output_source_terms_domain)
        assert not u_prime.isnan().any()
        assert not u_yy.isnan().any()
        assert not u_xx.isnan().any()
        del u_xx, u_yy, u_x, u_y, left_hand_side

        return loss
    
    def __call__(self, y_pred, **kwargs):
        if self.method == "autograd":
            return self.autograd(u=y_pred, **kwargs)
        elif self.method == "finite_difference":
            return self.finite_difference(u=y_pred, **kwargs)
        else:
            raise NotImplementedError()

class PoissonBoundaryLoss(object):
    def __init__(self, loss=F.mse_loss, debug=False):
        super().__init__()
        self.loss = loss
        self.debug = debug
        self.counter = 0

    def __call__(self, y_pred, num_boundary, out_sub_level, y, output_queries, **kwargs):
        num_boundary = int(num_boundary.item() * out_sub_level)
        boundary_pred = y_pred.squeeze(0).squeeze(-1)[:num_boundary]
        y_bound = y.squeeze(0).squeeze(-1)[:num_boundary]
        
        assert boundary_pred.shape == y_bound.shape
        return self.loss(boundary_pred, y_bound)
    
class PoissonEqnLoss(object):
    '''
    Weighted sum of interior PDE loss and MSE on the boundary
    '''
    def __init__(self, boundary_weight, interior_weight, diff_method: str='autograd', base_loss=F.mse_loss): 
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_loss = PoissonBoundaryLoss(loss=base_loss)

        self.interior_weight = interior_weight
        self.interior_loss = PoissonInteriorLoss(method=diff_method, loss=base_loss)

    def __call__(self, out, y, **kwargs):
        interior_loss = self.interior_weight * self.interior_loss(out['domain'], **kwargs)
        bc_loss = self.boundary_weight * self.boundary_loss(out['boundary'], y=y['boundary'],  **kwargs)
        return interior_loss + bc_loss
