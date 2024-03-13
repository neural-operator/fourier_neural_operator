import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..layers.recurrent_layers import RNO_layer
from ..layers.padding import DomainPadding
from ..layers.mlp import MLP
from .base_model import BaseModel

class RNO(BaseModel, name='RNO'):
    """
    N-Dimensional Recurrent Neural Operator.
    
    The RNO has an identical architecture to the finite-dimensional GRU, with 
    the exception that linear matrix-vector multiplications are replaced by  
    Fourier layers (Li et al., 2021), and for regression problems, the output 
    nonlinearity is replaced by a SELU activation.

    The operation of the GRU is as follows:
    z_t = sigmoid(W_z x + U_z h_{t-1} + b_z)
    r_t = sigmoid(W_r x + U_r h_{t-1} + b_r)
    \hat h_t = selu(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * \hat h_t,

    where * is element-wise, the b_i's are bias functions, and W_i, U_i are
    linear Fourier layers.

    Paper:
    .. [RNO] Liu-Schiaffini, M., Singer, C. E., Kovachki, N., Schneider, T., 
        Azizzadenesheli, K., & Anandkumar, A. (2023). Tipping point forecasting 
        in non-stationary dynamics on function spaces. arXiv preprint 
        arXiv:2308.08794.

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``.
    hidden_channels : int
        width of the RNO (i.e. number of channels).
    in_channels : int, optional
        Number of input channels.
    out_channels : int, optional
        Number of output channels.
    n_layers : int
        Number of RNO layers to use.
    lifting_channels : int or None
        Number of channels to use in hidden layer of lifting. If None, then
        a linear layer is used.
    projection_channels : int or None
        Number of channels to use in hidden layer of projection. If None, then
        a linear layer is used.
    residual : bool
        Whether to use residual connections in the hidden layers.
    domain_padding : float list, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'.
    output_scaling_factor : List or None
        Scaling factor of output resolution for each layer, by default None.
    fft_norm : str, optional
        by default 'forward'.
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    """
    def __init__(self, n_modes,
                hidden_channels,
                in_channels, 
                out_channels, 
                n_layers, 
                lifting_channels=None,
                projection_channels=None,
                residual=False, 
                domain_padding=None, 
                domain_padding_mode='one-sided', 
                output_scaling_factor=None,
                fft_norm='forward',  
                separable=False,
                factorization=None
                ):
        super(RNO, self).__init__()

        self.n_modes = n_modes
        self.n_dims = len(n_modes)
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        if output_scaling_factor:
            assert len(output_scaling_factor) == n_layers
            self.output_scaling_factor = output_scaling_factor
        else:
            self.output_scaling_factor = [None] * n_layers

        if domain_padding is not None and ((isinstance(domain_padding, list) and sum(domain_padding) > 0)):
                domain_padding = [0,] + domain_padding # avoid padding channel dimension
                self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode, output_scaling_factor=output_scaling_factor)
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.residual = residual

        # if lifting_channels is passed, make lifting an MLP with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(in_channels=in_channels, out_channels=self.hidden_channels, hidden_channels=self.lifting_channels, n_layers=2, n_dim=self.n_dims)
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(in_channels=in_channels, out_channels=self.hidden_channels, hidden_channels=self.hidden_channels, n_layers=1, n_dim=self.n_dims)

        module_list = [RNO_layer(n_modes, hidden_channels, return_sequences=True, output_scaling_factor=self.output_scaling_factor[i], fft_norm=fft_norm, factorization=factorization, separable=separable)
                                     for i in range(n_layers - 1)]
        module_list.append(RNO_layer(n_modes, hidden_channels, return_sequences=False, output_scaling_factor=self.output_scaling_factor[-1], fft_norm=fft_norm, factorization=factorization, separable=separable))
        self.layers = nn.ModuleList(module_list)

        # if projection_channels is passed, make lifting an MLP with a hidden layer of size lifting_channels
        if self.projection_channels:
            self.projection = MLP(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=self.projection_channels, n_layers=2, n_dim=self.n_dims)
        # otherwise, make it a linear layer
        else:
            self.projection = MLP(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=out_channels, n_layers=1, n_dim=self.n_dims)
    
    def forward(self, x, init_hidden_states=None): # h must be padded if using padding
        # x shape (batch, timesteps, dim, dom_size1, dom_size2, ..., dom_sizen)
        batch_size, timesteps = x.shape[:2]
        dim = x.shape[2]
        dom_sizes = x.shape[3 : 3 + self.n_dims]
        x_size = len(x.shape)

        if init_hidden_states is None:
            init_hidden_states = [None] * self.n_layers
        
        x = self.lifting(x.reshape(batch_size * timesteps, *x.shape[2:]))
        x = x.reshape(batch_size, timesteps, *x.shape[1:])

        if self.domain_padding:
            x = self.domain_padding.pad(x)

        final_hidden_states = []
        for i in range(self.n_layers):
            pred_x = self.layers[i](x, init_hidden_states[i])
            if i < self.n_layers - 1:
                if self.residual:
                    x = x + pred_x
                else:
                    x = pred_x
                final_hidden_states.append(x[:, -1])
            else:
                x = pred_x
                final_hidden_states.append(x)
        h = final_hidden_states[-1]

        if self.domain_padding:
            h = h.unsqueeze(1) # add dim for padding compatibility
            h = self.domain_padding.unpad(h)
            h = h.squeeze(1) # remove extraneous dim

        pred = self.projection(h)

        return pred, final_hidden_states

    def predict(self, x, num_steps): # num_steps is the number of steps ahead to predict
        output = []
        states = [None] * self.n_layers
        
        for i in range(num_steps):
            pred, states = self.forward(x, states)
            output.append(pred)
            x = pred.reshape((pred.shape[0], 1, *pred.shape[1:]))

        return torch.stack(output, dim=1)