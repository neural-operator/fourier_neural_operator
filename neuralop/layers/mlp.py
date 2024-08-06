import torch
from torch import nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


# Reimplementation of the MLP class using Linear instead of Conv
class MLPLinear(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x


class Sine(nn.Module):
    """
    Sine activation function with a scalar scaling factor
    """
    def __init__(self, omega0=1.):
        super().__init__()
        self.omega0 = omega0

    def forward(self, x):
        return torch.sin(self.omega0 * x)


class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 omega0=1.,
                 c=6.,
                 is_first=False,
                 use_bias=True,
                 activation=None):
        """
            code modified from:
                 https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
            SIREN paper: https://arxiv.org/abs/2006.09661

            The Siren layer is a linear layer followed by a sine activation function.
            The initialization of the linear layer's weight follows the description Siren's official implementation.

            Parameters:
                dim_in: int, Number of input channels.
                dim_out: int, Number of output channels.
                omega0: float, scalar scaling factor used to modulate the features before sine activation function.
                        It is used to accelerate the convergence of the network (see Appendix Section 1.6 of Siren paper)
                c: float, scaling factor (numerator) used to initialize the weights, by default 6.
                is_first: bool, Whether this is the first layer of the network, by default False.
                use_bias: bool, Whether to use bias or not, by default True.
                activation: nn.Module, Activation function to use, by default None (which uses Sine activation).

        """

        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.use_bias = use_bias

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.init_(c=c, omega0=omega0)

        self.activation = Sine(omega0) if activation is None else activation

    def init_(self, c, omega0):
        """
        Initialize the weights heuristically, which usually results in faster convergence
        w_i is draw from a uniform distribution U(-std, std) where std = (1 / dim) if is_first
            else (sqrt(c / dim) / omega0)
        (Reference: Siren's official demo, https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb)

        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / omega0)
        torch.nn.init.uniform_(self.linear.weight, -w_std, w_std)

        if self.use_bias:
            torch.nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out


class SirenMLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 num_layers,
                 omega0=1.,
                 omega0_initial=30.,
                 use_bias=True,
                 final_activation=None,
                 rescale_input=True):
        """
            A MLP network with Siren layers.

            Parameters:
                dim_in: int, Number of input channels.
                dim_hidden: int, Number of hidden channels.
                dim_out: int, Number of output channels.
                num_layers: int, Number of layers in the network.
                omega0: float, scaling factor before activation, applied to each layer except for the first layer, by default 1.
                omega0_initial: float, scaling factor before activation, applied to the very first layer, by default 30.
                use_bias: bool, Whether to use bias or not, by default True.
                final_activation: nn.Module, Activation function to use in the final layer, by default None.
                rescale_input: bool, Whether to rescale the input from [0, 1] to [-1, 1], by default True.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.rescale_input = rescale_input

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0

            # for the first layer, use a scalar scaling omega0 30
            # as recommended in page 5 of the Siren paper
            layer_omega0 = omega0_initial if is_first else omega0

            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                omega0=layer_omega0,
                use_bias=use_bias,
                is_first=is_first,
            ))

        self.final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = Siren(dim_in=dim_hidden,
                                dim_out=dim_out,
                                omega0=omega0,
                                use_bias=use_bias,
                                activation=final_activation)

    def forward(self, x):
        if self.rescale_input:
            x = 2.0 * x - 1.0  # assume x is in [0, 1], the rescaling gives slightly better performance

        for layer in self.layers:
            x = layer(x)

        x = self.last_layer(x)
        x = self.final_activation(x)
        return x
