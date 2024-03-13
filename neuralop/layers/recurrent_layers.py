import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_block import FNOBlocks

class RNO_cell(nn.Module):
    """N-Dimensional Recurrent Neural Operator cell. The RNO cell takes in an
    input and history function, and it outputs the next step of the hidden function.
    
    Paper: https://arxiv.org/abs/2308.08794 

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    width : int
        width of the RNO (i.e. number of channels)
    output_scaling_factor : int, optional
        factor by which to scale the resolution of the output
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in fno, by default 'linear'
    fft_norm : str, optional
        by default 'forward'
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    """
    def __init__(self, n_modes, width, output_scaling_factor=None, skip='linear', fft_norm='ortho', factorization=None, separable=False):
        # output_scaling_factor is provided here as an integer or float
        super(RNO_cell, self).__init__()

        self.width = width

        scaling_factor = None if not output_scaling_factor else [output_scaling_factor]

        # Some output_scaling_factors are None to super-resolution purposes. We use the hidden representation in the scaled size always (it's initialized that way),
        # so we only need to scale the dimensions of f1, f3, and f5, which act on x (original dimensionality).
        self.f1 = FNOBlocks(width, width, n_modes, output_scaling_factor=scaling_factor, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f2 = FNOBlocks(width, width, n_modes, output_scaling_factor=None, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f3 = FNOBlocks(width, width, n_modes, output_scaling_factor=scaling_factor, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f4 = FNOBlocks(width, width, n_modes, output_scaling_factor=None, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f5 = FNOBlocks(width, width, n_modes, output_scaling_factor=scaling_factor, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.f6 = FNOBlocks(width, width, n_modes, output_scaling_factor=None, fno_skip=skip, fft_norm=fft_norm, factorization=factorization, separable=separable)

        self.b1 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.))) # constant bias terms
        self.b2 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
        self.b3 = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))
    
    def forward(self, x, h):
        z = torch.sigmoid(self.f1(x) + self.f2(h) + self.b1)
        r = torch.sigmoid(self.f3(x) + self.f4(h) + self.b2)
        h_hat = F.selu(self.f5(x) + self.f6(r * h) + self.b3) # selu for regression problem

        h_next = (1. - z) * h + z * h_hat

        return h_next

class RNO_layer(nn.Module):
    """N-Dimensional Recurrent Neural Operator layer. The RNO layer extends the
    action of the RNO cell to take in some sequence of time-steps as input
    and output the next output function. 

    Paper: https://arxiv.org/abs/2308.08794 

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the RNO is inferred from ``len(n_modes)``
    width : int
        width of the RNO (i.e. number of channels)
    return_sequences : boolean, optional
        Whether to return the sequence of hidden states associated with processing
        the inputs sequence of functions.
    output_scaling_factor : int, optional
        factor by which to scale the resolution of the output
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in fno, by default 'linear'
    fft_norm : str, optional
        by default 'forward'
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    """
    def __init__(self, n_modes, width, return_sequences=False, output_scaling_factor=None, skip='linear', fft_norm='ortho', factorization=None, separable=False):
        # output_scaling_factor is an integer or float here
        super(RNO_layer, self).__init__()

        self.width = width
        self.return_sequences = return_sequences
        self.output_scaling_factor = output_scaling_factor

        self.cell = RNO_cell(n_modes, width, output_scaling_factor=output_scaling_factor, skip='linear', fft_norm=fft_norm, factorization=factorization, separable=separable)
        self.bias_h = nn.Parameter(torch.normal(torch.tensor(0.),torch.tensor(1.)))

    def forward(self, x, h=None):
        batch_size, timesteps, dim = x.shape[:3]
        dom_sizes = x.shape[3:]

        if h is None:
            h_shape = (batch_size, self.width, *dom_sizes) if not self.output_scaling_factor else (batch_size, self.width,) + tuple([int(round(self.output_scaling_factor*s)) for s in dom_sizes])
            h = torch.zeros(h_shape).to(x.device)
            h += self.bias_h

        outputs = []
        for i in range(timesteps):
            h = self.cell(x[:, i], h)
            if self.return_sequences:
                outputs.append(h)

        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            return h