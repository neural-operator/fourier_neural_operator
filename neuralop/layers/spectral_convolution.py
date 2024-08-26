from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:])  # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    
    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    if x.dtype == torch.complex32:
        # if x is half precision, run a specialized einsum
        return einsum_complexhalf(eq, x, weight)
    else:
        return tl.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable):
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
    return x * weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order + 1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1] + rank_sym]  # in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1] + rank_sym, out_sym + rank_sym]  # in, out
    factor_syms += [xs + rank_sym for xs in x_syms[2:]]  # x, y, ...
    eq = f'{x_syms},{rank_sym},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, cp_weight.weights, *cp_weight.factors)
    else:
        return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order + 1 : 2 * order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        # x, y, ...
        factor_syms = [xs + rs for (xs, rs) in zip(x_syms[1:], core_syms)]

    else:
        core_syms = einsum_symbols[order + 1 : 2 * order + 1]
        out_syms[1] = out_sym
        factor_syms = [
            einsum_symbols[1] + core_syms[0],
            out_sym + core_syms[1],
        ]  # out, in
        # x, y, ...
        factor_syms += [xs + rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])]

    eq = f'{x_syms},{core_syms},{",".join(factor_syms)}->{"".join(out_syms)}'

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, tucker_weight.core, *tucker_weight.factors)
    else:
        return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:])  # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order])  # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order + 1 :])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i + 1]])
    eq = (
        "".join(x_syms)
        + ","
        + ",".join("".join(f) for f in tt_syms)
        + "->"
        + "".join(out_syms)
    )

    if x.dtype == torch.complex32:
        return einsum_complexhalf(eq, x, *tt_weight.factors)
    else:
        return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation="reconstructed", separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    separable: bool
        if True, performs contraction with individual tensor factors. 
        if False, 
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == "factorized":
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith("dense"):
                return _contract_dense
            elif weight.name.lower().endswith("tucker"):
                return _contract_tucker
            elif weight.name.lower().endswith("tt"):
                return _contract_tt
            elif weight.name.lower().endswith("cp"):
                return _contract_cp
            else:
                raise ValueError(f"Got unexpected factorized weight type {weight.name}")
        else:
            raise ValueError(
                f"Got unexpected weight type of class {weight.__class__.__name__}"
            )
    else:
        raise ValueError(
            f'Got implementation={implementation}, expected "reconstructed" or "factorized"'
        )


Number = Union[int, float]


class SpectralConv(BaseSpectralConv):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_modes : int or int tuple
        Number of modes to use for contraction in Fourier domain during training.
 
        .. warning::
            
            We take care of the redundancy in the Fourier modes, therefore, for an input 
            of size I_1, ..., I_N, please provide modes M_K that are I_1 < M_K <= I_N
            We will automatically keep the right amount of modes: specifically, for the 
            last mode only, if you specify M_N modes we will use M_N // 2 + 1 modes 
            as the real FFT is redundant along that last dimension.

            
        .. note::

            Provided modes should be even integers. odd numbers will be rounded to the closest even number.  

        This can be updated dynamically during training.

    max_n_modes : int tuple or None, default is None
        * If not None, **maximum** number of modes to keep in Fourier Layer, along each dim
            The number of modes (`n_modes`) cannot be increased beyond that.
        * If None, all the n_modes are used.

    separable : bool, default is True
        whether to use separable implementation of contraction
        if True, contracts factors of factorized 
        tensor weight individually
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    factorization : str or None, {'tucker', 'cp', 'tt'}, default is None
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
        Ignored if ``factorization is None``
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
        Ignored if ``factorization is None``
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the
          factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of
          the decomposition
        Ignored if ``factorization is None``
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
        Ignored if ``factorization is None``
    complex_data: bool, optional
        whether data takes on complex values in the spatial domain, by default False
        if True, uses different logic for FFT contraction and uses full FFT instead of real-valued
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        separable=False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        complex_data: bool=False,
        init_std="auto",
        fft_norm="backward",
        device=None,
        dtype=None,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.complex_data = complex_data

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        if factorization is None:
            factorization = "Dense"  # No factorization

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}

        # Create/init spectral weight tensor

        if factorization is None:
            self.weight = torch.tensor(weight_shape, dtype=torch.cfloat)
        else:
            self.weight = FactorizedTensor.new(weight_shape, rank=self.rank, 
                                     factorization=factorization, fixed_rank_modes=fixed_rank_modes,
                                     **tensor_kwargs, dtype=torch.cfloat) 
        self.weight.normal_(0, init_std)
        
        self._contract = get_contract_fun(
            self.weight, implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std * torch.randn(*(tuple([self.out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.output_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.output_scaling_factor[layer_index])]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
    
    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # the real FFT is skew-symmetric, so the last mode has a redundacy if our data is real in space 
        # As a design choice we do the operation here to avoid users dealing with the +1
        # if we use the full FFT we cannot cut off informtion from the last mode
        if not self.complex_data:
            n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(
        self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        else: 
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        
        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]

        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        if self.separable: 
            slices_w = [slice(None)] # channels
        else:
            slices_w =  [slice(None), slice(None)] # in_channels, out_channels
        if self.complex_data:
            slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        
        weight = self.weight[slices_w]

        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2
        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[weight_start_idx:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels

        if self.complex_data:
            slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts]
        else:
            slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
            slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=self.separable)

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.output_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
        
        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias

        return x


class SubConv(nn.Module):
    """Class representing one of the convolutions from the mother joint
    factorized convolution.

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to
    the same data, which is shared.
    """

    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x, **kwargs):
        return self.main_conv.forward(x, self.indices, **kwargs)

    def transform(self, x, **kwargs):
        return self.main_conv.transform(x, self.indices, **kwargs)

    @property
    def weight(self):
        return self.main_conv.get_weight(indices=self.indices)

class SpectralConv1d(SpectralConv):
    """1D Spectral Conv

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape

        x = torch.fft.rfft(x, norm=self.fft_norm)

        out_fft = torch.zeros(
            [batchsize, self.out_channels, width // 2 + 1],
            device=x.device,
            dtype=torch.cfloat,
        )
        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None, self.n_modes[0]), # :half_n_modes[0]]
        )
        out_fft[slices] = self._contract(
            x[slices], self.weight[slices], separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])

        x = torch.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[...]

        return x


class SpectralConv2d(SpectralConv):
    """2D Spectral Conv, see :class:`neuralop.layers.SpectraConv` for the general case

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def forward(self, x):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm, dim=(-2, -1))

        # The output will be of size (batch_size, self.out_channels,
        # x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1]),  #      :half_n_modes[1]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1]),  # ......      :half_n_modes[1]]
        )
        print(f'2D: {x[slices0].shape=}, {self.weight[slices0].shape=}, {self.weight.shape=}')

        """Upper block (truncate high frequencies)."""
        out_fft[slices0] = self._contract(
            x[slices0], self.weight[slices1], separable=self.separable
        )

        """Lower block"""
        out_fft[slices1] = self._contract(
            x[slices1], self.weight[slices0], separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])
            height = round(height * self.output_scaling_factor[1])

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias

        return x


class SpectralConv3d(SpectralConv):
    """3D Spectral Conv, see :class:`neuralop.layers.SpectraConv` for the general case

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def forward(self, x):
        batchsize, channels, height, width, depth = x.shape

        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[-3, -2, -1])

        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width, depth // 2 + 1],
            device=x.device,
            dtype=torch.cfloat,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1] // 2),  # :half_n_modes[1],
            slice(self.n_modes[2]),  # :half_n_modes[2]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(self.n_modes[0] // 2),  # ...... :half_n_modes[0],
            slice(-self.n_modes[1] // 2, None),  # -half_n_modes[1]:,
            slice(self.n_modes[2]),  # ......      :half_n_modes[0]]
        )
        slices2 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1] // 2),  # ...... :half_n_modes[1],
            slice(self.n_modes[2]),  # ......      :half_n_modes[2]]
        )
        slices3 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0],
            slice(-self.n_modes[1] // 2, None),  # -half_n_modes[1],
            slice(self.n_modes[2]),  # ......      :half_n_modes[2]]
        )

        """Upper block -- truncate high frequencies."""
        out_fft[slices0] = self._contract(
            x[slices0], self.weight[slices3], separable=self.separable
        )

        """Low-pass filter for indices 2 & 4, and high-pass filter for index 3."""
        out_fft[slices1] = self._contract(
            x[slices1], self.weight[slices2], separable=self.separable
        )

        """Low-pass filter for indices 3 & 4, and high-pass filter for index 2."""
        out_fft[slices2] = self._contract(
            x[slices2], self.weight[slices1], separable=self.separable
        )

        """Lower block -- low-cut filter in indices 2 & 3
        and high-cut filter in index 4."""
        out_fft[slices3] = self._contract(
            x[slices3], self.weight[slices0], separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])
            height = round(height * self.output_scaling_factor[1])
            depth = round(depth * self.output_scaling_factor[2])

        x = torch.fft.irfftn(out_fft, s=(height, width, depth), dim=[-3, -2, -1], norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias
        return x
