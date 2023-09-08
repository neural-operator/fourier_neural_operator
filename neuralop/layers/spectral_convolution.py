import itertools
from typing import Optional, Tuple

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor

from .einsum_utils import einsum_complexhalf


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


def _contract_dense_separable(x, weight, separable=True):
    if not separable:
        raise ValueError("This function is only for separable=True")
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
    separable : bool
        whether to use the separable implementation of contraction. This arg is
        only checked when `implementation=reconstructed`.

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == "reconstructed":
        if separable:
            print("SEPARABLE")
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
            f'Got {implementation=}, expected "reconstructed" or "factorized"'
        )


class SpectralConv(nn.Module):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    n_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes
          in Fourier domain during training. Has to verify n <= N for (n, m) in
          zip(incremental_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str or None, {'tucker', 'cp', 'tt'}, default is None
        If None, a single dense weight is learned for the FNO.
        Otherwise, that weight, used for the contraction in the Fourier domain
        is learned in factorized form. In that case, `factorization` is the
        tensor factorization of the parameters weight used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor
        (vs one per layer), by default False Ignored if ``factorization is None``
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
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        incremental_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor=None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="backward",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_n_modes is half of that except in the last mode, correponding to
        # the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.order = len(n_modes)

        half_total_n_modes = [m // 2 for m in n_modes]
        self.half_total_n_modes = half_total_n_modes

        # We use half_total_n_modes to build the full weights
        # During training we can adjust incremental_n_modes which will also
        # update half_n_modes
        # So that we can train on a smaller part of the Fourier modes and total
        # weights
        self.incremental_n_modes = incremental_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if output_scaling_factor is not None:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [
                    [float(output_scaling_factor)] * len(self.n_modes)
                ] * n_layers
            elif isinstance(output_scaling_factor[0], (float, int)):
                output_scaling_factor = [
                    [s] * len(self.n_modes) for s in output_scaling_factor
                ]
        self.output_scaling_factor = output_scaling_factor

        if init_std == "auto":
            init_std = 1 / (in_channels * out_channels)
        else:
            init_std = 0.02

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor to parametrize the
        # conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got {in_channels=} and {out_channels=}",
                )
            weight_shape = (in_channels, *half_total_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *half_total_n_modes)
        self.separable = separable

        self.n_weights_per_layer = 2 ** (self.order - 1)
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_weights_per_layer * n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(self.n_weights_per_layer * n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _get_weight(self, index):
        if self.incremental_n_modes is not None:
            return self.weight[index][self.weight_slices]
        else:
            return self.weight[index]

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        if incremental_n_modes is None:
            self._incremental_n_modes = None
            self.half_n_modes = [m // 2 for m in self.n_modes]

        else:
            if isinstance(incremental_n_modes, int):
                self._incremental_n_modes = [incremental_n_modes] * len(self.n_modes)
            else:
                if len(incremental_n_modes) == len(self.n_modes):
                    self._incremental_n_modes = incremental_n_modes
                else:
                    raise ValueError(
                        f"Provided {incremental_n_modes} for actual {self.n_modes=}."
                    )
            self.weight_slices = [slice(None)] * 2 + [
                slice(None, n // 2) for n in self._incremental_n_modes
            ]
            self.half_n_modes = [m // 2 for m in self._incremental_n_modes]

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient

        # Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_fft = torch.zeros(
                [batchsize, self.out_channels, *fft_size],
                device=x.device,
                dtype=torch.chalf,
            )
        else:
            out_fft = torch.zeros(
                [batchsize, self.out_channels, *fft_size],
                device=x.device,
                dtype=torch.cfloat,
            )

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes
        # were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_n_modes[:-1]] + [
            ((None, self.half_n_modes[-1]),)
        ]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            out_fft[idx_tuple] = self._contract(
                x[idx_tuple],
                self._get_weight(self.n_weights_per_layer * indices + i),
                separable=self.separable,
            )

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        x = torch.fft.irfftn(out_fft, s=mode_sizes, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


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

    def forward(self, x):
        return self.main_conv.forward(x, self.indices)


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
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
        )
        out_fft[slices] = self._contract(
            x[slices], self._get_weight(indices), separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])

        x = torch.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SpectralConv2d(SpectralConv):
    """2D Spectral Conv, see :class:`neuralop.layers.SpectraConv` for the general case

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm)

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
            slice(self.half_n_modes[0]),  # :half_n_modes[0],
            slice(self.half_n_modes[1]),  # :half_n_modes[1]]
        )
        """Upper block (truncate high frequencies)."""
        out_fft[slices0] = self._contract(
            x[slices0], self._get_weight(2 * indices), separable=self.separable
        )

        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.half_n_modes[0], None),  # -half_n_modes[0]:,
            slice(self.half_n_modes[1]),  # ...... :half_n_modes[1]]
        )
        """Lower block"""
        out_fft[slices1] = self._contract(
            x[slices1], self._get_weight(2 * indices + 1), separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[indices][0])
            height = round(height * self.output_scaling_factor[indices][1])

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SpectralConv3d(SpectralConv):
    """3D Spectral Conv, see :class:`neuralop.layers.SpectraConv` for the general case

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def forward(self, x, indices=0):
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
            slice(self.half_n_modes[0]),  # :half_n_modes[0],
            slice(self.half_n_modes[1]),  # :half_n_modes[1],
            slice(self.half_n_modes[2]),  # :half_n_modes[2]]
        )
        """Upper block -- truncate high frequencies."""
        out_fft[slices0] = self._contract(
            x[slices0], self._get_weight(4 * indices + 0), separable=self.separable
        )

        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(self.half_n_modes[0]),  # ...... :half_n_modes[0],
            slice(-self.half_n_modes[1], None),  # -half_n_modes[1]:,
            slice(self.half_n_modes[2]),  # ...... :half_n_modes[0]]
        )
        """Low-pass filter for indices 2 & 4, and high-pass filter for index 3."""
        out_fft[slices1] = self._contract(
            x[slices1], self._get_weight(4 * indices + 1), separable=self.separable
        )

        slices2 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.half_n_modes[0], None),  # -half_n_modes[0]:,
            slice(self.half_n_modes[1]),  # ...... :half_n_modes[1],
            slice(self.half_n_modes[2]),  # ...... :half_n_modes[2]]
        )
        """Low-pass filter for indices 3 & 4, and high-pass filter for index 2."""
        out_fft[slices2] = self._contract(
            x[slices2], self._get_weight(4 * indices + 2), separable=self.separable
        )

        slices3 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.half_n_modes[0], None),  # -half_n_modes[0],
            slice(-self.half_n_modes[1], None),  # -half_n_modes[1],
            slice(self.half_n_modes[2]),  # ...... :half_n_modes[2]]
        )
        """Lower block -- low-cut filter in indices 2 & 3
        and high-cut filter in index 4."""
        out_fft[slices3] = self._contract(
            x[slices3], self._get_weight(4 * indices + 3), separable=self.separable
        )

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])
            height = round(height * self.output_scaling_factor[1])
            depth = round(depth * self.output_scaling_factor[2])

        x = torch.fft.irfftn(out_fft, s=(height, width, depth), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]
        return x
