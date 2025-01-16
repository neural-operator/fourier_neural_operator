import abc
import math
import numpy

import torch
import torch.nn as nn

from typing import Union, List, Optional, Tuple
from functools import partial

# import the base class from torch-harmonics
from torch_harmonics.quadrature import _precompute_grid
from torch_harmonics.convolution import (
    _compute_support_vals_isotropic,
    _compute_support_vals_anisotropic,
)

# def _compute_kernel_basis_isotropic()


def _normalize_convolution_filter_matrix(
    psi_idx,
    psi_vals,
    grid_in,
    grid_out,
    kernel_shape,
    quadrature_weights,
    transpose_normalization=False,
    eps=1e-9,
):
    """
    Discretely normalizes the convolution tensor.
    """

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-2]

    if len(kernel_shape) == 1:
        kernel_size = math.ceil(kernel_shape[0] / 2)
    elif len(kernel_shape) == 2:
        kernel_size = (kernel_shape[0] // 2) * kernel_shape[1] + kernel_shape[0] % 2

    # # reshape the indices implicitly to be ikernel, n_in, n_out
    # idx = torch.stack([psi_idx[0], psi_idx[1], psi_idx[2] // nlon_in, psi_idx[2] % nlon_in], dim=0)
    idx = psi_idx

    if transpose_normalization:
        # pre-compute the quadrature weights
        q = quadrature_weights[idx[1]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iin in range(n_in):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[2] == iin))
                # normalize, while summing also over the input longitude dimension here as this is not available for the output
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)
    else:
        # pre-compute the quadrature weights
        q = quadrature_weights[idx[2]].reshape(-1)

        # loop through dimensions which require normalization
        for ik in range(kernel_size):
            for iout in range(n_out):
                # get relevant entries
                iidx = torch.argwhere((idx[0] == ik) & (idx[1] == iout))
                # normalize
                vnorm = torch.sum(psi_vals[iidx] * q[iidx])
                psi_vals[iidx] = psi_vals[iidx] / (vnorm + eps)

    return psi_vals


def _precompute_convolution_filter_matrix(
    grid_in,
    grid_out,
    kernel_shape,
    quadrature_weights,
    normalize=True,
    radius_cutoff=0.01,
    periodic=False,
    transpose_normalization=False,
):
    """
    Precomputes the values stored in Psi, the local convolution filter matrix.
    The values are the results of a set of kernel basis "hat" functions applied to
    pairwise distances between each points on the input and output grids.

    The hat functions are the absolute differences between a squared distance and a
    multiple of the radius scaled by the kernel size.

    Assume the kernel is an array of shape ``(k0, k1)``. Then:

    If the kernel is isotropic (``k0 == k1``), the basis functions are a series of
    ``k0`` distances re-centered around multiples of the discretization size of the
    convolution's radius. If the kernel is anisotropic, the outputs of these hat
    functions are then multiplied by the outputs of another series of ``k1`` hat
    functions evaluated on the arctangents of these pairwise distances.

    Compared to the ``torch_harmonics`` routine for spherical support values, this
    function also returns the translated filters at positions
    $T^{-1}_j \omega_i = T^{-1}_j T_i \nu$, but assumes a non-periodic subset of the
    euclidean plane.
    """

    # check that input arrays are valid point clouds in 2D
    assert len(grid_in) == 2
    assert len(grid_out) == 2
    assert grid_in.shape[0] == 2
    assert grid_out.shape[0] == 2

    n_in = grid_in.shape[-1]
    n_out = grid_out.shape[-1]

    if len(kernel_shape) == 1:
        kernel_handle = partial(
            _compute_support_vals_isotropic, nr=kernel_shape[0], r_cutoff=radius_cutoff
        )
    elif len(kernel_shape) == 2:
        kernel_handle = partial(
            _compute_support_vals_anisotropic,
            nr=kernel_shape[0],
            nphi=kernel_shape[1],
            r_cutoff=radius_cutoff,
        )
    else:
        raise ValueError("kernel_shape should be either one- or two-dimensional.")

    grid_in = grid_in.reshape(2, 1, n_in)
    grid_out = grid_out.reshape(2, n_out, 1)

    diffs = grid_in - grid_out
    if periodic:
        periodic_diffs = torch.where(diffs > 0.0, diffs - 1, diffs + 1)
        diffs = torch.where(diffs.abs() < periodic_diffs.abs(), diffs, periodic_diffs)

    r = torch.sqrt(diffs[0] ** 2 + diffs[1] ** 2)
    phi = torch.arctan2(diffs[1], diffs[0]) + torch.pi

    idx, vals = kernel_handle(r, phi)

    idx = idx.permute(1, 0)

    if normalize:
        vals = _normalize_convolution_filter_matrix(
            idx,
            vals,
            grid_in,
            grid_out,
            kernel_shape,
            quadrature_weights,
            transpose_normalization=transpose_normalization,
        )

    return idx, vals


class DiscreteContinuousConv(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for DISCO convolutions, reproduced with permission
    from ``torch_harmonics.convolution``. If you use DISCO convs, please cite
    [1]_ and [2]_.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    kernel_shape : int or [int, int]
        shape of convolution kernel

        * If a single int is passed, kernel will isotropic

        * If a list of two nonequal ints are passed, kernel will be anisotropic.
    groups : int, optional
        number of groups in the convolution, default 1
    bias : bool, optional
        whether to create a separate bias parameter, default True
    transpose : bool, optional
        whether conv is a transpose conv, default False
    References
    ----------
    .. [1] : Bonev B., Kurth T., Hundt C., Pathak J., Baust M., Kashinath K., Anandkumar A.
        Spherical Neural Operators: Learning Stable Dynamics on the Sphere; arxiv:2306.03838

    .. [2] : Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int]],
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
    ):
        super().__init__()

        if isinstance(kernel_shape, int):
            self.kernel_shape = [kernel_shape]
        else:
            self.kernel_shape = kernel_shape

        if len(self.kernel_shape) == 1:
            self.kernel_size = self.kernel_shape[0]
        elif len(self.kernel_shape) == 2:
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] + 1
        else:
            raise ValueError("kernel_shape should be either one- or two-dimensional.")

        # groups
        self.groups = groups

        # weight tensor
        if in_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of input channels has to be an integer multiple of the group size"
            )
        if out_channels % self.groups != 0:
            raise ValueError(
                "Error, the number of output channels has to be an integer multiple of the group size"
            )

        self.groupsize = in_channels // self.groups

        scale = math.sqrt(1.0 / self.groupsize)

        self.weight = nn.Parameter(
            scale * torch.randn(out_channels, self.groupsize, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    @abc.abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError


class DiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on arbitrary 2d grids
    as implemented in [1]_. To evaluate continuous convolutions on a
    computer, they can be evaluated semi-discretely, where the translation
    operation is performed continuously, and the quadrature/projection is
    performed discretely on a grid [2]_. They are the main building blocks
    for local Neural Operators [1]_. Forward call expects an input of shape
    (batch_size, in_channels, n_in).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    grid_in: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        input grid in the form of a point cloud of shape (n_in, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    grid_out: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        output grid in the form of a point cloud (n_out, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    kernel_shape: Union[int, List[int]]
        kernel shape. Expects either a single integer for isotropic
        kernels or two integers for anisotropic kernels
    n_in: Tuple[int], optional
        number of input points along each dimension. Only used
        if grid_in is passed as a str. See ``torch_harmonics.quadrature``.
    n_out: Tuple[int], optional
        number of output points along each dimension. Only used
        if grid_out is passed as a str. See ``torch_harmonics.quadrature``.
    quadrature_weights: torch.Tensor, optional
        quadrature weights on the input grid
        expects a tensor of shape (n_in,)
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845

    .. [2] Ocampo J., Price M.A. , McEwen J.D.; Scalable and equivariant spherical CNNs by
        discrete-continuous (DISCO) convolutions, ICLR (2023), arXiv:2209.13603
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quadrature_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quadrature_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quadrature_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quadrature_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = radius_cutoff = 2 / float(math.sqrt(self.n_out) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quadrature_weights", quadrature_weights, persistent=False)

        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            radius_cutoff=radius_cutoff,
            periodic=periodic,
        )

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[1], idx[2]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        psi = torch.sparse_coo_tensor(
            self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in)
        )
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x n_in.
        """

        # pre-multiply x with the quadrature weights
        x = self.quadrature_weights * x

        psi = self.get_local_filter_matrix()

        # extract shape
        B, C, _ = x.shape

        # bring x into the right shape for the bmm (batch_size x channels, n_in) and pre-apply psi to x
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)
        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum(
            "bgckx,gock->bgox",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        )
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class DiscreteContinuousConvTranspose2d(DiscreteContinuousConv):
    """
    Transpose variant of discrete-continuous convolutions on arbitrary
    2d grids as implemented for [1]_. Forward call expects an input of shape
    (batch_size, in_channels, n_in).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    grid_in: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        input grid in the form of a point cloud of shape (n_in, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    grid_out: torch.Tensor or literal ``{'equidistant', 'legendre-gauss', 'equiangular', 'lobatto'}``
        output grid in the form of a point cloud (n_out, 2).
        Can also pass a string to generate a regular (tensor) grid.
        For exact options see ``torch_harmonics.quadrature``.
    kernel_shape: Union[int, List[int]]
        kernel shape. Expects either a single integer for isotropic kernels or two integers for anisotropic kernels
    n_in: Tuple[int], optional
        number of input points along each dimension. Only used
        if grid_in is passed as a str. See ``torch_harmonics.quadrature``.
    n_out: Tuple[int], optional
        number of output points along each dimension. Only used
        if grid_out is passed as a str. See ``torch_harmonics.quadrature``.
    quadrature_weights: torch.Tensor, optional
        quadrature weights on the input grid
        expects a tensor of shape (n_in,)
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_in: torch.Tensor,
        grid_out: torch.Tensor,
        kernel_shape: Union[int, List[int]],
        n_in: Optional[Tuple[int]] = None,
        n_out: Optional[Tuple[int]] = None,
        quadrature_weights: Optional[torch.Tensor] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # the instantiator supports convenience constructors for the input and output grids
        if isinstance(grid_in, torch.Tensor):
            assert isinstance(quadrature_weights, torch.Tensor)
            assert not periodic
        elif isinstance(grid_in, str):
            assert n_in is not None
            assert len(n_in) == 2
            x, wx = _precompute_grid(n_in[0], grid=grid_in, periodic=periodic)
            y, wy = _precompute_grid(n_in[1], grid=grid_in, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            wx, wy = torch.meshgrid(torch.from_numpy(wx), torch.from_numpy(wy))
            grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
            quadrature_weights = (wx * wy).reshape(-1)
        else:
            raise ValueError(f"Unknown grid input type of type {type(grid_in)}")

        if isinstance(grid_out, torch.Tensor):
            pass
        elif isinstance(grid_out, str):
            assert n_out is not None
            assert len(n_out) == 2
            x, wx = _precompute_grid(n_out[0], grid=grid_out, periodic=periodic)
            y, wy = _precompute_grid(n_out[1], grid=grid_out, periodic=periodic)
            x, y = torch.meshgrid(torch.from_numpy(x), torch.from_numpy(y))
            grid_out = torch.stack([x.reshape(-1), y.reshape(-1)])
        else:
            raise ValueError(f"Unknown grid output type of type {type(grid_out)}")

        # check that input arrays are valid point clouds in 2D
        assert len(grid_in.shape) == 2
        assert len(grid_out.shape) == 2
        assert len(quadrature_weights.shape) == 1
        assert grid_in.shape[0] == 2
        assert grid_out.shape[0] == 2

        self.n_in = grid_in.shape[-1]
        self.n_out = grid_out.shape[-1]

        # compute the cutoff radius based on the bandlimit of the input field
        # TODO: Attention - this heuristic is ad-hoc! Make sure to set it yourself!
        if radius_cutoff is None:
            radius_cutoff = 2 / float(math.sqrt(self.n_in) - 1)

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # integration weights
        self.register_buffer("quadrature_weights", quadrature_weights, persistent=False)

        # precompute the transposed tensor
        idx, vals = _precompute_convolution_filter_matrix(
            grid_out,
            grid_in,
            self.kernel_shape,
            quadrature_weights,
            radius_cutoff=radius_cutoff,
            periodic=periodic,
            transpose_normalization=True,
        )

        # to improve performance, we make psi a matrix by merging the first two dimensions
        # This has to be accounted for in the forward pass
        idx = torch.stack([idx[0] * self.n_out + idx[2], idx[1]], dim=0)

        self.register_buffer("psi_idx", idx.contiguous(), persistent=False)
        self.register_buffer("psi_vals", vals.contiguous(), persistent=False)

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        psi = torch.sparse_coo_tensor(
            self.psi_idx, self.psi_vals, size=(self.kernel_size * self.n_out, self.n_in)
        )
        return psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x n_in.
        """

        # pre-multiply x with the quadrature weights
        x = self.quadrature_weights * x

        psi = self.get_local_filter_matrix()

        # extract shape
        B, C, _ = x.shape

        # bring x into the right shape for the bmm (batch_size x channels, n_in) and pre-apply psi to x
        x = x.reshape(B * C, self.n_in).permute(1, 0).contiguous()
        x = torch.mm(psi, x)

        x = x.permute(1, 0).reshape(B, C, self.kernel_size, self.n_out)
        x = x.reshape(B, self.groups, self.groupsize, self.kernel_size, self.n_out)

        # do weight multiplication
        out = torch.einsum(
            "bgckx,gock->bgox",
            x,
            self.weight.reshape(
                self.groups, -1, self.weight.shape[1], self.weight.shape[2]
            ),
        )
        out = out.reshape(out.shape[0], -1, out.shape[-1])

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1)

        return out


class EquidistantDiscreteContinuousConv2d(DiscreteContinuousConv):
    """
    Discrete-continuous convolutions (DISCO) on equidistant 2d grids
    as implemented for [1]_. This implementation maps to 2d convolution
    kernels which makes it more efficient than the unstructured implementation
    above. Due to the mapping to an equidistant grid, the domain lengths need
    to be specified in order to compute the effective resolution and the
    corresponding cutoff radius. Forward call expects an input of shape
    (batch_size, in_channels, in_shape[0], in_shape[1]).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    in_shape: Tuple[int, int]
        shape of the (regular) input grid.
    out_shape: torch.Tensor or str
        shape of the (regular) output grid. Note that the side lengths
        of out_shape must be less than or equal to the side lengths
        of in_shape, and must be integer divisions of the corresponding
        in_shape side lengths.
    kernel_shape: Union[int, List[int]]
        kernel shape. Expects either a single integer for isotropic kernels or two integers for anisotropic kernels
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int, int],
        out_shape: Tuple[int, int],
        kernel_shape: Union[int, List[int]],
        domain_length: Optional[Tuple[float]] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)

        # to ensure compatibility with the unstructured code, only constant zero and periodic padding are supported currently
        self.padding_mode = "circular" if periodic else "zeros"

        # if domain length is not specified we use
        self.domain_length = [2, 2] if domain_length is None else domain_length

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = max(
                [self.domain_length[i] / float(out_shape[i]) for i in (0, 1)]
            )

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # compute how big the discrete kernel needs to be for the 2d convolution kernel to work
        self.psi_local_h = (
            math.floor(2 * radius_cutoff * in_shape[0] / self.domain_length[0]) + 1
        )
        self.psi_local_w = (
            math.floor(2 * radius_cutoff * in_shape[1] / self.domain_length[1]) + 1
        )

        # compute the scale_factor
        assert (in_shape[0] >= out_shape[0]) and (in_shape[0] % out_shape[0] == 0)
        self.scale_h = in_shape[0] // out_shape[0]
        assert (in_shape[1] >= out_shape[1]) and (in_shape[1] % out_shape[1] == 0)
        self.scale_w = in_shape[1] // out_shape[1]

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_h)
        y = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_w)
        x, y = torch.meshgrid(x, y)
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])

        # compute quadrature weights on the incoming grid
        self.q_weight = (
            self.domain_length[0] * self.domain_length[1] / in_shape[0] / in_shape[1]
        )
        quadrature_weights = self.q_weight * torch.ones(
            self.psi_local_h * self.psi_local_w
        )
        grid_out = torch.Tensor([[0.0], [0.0]])

        # precompute psi using conventional routines onto the local grid
        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            radius_cutoff=radius_cutoff,
            periodic=False,
            normalize=True,
        )

        # extract the local psi as a dense representation
        local_filter_matrix = torch.zeros(
            self.kernel_size, self.psi_local_h * self.psi_local_w
        )
        for ie in range(len(vals)):
            f = idx[0, ie]
            j = idx[2, ie]
            v = vals[ie]
            local_filter_matrix[f, j] = v

        # compute local version of the filter matrix
        local_filter_matrix = local_filter_matrix.reshape(
            self.kernel_size, self.psi_local_h, self.psi_local_w
        )

        self.register_buffer(
            "local_filter_matrix", local_filter_matrix, persistent=False
        )

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        return self.local_filter_matrix.permute(0, 2, 1).flip(dims=(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x in_shape[0] x in_shape[1].
        """

        kernel = torch.einsum(
            "kxy,ogk->ogxy", self.get_local_filter_matrix(), self.weight
        )
        # padding is rounded down to give the right result when even kernels are applied
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for output shape math
        h_pad = (self.psi_local_h + 1) // 2 - 1
        w_pad = (self.psi_local_w + 1) // 2 - 1
        out = nn.functional.conv2d(
            self.q_weight * x,
            kernel,
            self.bias,
            stride=[self.scale_h, self.scale_w],
            dilation=1,
            padding=[h_pad, w_pad],
            groups=self.groups,
        )

        return out


class EquidistantDiscreteContinuousConvTranspose2d(DiscreteContinuousConv):
    """
    Transpose Discrete-continuous convolutions (DISCO) on equidistant 2d grids
    as implemented for [1]_. This implementation maps to 2d convolution kernels
    which makes it more efficient than the unstructured implementation above.
    Due to the mapping to an equidistant grid, the domain lengths need to be
    specified in order to compute the effective resolution and the corresponding
    cutoff radius. Forward call expects an input of shape
    (batch_size, in_channels, in_shape[0], in_shape[1]).

    Parameters
    ----------
    in_channels: int
        input channels to DISCO convolution
    out_channels: int
        output channels of DISCO convolution
    in_shape: Tuple[int]
        shape of the (regular) input grid.
    out_shape: torch.Tensor or str
        shape of the (regular) output grid. Note that the side lengths
        of out_shape must be greater than or equal to the side lengths
        of in_shape, and must be integer multiples of the corresponding
        in_shape side lengths.
    kernel_shape: Union[int, List[int]]
        kernel shape. Expects either a single integer for isotropic kernels or two integers for anisotropic kernels
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    periodic: bool, optional
        whether the domain is periodic, by default False
    groups: int, optional
        number of groups in the convolution, by default 1
    bias: bool, optional
        whether to use a bias, by default True
    radius_cutoff: float, optional
        cutoff radius for the kernel. For a point ``x`` on the input grid,
        every point ``y`` on the output grid with ``||x - y|| <= radius_cutoff``
        will be affected by the value at ``x``.
        By default, set to 2 / sqrt(# of output points)

    References
    ----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        Neural Operators with Localized Integral and Differential Kernels;  arxiv:2402.16845
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_shape: Tuple[int],
        out_shape: Tuple[int],
        kernel_shape: Union[int, List[int]],
        domain_length: Optional[Tuple[float]] = None,
        periodic: Optional[bool] = False,
        groups: Optional[int] = 1,
        bias: Optional[bool] = True,
        radius_cutoff: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, kernel_shape, groups, bias)
        # torch ConvTranspose2d expects grouped weights stacked along the out_channels
        # shape (in_channels, out_channels/groups, h, w)
        self.weight = nn.Parameter(
            self.weight.permute(1, 0, 2).reshape(
                self.groupsize * self.groups, -1, self.weight.shape[-1]
            )
        )

        # to ensure compatibility with the unstructured code, only constant zero and periodic padding are supported currently
        self.padding_mode = "circular" if periodic else "zeros"

        # if domain length is not specified we use
        self.domain_length = [2, 2] if domain_length is None else domain_length

        # compute the cutoff radius based on the assumption that the grid is [-1, 1]^2
        # this still assumes a quadratic domain
        if radius_cutoff is None:
            radius_cutoff = max(
                [self.domain_length[i] / float(in_shape[i]) for i in (0, 1)]
            )

        if radius_cutoff <= 0.0:
            raise ValueError("Error, radius_cutoff has to be positive.")

        # compute how big the discrete kernel needs to be for the 2d convolution kernel to work
        self.psi_local_h = (
            math.floor(2 * radius_cutoff * out_shape[0] / self.domain_length[0]) + 1
        )
        self.psi_local_w = (
            math.floor(2 * radius_cutoff * out_shape[1] / self.domain_length[1]) + 1
        )

        # compute the scale_factor
        assert (in_shape[0] <= out_shape[0]) and (out_shape[0] % in_shape[0] == 0)
        self.scale_h = out_shape[0] // in_shape[0]
        assert (in_shape[1] <= out_shape[1]) and (out_shape[1] % in_shape[1] == 0)
        self.scale_w = out_shape[1] // in_shape[1]

        # psi_local is essentially the support of the hat functions evaluated locally
        x = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_h)
        y = torch.linspace(-radius_cutoff, radius_cutoff, self.psi_local_w)
        x, y = torch.meshgrid(x, y)
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        grid_out = torch.Tensor([[0.0], [0.0]])

        # compute quadrature weights on the incoming grid
        self.q_weight = (
            self.domain_length[0] * self.domain_length[1] / out_shape[0] / out_shape[1]
        )
        quadrature_weights = self.q_weight * torch.ones(
            self.psi_local_h * self.psi_local_w
        )

        # precompute psi using conventional routines onto the local grid
        idx, vals = _precompute_convolution_filter_matrix(
            grid_in,
            grid_out,
            self.kernel_shape,
            quadrature_weights,
            radius_cutoff=radius_cutoff,
            periodic=False,
            normalize=True,
            transpose_normalization=False,
        )

        # extract the local psi as a dense representation
        local_filter_matrix = torch.zeros(
            self.kernel_size, self.psi_local_h * self.psi_local_w
        )
        for ie in range(len(vals)):
            f = idx[0, ie]
            j = idx[2, ie]
            v = vals[ie]
            local_filter_matrix[f, j] = v

        # compute local version of the filter matrix
        local_filter_matrix = local_filter_matrix.reshape(
            self.kernel_size, self.psi_local_h, self.psi_local_w
        )

        self.register_buffer(
            "local_filter_matrix", local_filter_matrix, persistent=False
        )

    def get_local_filter_matrix(self):
        """
        Returns the precomputed local convolution filter matrix Psi.
        Psi parameterizes the kernel function as triangular basis functions
        evaluated on pairs of points on the convolution's input and output grids,
        such that Psi[l, i, j] is the l-th basis function evaluated on point i in
        the output grid and point j in the input grid.
        """

        return self.local_filter_matrix.permute(0, 2, 1).flip(dims=(-1, -2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call. Expects an input of shape batch_size x in_channels x in_shape[0] x in_shape[1].
        """
        kernel = torch.einsum(
            "kxy,ogk->ogxy", self.get_local_filter_matrix(), self.weight
        )

        # padding is rounded down to give the right result when even kernels are applied
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for output shape math
        h_pad = (self.psi_local_h + 1) // 2 - 1
        w_pad = (self.psi_local_w + 1) // 2 - 1
        # additional one-sided padding. See https://discuss.pytorch.org/t/question-of-2d-transpose-convolution/99419
        h_pad_out = self.scale_h - (self.psi_local_h // 2 - h_pad) - 1
        w_pad_out = self.scale_w - (self.psi_local_w // 2 - w_pad) - 1

        out = nn.functional.conv_transpose2d(
            self.q_weight * x,
            kernel,
            self.bias,
            stride=[self.scale_h, self.scale_w],
            dilation=[1, 1],
            padding=[h_pad, w_pad],
            output_padding=[h_pad_out, w_pad_out],
            groups=self.groups,
        )

        return out
