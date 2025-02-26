from typing import List, Optional, Union
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP
from .fno_block import SubModule
from .differential_conv import FiniteDifferenceConvolution
try:
    from .discrete_continuous_convolution import EquidistantDiscreteContinuousConv2d
    DISCO_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    DISCO_AVAILABLE = False
    EquidistantDiscreteContinuousConv2d = None
from .normalization_layers import AdaIN, InstanceNorm
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class LocalNOBlocks(nn.Module):
    """LocalNOBlocks implements a sequence of Fourier layers, the operations of which 
    are first described in [1]_. The exact implementation details of the Fourier 
    layer architecture are discussed in [2]_. The Fourier layers are placed in parallel 
    with differential kernel layers and local integral layers, as described in [3]_.

    Parameters
    ----------
    in_channels : int
        input channels to Fourier layers
    out_channels : int
        output channels after Fourier layers
    n_modes : int, List[int]
        number of modes to keep along each dimension 
        in frequency space. Can either be specified as
        an int (for all dimensions) or an iterable with one
        number per dimension
    default_in_shape : Tuple[int]
        Default input shape on spatiotemporal dimensions.
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        number of Fourier layers to apply in sequence, by default 1
    disco_layers : bool or bool list, optional
        Must be same length as n_layers, dictates whether to include a
        local integral kernel parallel connection at each layer. If a single
        bool, shared for all layers.
    disco_kernel_shape: Union[int, List[int]]
        kernel shape for local integral. Expects either a single integer for isotropic kernels or two integers for anisotropic kernels
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    disco_groups: int, optional
        number of groups in the local integral convolution, by default 1
    disco_bias: bool, optional
        whether to use a bias for the integral kernel, by default True
    radius_cutoff: float, optional
        cutoff radius (with respect to domain_length) for the local integral kernel, by default None
    diff_layers : bool or bool list, optional
        Must be same length as n_layers, dictates whether to include a
        differential kernel parallel connection at each layer. If a single
        bool, shared for all layers.
    conv_padding_mode : str in ['periodic', 'circular', 'replicate', 'reflect', 'zeros'], optional
        Padding mode for spatial convolution kernels.
    fin_diff_kernel_size : odd int, optional
        Conv kernel size for finite difference convolution.
    mix_derivatives : bool, optional
        Whether to mix derivatives across channels.
    max_n_modes : int, List[int], optional
        maximum number of modes to keep along each dimension, by default None
    local_no_block_precision : str, optional
        floating point precision to use for computations, by default "full"
    channel_mlp_dropout : int, optional
        dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        stabilizing module to use between certain layers, by default None
        if "tanh", use tanh
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use, by default None
    ada_in_features : int, optional
        number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        whether to call forward pass with pre-activation, by default False
        if True, call nonlinear activation and norm before Fourier convolution
        if False, call activation and norms after Fourier convolutions
    local_no_skip : str, optional
        module to use for Local NO skip connections, by default "linear"
        see layers.skip_connections for more details
    channel_mlp_skip : str, optional
        module to use for ChannelMLP skip connections, by default "soft-gating"
        see layers.skip_connections for more details

    Other Parameters
    -------------------
    complex_data : bool, optional
        whether the Local NO's data takes on complex values in space, by default False
    separable : bool, optional
        separable parameter for SpectralConv, by default False
    factorization : str, optional
        factorization parameter for SpectralConv, by default None
    rank : float, optional
        rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        module to use for convolutions in Local NO block, by default SpectralConv
    joint_factorization : bool, optional
        whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        implementation parameter for SpectralConv, by default "factorized"
    decomposition_kwargs : _type_, optional
        kwargs for tensor decomposition in SpectralConv, by default dict()
    
    References
    -----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential 
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024). 
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    .. [3] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
           "Neural Operators with Localized Integral and Differential Kernels" (2024).  
           ICML 2024, https://arxiv.org/pdf/2402.16845.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        default_in_shape,
        resolution_scaling_factor=None,
        n_layers=1,
        disco_layers=True,
        disco_kernel_shape=[2,4],
        radius_cutoff=None,
        domain_length=[2,2],
        disco_groups=1,
        disco_bias=True,
        diff_layers=True,
        conv_padding_mode='periodic',
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        max_n_modes=None,
        local_no_block_precision="full",
        use_channel_mlp=False,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        local_no_skip="linear",
        channel_mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes

        assert len(n_modes) == len(default_in_shape), "Spatiotemporal dimensions must be consistent"
        
        # If a single bool is passed for disco_layers or diff_layers, set values for all layers
        if isinstance(disco_layers, bool):
            disco_layers = [disco_layers] * n_layers
        if isinstance(diff_layers, bool):
            diff_layers = [diff_layers] * n_layers
        
        if len(n_modes) > 3 and True in diff_layers:
            NotImplementedError("Differential convs not implemented for dimensions higher than 3.")

        if len(n_modes) != 2 and True in disco_layers:
            NotImplementedError("Local conv layers only implemented for dimension 2.")
            
        if conv_padding_mode not in ['circular', 'periodic', 'zeros'] and True in disco_layers:
            warnings.warn("Local conv layers only support periodic or zero padding, defaulting to zero padding for local convs.")
        
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.local_no_block_precision = local_no_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.local_no_skip = local_no_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        self.diff_layers = diff_layers
        self.conv_padding_mode = conv_padding_mode
        self.default_in_shape = default_in_shape
        self.fin_diff_kernel_size = fin_diff_kernel_size
        self.mix_derivatives = mix_derivatives

        self.disco_layers = disco_layers
        self.disco_kernel_shape = disco_kernel_shape
        self.radius_cutoff = radius_cutoff
        self.domain_length = domain_length
        self.disco_groups = disco_groups
        self.disco_bias = disco_bias
        self.periodic = (self.conv_padding_mode in ['circular', 'periodic'])

        assert len(diff_layers) == n_layers,\
            f"diff_layers must either provide a single bool value or a list of booleans of length n_layers,\
                got {len(diff_layers)=}"
        assert len(disco_layers) == n_layers,\
            f"disco_layers must either provide a single bool value or a list of booleans of length n_layers,\
                    got {len(disco_layers)=}"

        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    resolution_scaling_factor=None if resolution_scaling_factor is None else self.resolution_scaling_factor[i],
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                ) for i in range(n_layers)
            ]
        )

        self.local_no_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=local_no_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.diff_groups = 1 if mix_derivatives else in_channels
        self.differential = nn.ModuleList(
            [
                FiniteDifferenceConvolution(self.in_channels, self.out_channels,
                                            self.n_dim, self.fin_diff_kernel_size, 
                                            self.diff_groups, self.conv_padding_mode)
                for _ in range(sum(self.diff_layers))
            ]
        )

        self.local_convs = nn.ModuleList(
            [
                EquidistantDiscreteContinuousConv2d(self.in_channels, self.out_channels, in_shape=self.default_in_shape, out_shape=self.default_in_shape,
                                                    kernel_shape=self.disco_kernel_shape, domain_length=self.domain_length,
                                                    radius_cutoff=self.radius_cutoff, periodic=self.periodic, groups=self.disco_groups, bias=self.disco_bias)
                for _ in range(sum(self.disco_layers))
            ]
        )

        # Helper for calling differential layers
        self.differential_idx_list = []
        j = 0
        for i in range(n_layers):
            if self.diff_layers[i]:
                self.differential_idx_list.append(j)
                j += 1
            else:
                self.differential_idx_list.append(-1)

        assert max(self.differential_idx_list) == sum(self.diff_layers) - 1

        # Helper for calling local conv layers
        self.disco_idx_list = []
        j = 0
        for i in range(n_layers):
            if self.disco_layers[i]:
                self.disco_idx_list.append(j)
                j += 1
            else:
                self.disco_idx_list.append(-1)

        assert max(self.disco_idx_list) == sum(self.disco_layers) - 1

        if use_channel_mlp:
            self.mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.channel_mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=channel_mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                    [
                        InstanceNorm()
                        for _ in range(n_layers * self.n_norms)
                    ]
                )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

        if 'use_disco' in kwargs and kwargs['use_disco']:
            if not DISCO_AVAILABLE:
                raise ImportError(
                    "DISCO convolutions requested but torch-harmonics is not available. "
                    "Please install with `pip install torch-harmonics` to use DISCO convolutions."
                )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_local_no = self.local_no_skips[index](x)
        x_skip_local_no = self.convs[index].transform(x_skip_local_no, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.channel_mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_local_no = self.convs[index](x, output_shape=output_shape)

        if self.differential_idx_list[index] != -1:
            grid_width_scaling_factor = 1 / (x.shape[-1] / self.default_in_shape[0])
            x_differential = self.differential[self.differential_idx_list[index]](x, grid_width_scaling_factor)
            x_differential = self.convs[index].transform(x_differential, output_shape=output_shape)
        else:
            x_differential = 0

        if self.disco_idx_list[index] != -1:
            x_localconv = self.local_convs[self.disco_idx_list[index]](x)
            x_localconv = self.convs[index].transform(x_localconv, output_shape=output_shape)
        else:
            x_localconv = 0

        x_local_no_diff_disco = x_local_no + x_differential + x_localconv

        if self.norm is not None:
            x_local_no_diff_disco = self.norm[self.n_norms * index](x_local_no_diff_disco)

        x = x_local_no_diff_disco + x_skip_local_no

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        if self.differential_idx_list[index] != -1:
            grid_width_scaling_factor = 1 / (x.shape[-1] / self.default_grid_res)
            x_differential = self.differential[self.differential_idx_list[index]](x, grid_width_scaling_factor)
        else:
            x_differential = 0

        if self.disco_idx_list[index] != -1:
            x_localconv = self.local_convs[self.disco_idx_list[index]](x)
        else:
            x_localconv = 0

        x_skip_local_no = self.local_no_skips[index](x)
        x_skip_local_local_no = self.convs[index].transform(x_skip_local_no + x_differential + x_localconv, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.channel_mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_local_no = self.convs[index](x, output_shape=output_shape)

        x = x_local_no + x_skip_local_local_no

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-NO Block layer from the jointly parametrized main block

        The parametrization of an LocalNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)