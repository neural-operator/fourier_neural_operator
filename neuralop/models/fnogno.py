import torch
import torch.nn.functional as F

from .base_model import BaseModel

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.gno_block import GNOBlock
from ..layers.gno_weighting_functions import dispatch_weighting_fn

class FNOGNO(BaseModel, name="FNOGNO"):
    """FNOGNO: Fourier/Geometry Neural Operator. The FNOGNO
    maps from a regular N-d grid to an arbitrary query point cloud.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    projection_channels : int, defaults to 256
         number of hidden channels in embedding block of FNO.
    gno_coord_dim : int, defaults to 3
        dimension of coordinate space where GNO is computed
    gno_pos_embed_type : literal `{'transformer', 'nerf'}` | None
        type of optional sinusoidal positional embedding to use in GNOBlock,
        by default `'transformer'`
    gno_radius : float, defaults to 0.033
        radius parameter to construct graph.
    gno_transform_type : str, defaults to 'linear'
        type of kernel integral transform to apply in GNO.
        kernel k(x,y): parameterized as ChannelMLP MLP integrated over a neighborhood of x
        options: 'linear_kernelonly': integrand is k(x, y)
                    'linear' : integrand is k(x, y) * f(y)
                    'nonlinear_kernelonly' : integrand is k(x, y, f(y))
                    'nonlinear' : integrand is k(x, y, f(y)) * f(y)
     gno_weighting_function : str, optional
        Choice of weighting function to use in the output GNO for 
        Mollified Graph Neural Operator-based models
    gno_weight_function_scale : float, optional
        Factor by which to scale weights from GNO weighting function
        by default 1
    fno_n_modes : tuple, defaults to (16, 16, 16)
        number of modes to keep along each spectral dimension of FNO block
    fno_hidden_channels : int, defaults to 64
        number of hidden channels of fno block.
    fno_lifting_channel_ratio : int, defaults to 4
        ratio of lifting channels to FNO hidden channels.
    fno_n_layers : int, defaults to 4
        number of FNO layers in the block.

    Other Parameters
    ----------------
    gno_embed_channels: int
        dimension of optional per-channel embedding to use in GNOBlock,
        by default 32
    gno_embed_max_positions: int
        max positions of optional per-channel embedding to use in GNOBlock,
        by default 10000. If `gno_pos_embed_type != 'transformer'`, value is unused.
    gno_channel_mlp_hidden_layers : list, defaults to [512, 256]
        dimension of hidden ChannelMLP layers of GNO.
    gno_channel_mlp_non_linearity : nn.Module, defaults to F.gelu
        nonlinear activation function between layers
    gno_use_open3d : bool, defaults to True
        whether to use Open3D functionality
        if False, uses simple fallback neighbor search
    gno_batched: bool, defaults to False
        whether to use IntegralTransform/GNO layer in
        "batched" mode. If False, sets batched=False.
    fno_resolution_scaling_factor : float | None, defaults to None
        factor by which to rescale output predictions in the original domain
    fno_incremental_n_modes : list[int] | None, defaults to None
        if passed, sets n_modes separately for each FNO layer.
    fno_block_precision : str, defaults to 'full'
        data precision to compute within fno block
    fno_use_channel_mlp : bool, defaults to False
        Whether to use a ChannelMLP layer after each FNO block.
    fno_channel_mlp_dropout : float, defaults to 0
        dropout parameter of above ChannelMLP.
    fno_channel_mlp_expansion : float, defaults to 0.5
        expansion parameter of above ChannelMLP.
    fno_non_linearity : nn.Module, defaults to F.gelu
        nonlinear activation function between each FNO layer.
    fno_stabilizer : nn.Module | None, defaults to None
        By default None, otherwise tanh is used before FFT in the FNO block.
    fno_norm : nn.Module | None, defaults to None
        normalization layer to use in FNO.
    fno_ada_in_features : int | None, defaults to None
        if an adaptive mesh is used, number of channels of its positional embedding.
    fno_ada_in_dim : int, defaults to 1
        dimensions of above FNO adaptive mesh.
    fno_preactivation : bool, defaults to False
        whether to use Resnet-style preactivation.
    fno_skip : str, defaults to 'linear'
        type of skip connection to use.
    fno_channel_mlp_skip : str, defaults to 'soft-gating'
        type of skip connection to use in the FNO
        'linear': conv layer
        'soft-gating': weights the channels of the input
        'identity': nn.Identity
    fno_separable : bool, defaults to False
        if True, use a depthwise separable spectral convolution.
    fno_factorization : str {'tucker', 'tt', 'cp'} |  None, defaults to None
        Tensor factorization of the parameters weight to use
    fno_rank : float, defaults to 1.0
        Rank of the tensor factorization of the Fourier weights.
    fno_joint_factorization : bool, defaults to False
        Whether all the Fourier layers should be parameterized by a single tensor (vs one per layer).
    fno_fixed_rank_modes : bool, defaults to False
        Modes to not factorize.
    fno_implementation : str {'factorized', 'reconstructed'} | None, defaults to 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    fno_decomposition_kwargs : dict, defaults to dict()
        Optionaly additional parameters to pass to the tensor decomposition.
    fno_conv_module : nn.Module, defaults to SpectralConv
         Spectral Convolution module to use.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        projection_channels=256,
        gno_coord_dim=3,
        gno_pos_embed_type='transformer',
        gno_transform_type="linear",
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=4,
        fno_n_layers=4,
        # Other GNO params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        gno_radius=0.033,
        gno_weighting_function=None,
        gno_weight_function_scale=1.,
        gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=True,
        gno_batched=False,
        # Other FNO params
        fno_resolution_scaling_factor=None,
        fno_incremental_n_modes=None,
        fno_block_precision="full",
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None,
        fno_norm=None,
        fno_ada_in_features=None,
        fno_ada_in_dim=1,
        fno_preactivation=False,
        fno_skip="linear",
        fno_channel_mlp_skip="soft-gating",
        fno_separable=False,
        fno_factorization=None,
        fno_rank=1.0,
        fno_joint_factorization=False,
        fno_fixed_rank_modes=False,
        fno_implementation="factorized",
        fno_decomposition_kwargs=dict(),
        fno_conv_module=SpectralConv,
        **kwargs,
    ):
        super().__init__()

        self.gno_coord_dim = gno_coord_dim
        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(
                f"Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data"
            )

        self.in_coord_dim = len(fno_n_modes)
        if self.in_coord_dim != self.gno_coord_dim:
            print(
                f"Warning: FNO expects {self.in_coord_dim}-d data while GNO expects {self.gno_coord_dim}-d data"
            )

        # these lists contain the interior dimensions of the input
        # in order to reshape without explicitly providing dims
        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [
            j + 1 for j in self.in_coord_dim_forward_order
        ]

        self.gno_batched = gno_batched  # used in forward call to GNO

        # if batched, we must account for the extra batch dim
        # which causes previous dims to be incremented by 1
        if self.gno_batched:
            self.in_coord_dim_forward_order = [
                j + 1 for j in self.in_coord_dim_forward_order
            ]
            self.in_coord_dim_reverse_order = [
                j + 1 for j in self.in_coord_dim_reverse_order
            ]

        if fno_norm == "ada_in":
            if fno_ada_in_features is not None:
                self.adain_pos_embed = SinusoidalEmbedding(in_channels=fno_ada_in_dim,
                                                           num_frequencies=fno_ada_in_features,
                                                           embedding_type='transformer',
                                                           )
                # if ada_in positional embedding is provided, set the input dimension
                # of the ada_in norm to the output channels of positional embedding
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None

        # Create lifting for FNOBlock separately
        fno_lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels
        self.lifting = ChannelMLP(in_channels=in_channels + self.in_coord_dim,
                                  hidden_channels=fno_lifting_channels,
                                  out_channels=fno_hidden_channels,
                                  n_layers=3)
        
        self.fno_hidden_channels = fno_hidden_channels
        self.fno_blocks = FNOBlocks(
                n_modes=fno_n_modes,
                hidden_channels=fno_hidden_channels,
                in_channels=fno_hidden_channels,
                out_channels=fno_hidden_channels,
                positional_embedding=None,
                n_layers=fno_n_layers,
                resolution_scaling_factor=fno_resolution_scaling_factor,
                incremental_n_modes=fno_incremental_n_modes,
                fno_block_precision=fno_block_precision,
                channel_mlp_expansion=fno_channel_mlp_expansion,
                channel_mlp_dropout=fno_channel_mlp_dropout,
                non_linearity=fno_non_linearity,
                stabilizer=fno_stabilizer, 
                norm=fno_norm,
                ada_in_features=self.ada_in_dim,
                preactivation=fno_preactivation,
                fno_skip=fno_skip,
                channel_mlp_skip=fno_channel_mlp_skip,
                separable=fno_separable,
                factorization=fno_factorization,
                rank=fno_rank,
                joint_factorization=fno_joint_factorization, 
                fixed_rank_modes=fno_fixed_rank_modes,
                implementation=fno_implementation,
                decomposition_kwargs=fno_decomposition_kwargs,
                domain_padding=None,
                domain_padding_mode=None,
                conv_module=fno_conv_module,
                **kwargs
        )

        self.gno_radius = gno_radius

        if gno_weighting_function is not None: #sq radius**2?
            weight_fn = dispatch_weighting_fn(gno_weighting_function, sq_radius=gno_radius**2, scale=gno_weight_function_scale)
        else:
            weight_fn = None

        self.gno = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            radius=gno_radius,
            weighting_fn=weight_fn,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        self.projection = ChannelMLP(
            in_channels=fno_hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=1,
            non_linearity=fno_non_linearity,
        )

    # out_p : (n_out, gno_coord_dim)
    # in_p : (n_1, n_2, ..., n_k, k)
    # if batched shape is the same because this is just geometry
    # that remains constant across the entire batch
    # f : (n_1, n_2, ..., n_k,  in_channels)
    # if batched, (b, n_1, n_2, ..., n_k,  in_channels)
    # ada_in : (fno_ada_in_dim, )

    # returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, f, ada_in=None):
        if self.gno_batched:
            batch_size = f.shape[0]
            # repeat in_p along the batch dimension for latent embedding
            in_p = in_p.repeat([batch_size] + [1] * (in_p.ndim))
        in_p = torch.cat((f, in_p), dim=-1)

        if self.gno_batched:
            # shape: (b, k, n_1, n_2, ... n_k)
            in_p = in_p.permute(0, -1, *self.in_coord_dim_forward_order)
        else:
            in_p = in_p.permute(-1, *self.in_coord_dim_forward_order).unsqueeze(0)

        # Update Ada IN embedding
        if ada_in is not None:
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in

            self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        # Apply FNO blocks

        in_p = self.lifting(in_p)
        for layer_idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, layer_idx)

        if self.gno_batched:
            return in_p
        else:
            return in_p.squeeze(0)

    def integrate_latent(self, in_p, out_p, latent_embed):
        """
        Compute integration region for each output point
        """

        # (n_1*n_2*..., fno_hidden_channels)
        # if batched, (b, n1*n2*..., fno_hidden_channels)

        if self.gno_batched:
            batch_size = latent_embed.shape[0]
            latent_embed = latent_embed.permute(
                0, *self.in_coord_dim_reverse_order, 1
            ).reshape((batch_size, -1, self.fno_hidden_channels))
        else:
            latent_embed = latent_embed.permute(
                *self.in_coord_dim_reverse_order, 0
            ).reshape((-1, self.fno_hidden_channels))

        # (n_out, fno_hidden_channels)
        
        out = self.gno(
            y=in_p.reshape(-1, in_p.shape[-1]),
            x=out_p,
            f_y=latent_embed,
        )
        
        # if self.gno is variable and not batched
        if out.ndim == 2:
            out = out.unsqueeze(0)
        out = out.permute(0, 2, 1)  # b, c, n_out

        # Project pointwise to out channels
        out = self.projection(out)

        if self.gno_batched:
            out = out.permute(0, 2, 1)
        else:
            out = out.squeeze(0).permute(1, 0)

        return out

    def forward(self, in_p, out_p, f, ada_in=None, **kwargs):
        # Compute latent space embedding
        latent_embed = self.latent_embedding(in_p=in_p, f=f, ada_in=ada_in)
        # Integrate latent space
        out = self.integrate_latent(in_p=in_p, out_p=out_p, latent_embed=latent_embed)

        return out
