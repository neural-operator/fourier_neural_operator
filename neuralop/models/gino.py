from functools import partial
import torch
import torch.nn.functional as F
import time

from .base_model import BaseModel

from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.gno_block import GNOBlock
from ..layers.gno_weighting_functions import dispatch_weighting_fn

class GINO(BaseModel):
    """GINO: Geometry-informed Neural Operator. Learns a mapping between
       functions presented over arbitrary coordinate meshes. The model carries
       global integration through spectral convolution layers in an intermediate
       latent space, as described in [1]_.

        Parameters
        ----------
        in_channels : int
            feature dimension of input points
        out_channels : int
            feature dimension of output points
        latent_feature_channels : int, optional
            number of channels in optional latent feature map
            to concatenate onto latent embeddings before 
            the FNO's forward pass, default None
        projection_channels : int, optional
            number of channels in FNO pointwise projection
        gno_coord_dim : int, optional
            geometric dimension of input/output queries, by default 3
        in_gno_radius : float, optional
            radius in input space for GNO neighbor search, by default 0.033
        out_gno_radius : float, optional
            radius in output space for GNO neighbor search, by default 0.033
        gno_weight_function : str, optional
            Choice of weighting function to use in the output GNO for 
            Mollified Graph Neural Operator-based models
        gno_weight_function_scale : float, optional
            Factor by which to scale weights from GNO weighting function
            by default 1
        in_gno_transform_type : str, optional
            transform type parameter for input GNO, by default 'linear'
            see neuralop.layers.gno_block for more details
        out_gno_transform_type : str, optional
            transform type parameter for output GNO, by default 'linear'
            see neuralop.layers.gno_block for more details
        in_gno_pos_embed_type : literal `{'transformer', 'nerf'}` | None
            type of optional sinusoidal positional embedding to use in input GNOBlock,
            by default `'transformer'`
        out_gno_pos_embed_type : literal `{'transformer', 'nerf'}` | None
            type of optional sinusoidal positional embedding to use in output GNOBlock,
            by default `'transformer'`
        fno_in_channels : int, optional
            number of input channels for FNO, by default 26
        fno_n_modes : tuple, optional
            number of modes along each dimension 
            to use in FNO, by default (16, 16, 16)
        fno_hidden_channels : int, optional
            hidden channels for use in FNO, by default 64
        lifting_channel_ratio : int, optional
            ratio of lifting channels to `fno_hidden_channels`, by default 2
            The number of liting channels in the lifting block of the FNO is
            lifting_channel_ratio * hidden_channels (e.g. default 512)
        fno_n_layers : int, optional
            number of layers in FNO, by default 4
        
        Other Parameters
        ----------------
        gno_embed_channels: int
            dimension of optional per-channel embedding to use in GNOBlock,
            by default 32
        gno_embed_max_positions: int
            max positions of optional per-channel embedding to use in GNOBlock,
            by default 10000. If `gno_pos_embed_type != 'transformer'`, value is unused.
        in_gno_channel_mlp_hidden_layers : list, optional
            widths of hidden layers in input GNO, by default [80, 80, 80]
        out_gno_channel_mlp_hidden_layers : list, optional
            widths of hidden layers in output GNO, by default [512, 256]
        gno_channel_mlp_non_linearity : nn.Module, optional
            nonlinearity to use in gno ChannelMLP, by default F.gelu
        gno_use_open3d : bool, optional
            whether to use open3d neighbor search, by default True
            if False, uses pure-PyTorch fallback neighbor search
        gno_use_torch_scatter : bool, optional
            whether to use torch_scatter's neighborhood reduction function
            or the native PyTorch implementation in IntegralTransform layers.
            If False, uses the fallback PyTorch version.
        out_gno_tanh : bool, optional
            whether to use tanh to stabilize outputs of the output GNO, by default False
        fno_resolution_scaling_factor : float | None, optional
            factor by which to scale output of FNO, by default None
        fno_incremental_n_modes : list[int] | None, defaults to None
        if passed, sets n_modes separately for each FNO layer.
        fno_block_precision : str, defaults to 'full'
            data precision to compute within fno block
        fno_use_channel_mlp : bool, defaults to True
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
        fno_ada_in_features : int | None, defaults to 4
            if an adaptive mesh is used, number of channels of its positional embedding.
            If None, adaptive mesh embedding is not used.
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
        
            
        References
        -----------
        .. _[1] : 
        
        Li, Z., Kovachki, N., Choy, C., Li, B., Kossaifi, J., Otta, S., 
            Nabian, M., Stadler, M., Hundt, C., Azizzadenesheli, K., Anandkumar, A. (2023)
            Geometry-Informed Neural Operator for Large-Scale 3D PDEs. NeurIPS 2023,
            https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html
        """
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_feature_channels=None,
        projection_channels=256,
        gno_coord_dim=3,
        in_gno_radius=0.033,
        out_gno_radius=0.033,
        in_gno_transform_type='linear',
        out_gno_transform_type='linear',
        gno_weighting_function=None,
        gno_weight_function_scale=1,
        in_gno_pos_embed_type='transformer',
        out_gno_pos_embed_type='transformer',
        fno_in_channels=3,
        fno_n_modes=(16, 16, 16), 
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        fno_n_layers=4,
        # Other GNO Params
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu, 
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        # Other FNO Params
        fno_resolution_scaling_factor=None,
        fno_incremental_n_modes=None,
        fno_block_precision='full',
        fno_use_channel_mlp=True, 
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None, 
        fno_norm=None,
        fno_ada_in_features=4,
        fno_ada_in_dim=1,
        fno_preactivation=False,
        fno_skip='linear',
        fno_channel_mlp_skip='soft-gating',
        fno_separable=False,
        fno_factorization=None,
        fno_rank=1.0,
        fno_joint_factorization=False, 
        fno_fixed_rank_modes=False,
        fno_implementation='factorized',
        fno_decomposition_kwargs=dict(),
        fno_conv_module=SpectralConv,
        **kwargs
        ):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels

        self.lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels

        # TODO: make sure this makes sense in all contexts
        if in_gno_transform_type in ["nonlinear", "nonlinear_kernelonly"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = fno_in_channels

        self.fno_in_channels = in_gno_out_channels

        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels

        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3d expects 3-d data')
            gno_use_open3d = False

        self.in_coord_dim = len(fno_n_modes)
        self.gno_out_coord_dim = len(fno_n_modes) # gno output and fno will use same dimensions
        if self.in_coord_dim != self.gno_coord_dim:
            print(f'Warning: FNO expects {self.in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data')

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        # channels starting at 2 to permute everything after channel and batch dims
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.fno_norm = fno_norm
        if self.fno_norm == "ada_in":
            if fno_ada_in_features is not None and out_gno_pos_embed_type is not None:
                self.adain_pos_embed = SinusoidalEmbedding(in_channels=fno_ada_in_dim,
                                                        num_frequencies=fno_ada_in_features, 
                                                        max_positions=10000,
                                                        embedding_type=out_gno_pos_embed_type)                    
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.ada_in_dim = fno_ada_in_dim
                self.adain_pos_embed = None
        else:
            self.adain_pos_embed = None
            self.ada_in_dim = None
        
        self.in_gno_radius = in_gno_radius
        self.out_gno_radius = out_gno_radius

        self.out_gno_tanh = out_gno_tanh

        ### input GNO
        # input to the first GNO ChannelMLP: `x` pos encoding,
        # `y` (integrand) pos encoding, potentially `f_y`

        self.gno_in = GNOBlock(
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=in_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=in_gno_radius,
            reduction='mean',
            weighting_fn=None,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        ### Lifting layer before FNOBlocks
        self.lifting = ChannelMLP(in_channels=self.fno_in_channels,
                                  hidden_channels=self.lifting_channels,
                                  out_channels=fno_hidden_channels,
                                  n_layers=2) # CHANGED RECENTLY FOR THIS PAPER
        
        ### FNOBlocks in latent space
        # input: `in_p` intermediate embeddings,
        # possibly concatenated feature channels `latent_features` 
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
                use_channel_mlp=fno_use_channel_mlp,
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

        ### output GNO
        if gno_weighting_function is not None: #sq radius**2?
            weight_fn = dispatch_weighting_fn(gno_weighting_function, sq_radius=out_gno_radius**2, scale=gno_weight_function_scale)
        else:
            weight_fn = None
        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels, # number of channels in f_y
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.out_gno_radius,
            reduction='sum',
            weighting_fn=weight_fn,
            pos_embedding_type=out_gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
        )

        self.projection = ChannelMLP(in_channels=fno_hidden_channels, 
                              out_channels=self.out_channels, 
                              hidden_channels=projection_channels, 
                              n_layers=2, 
                              n_dim=1, 
                              non_linearity=fno_non_linearity) 

    #returns: (fno_hidden_channels, n_1, n_2, ...)
    def latent_embedding(self, in_p, ada_in=None):

        # in_p : (batch, n_1 , ... , n_k, in_channels + k)
        # ada_in : (fno_ada_in_dim, )

        # permute (b, n_1, ..., n_k, c) -> (b,c, n_1,...n_k)
        in_p = in_p.permute(0, len(in_p.shape)-1, *list(range(1,len(in_p.shape)-1)))
        #Update Ada IN embedding    
        if ada_in is not None:
            if ada_in.ndim == 2:
                ada_in = ada_in.squeeze(0)
            if self.adain_pos_embed is not None:
                ada_in_embed = self.adain_pos_embed(ada_in.unsqueeze(0)).squeeze(0)
            else:
                ada_in_embed = ada_in
            if self.fno_norm == "ada_in":
                self.fno_blocks.set_ada_in_embeddings(ada_in_embed)

        #Apply FNO blocks
        in_p = self.lifting(in_p)

        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(in_p, idx)

        return in_p 
    
    def forward(self, input_geom, latent_queries, output_queries, x=None, latent_features=None, ada_in=None, **kwargs):
        """The GINO's forward call:
        Input GNO --> FNOBlocks --> output GNO + projection to output queries.

        .. note ::
            GINO currently supports batching **only in cases where the geometry of
            inputs and outputs is shared across the entire batch**. Inputs can have a batch dim
            in ``x`` and ``latent_features``, but it must be shared for both. 

        Parameters
        ----------
        input_geom : torch.Tensor
            input domain coordinate mesh
            shape (1, n_in, gno_coord_dim)
        latent_queries : torch.Tensor
            latent geometry on which to compute FNO latent embeddings
            a grid on [0,1] x [0,1] x ....
            shape (1, n_gridpts_1, .... n_gridpts_n, gno_coord_dim)
        output_queries : torch.Tensor | dict[torch.Tensor]
            points at which to query the final GNO layer to get output.

            shape (1, n_out, gno_coord_dim) per tensor.

            * if a tensor, the model will output a tensor. 

            * if a dict of tensors, the model will return a dict of outputs, so
            that ``output[key]`` corresponds to the model queried at 
            ``output_queries[key]``. 
        x : torch.Tensor, optional
            input function a defined on the input domain `input_geom`
            shape (batch, n_in, in_channels). Default None
        latent_features : torch.Tensor, optional
            optional feature map to concatenate onto latent embedding
            before being passed into the latent FNO, default None
            if `latent_feature_channels` is set, must be passed
        ada_in : torch.Tensor, optional
            adaptive scalar instance parameter, defaults to None

        Returns
        -------
        out : torch.Tensor | dict[torch.Tensor]
            Function over the output query coordinates
            * tensor if if ``output_queries`` is a tensor
            * dict if if ``output_queries`` is a dict
        """

        # Ensure input functions on the input geom and latent geom
        # have compatible batch sizes
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        
        if latent_features is not None:
            assert self.latent_feature_channels is not None,\
                  "if passing latent features, latent_feature_channels must be set."
            assert latent_features.shape[-1] == self.latent_feature_channels

            # batch, n_gridpts_1, .... n_gridpts_n, gno_coord_dim
            assert latent_features.ndim == self.gno_coord_dim + 2,\
                f"Latent features must be of shape (batch, n_gridpts_1, ...n_gridpts_n, gno_coord_dim), got {latent_features.shape}"
            # latent features must have the same shape (except channels) as latent_queries 
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(batch_size, *[1]*(latent_features.ndim-1))

        input_geom = input_geom.squeeze(0) 
        latent_queries = latent_queries.squeeze(0)

        # Pass through input GNOBlock 
        in_p = self.gno_in(y=input_geom,
                           x=latent_queries.view((-1, latent_queries.shape[-1])),
                           f_y=x)
        
        grid_shape = latent_queries.shape[:-1] # disregard positional encoding dim
        
        # shape (batch_size, grid1, ...gridn, -1)
        in_p = in_p.view((batch_size, *grid_shape, -1))
        
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)
        # take apply fno in latent space
        latent_embed = self.latent_embedding(in_p=in_p, 
                                             ada_in=ada_in)

        # Integrate latent space to output queries
        #latent_embed shape (b, c, n_1, n_2, ..., n_k)
        batch_size = latent_embed.shape[0]
        # permute to (b, n_1, n_2, ...n_k, c)
        # then reshape to (b, n_1 * n_2 * ...n_k, out_channels)
        latent_embed = latent_embed.permute(0, *self.in_coord_dim_reverse_order, 1).reshape(batch_size, -1, self.fno_hidden_channels)
        
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed = torch.tanh(latent_embed)
        

        # integrate over the latent space
        # if output queries is a dict, query the output gno separately 
        # with each tensor of query points
        if isinstance(output_queries, dict):
            out = {}
            for key, out_p in output_queries.items():
                out_p = out_p.squeeze(0)

                sub_output = self.gno_out(y=latent_queries.reshape((-1, latent_queries.shape[-1])), 
                    x=out_p,
                    f_y=latent_embed,)
                sub_output = sub_output.permute(0, 2, 1)

                # Project pointwise to out channels
                #(b, n_in, out_channels)
                sub_output = self.projection(sub_output).permute(0, 2, 1)  

                out[key] = sub_output
        else:
            output_queries = output_queries.squeeze(0)

            # latent queries is of shape (d_1 x d_2 x... d_n x n), reshape to n_out x n
            out = self.gno_out(y=latent_queries.reshape((-1, latent_queries.shape[-1])), 
                        x=output_queries,
                        f_y=latent_embed,)
            out = out.permute(0, 2, 1)

            # Project pointwise to out channels
            #(b, n_in, out_channels)
            out = self.projection(out).permute(0, 2, 1)  
        
        return out