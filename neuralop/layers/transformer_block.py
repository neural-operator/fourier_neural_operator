import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_kernel_integral import AttentionKernelIntegral
from .mlp import MLPLinear, SirenMLP
from .embeddings import GaussianFourierEmbedding
from .skip_connections import skip_connection


def get_normalization(norm, channels):
    if norm == 'none':
        norm_fn = nn.Identity()
    elif norm == "instance_norm":
        norm_fn = nn.InstanceNorm1d(channels)
    elif norm == "group_norm":
        norm_fn = nn.GroupNorm(num_groups=32 if channels > 128 else 1, num_channels=channels)
    elif norm == 'layer_norm':
        norm_fn = nn.LayerNorm(channels)
    else:
        raise ValueError(
            f"Got norm={norm} but expected none or one of "
            "[instance_norm, group_norm, layer_norm]"
        )
    return norm_fn


def normalize(u, norm_fn):
    # transform into channel first, from: B N C to: B C N
    if isinstance(norm_fn, nn.GroupNorm) or isinstance(norm_fn, nn.InstanceNorm1d):
        u = u.permute(0, 2, 1).contiguous()
        u = norm_fn(u)
        u = u.permute(0, 2, 1).contiguous()
    else:
        u = norm_fn(u)
    return u


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with n_layers of self-attention + feedforward network (FFN),
            uses pre-normalization layout.

    For the detail definition of attention-based kernel integral, see `attention_kernel_integral.py`.

    Parameters:
        in_channels : int, input channels
        out_channels : int, output channels
        hidden_channels : int, hidden channels in the attention layers and MLP layers
        num_heads : int, number of attention heads
        head_n_channels : int, dimension of each attention head
        n_layers : int, number of (attention + FFN) layers
        use_mlp : bool, whether to use FFN after each attention layer, by default True
        mlp_dropout : float, dropout rate of the FFN, by default 0
        mlp_expansion : float, expansion factor of the FFN's hidden layer width, by default 2.0
        non_linearity : nn.Module, non-linearity module to use, by default F.gelu
        norm : string, normalization module to use, by default 'layer_norm', other available options are
            ['instance_norm', 'group_norm', 'none']
        attention_skip: string, type of skip connection to use in the attention layer, by default 'idenity'
        mlp_skip: string, type of skip connection to use in the FFN layer, by default 'identity'
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            num_heads,
            head_n_channels,
            n_layers,
            use_mlp=True,
            mlp_dropout=0,
            mlp_expansion=2.0,
            non_linearity=F.gelu,
            norm='layer_norm',
            attention_skip="identity",
            mlp_skip="identity",
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_n_channels = head_n_channels
        self.n_layers = n_layers
        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        self.lifting = nn.Linear(self.in_channels, self.hidden_channels) \
            if self.in_channels != self.hidden_channels else nn.Identity()

        self.to_out = nn.Linear(self.hidden_channels, self.out_channels) \
            if self.hidden_channels != self.out_channels else nn.Identity()

        self.attention_norms = nn.ModuleList([get_normalization(self.norm, self.hidden_channels) for _ in range(self.n_layers)])
        self.attention_layers = nn.ModuleList([
                                    AttentionKernelIntegral(
                                        in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        n_heads=self.num_heads,
                                        head_n_channels=self.head_n_channels,
                                        project_query=True)
                                    for _ in range(self.n_layers)])
        self.attention_skips = nn.ModuleList(
                                        [
                                            skip_connection(
                                                self.hidden_channels,
                                                self.hidden_channels,
                                                skip_type=attention_skip,
                                                n_dim=1,   # in transformer every spatial dimension got flattened
                                            )
                                            for _ in range(n_layers)
                                        ]
                                        )
        self.attention_skip_type = attention_skip

        if self.use_mlp:
            self.mlp_norms = nn.ModuleList([get_normalization(self.norm, self.hidden_channels) for _ in range(self.n_layers)])
            self.mlp_layers = nn.ModuleList([
                                    MLPLinear([self.hidden_channels,
                                       int(self.hidden_channels * self.mlp_expansion),
                                       self.hidden_channels],
                                       dropout=self.mlp_dropout)
                                for _ in range(self.n_layers)])

            self.mlp_skips = nn.ModuleList(
                                        [
                                            skip_connection(
                                                self.hidden_channels,
                                                self.hidden_channels,
                                                skip_type=mlp_skip,
                                                n_dim=1,
                                            )
                                            for _ in range(n_layers)
                                        ])

            self.mlp_skip_type = mlp_skip

    def compute_skip(self, u, skip_type, skip_module):
        if skip_type == 'soft-gating':
            # channel first
            u = u.permute(0, 2, 1).contiguous()
            u = skip_module(u)
            # channel second to spatial dimension
            u = u.permute(0, 2, 1).contiguous()
        else:
            u = skip_module(u)
        return u

    def forward(self,
                u,
                pos,
                pos_emb_module=None,
                **kwargs):
        """
        Encode the input function u using the Transformer Encoder Block.

        Parameters:
            u: torch.Tensor, input tensor of shape [batch_size, num_grid_points, channels]
            pos: torch.Tensor, grid point coordinates of shape [batch_size, num_grid_points, channels]
            pos_emb_module: nn.Module, positional embedding module, by default None

        """
        u = self.lifting(u)
        for l in range(self.n_layers):
            u_attention_skip = self.compute_skip(u, self.attention_skip_type, self.attention_skips[l])
            u = self.attention_layers[l](u_src=normalize(u, self.attention_norms[l]),
                                        pos_src=pos,
                                        positional_embedding_module=pos_emb_module,
                                        **kwargs)
            u = u + u_attention_skip
            if self.use_mlp:
                u_mlp_skip = self.compute_skip(u, self.mlp_skip_type, self.mlp_skips[l])
                u = self.mlp_layers[l](normalize(u, self.mlp_norms[l]))
                u = u + u_mlp_skip
        u = self.to_out(u)
        return u


# Note: this is not a causal-attention-based Transformer decoder as in language models
# but rather a "decoder" that maps from the latent grid to the output grid.
class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block using cross-attention to map input grid to output grid.

    For details regarding attention-based decoding, see:
    Transformer for Partial Differential Equations' Operator Learning: https://arxiv.org/abs/2205.13671
    Perceiver IO: A General Architecture for Structured Inputs & Outputs: https://arxiv.org/abs/2107.14795


    Parameters:
        n_dim: int, number of dimensions of the target domain
        in_channels : int, input channels
        out_channels : int, output channels
        hidden_channels : int, hidden channels in the attention layers and MLP layers
        num_heads : int, number of attention heads
        head_n_channels : int, dimension of each attention head
        query_basis: string, type of coordinate-based network to compute query basis function in the decoder,
            by default 'siren', other options are ['fourier', 'linear']
        use_mlp : bool, whether to use FFN after the cross-attention layer, by default True
        mlp_dropout : float, dropout rate of the FFN, by default 0
        mlp_expansion : float, expansion factor of the FFN's hidden layer width, by default 2.0
        non_linearity : nn.Module, non-linearity module to use, by default F.gelu
        norm : string, normalization module to use, by default 'layer_norm', other available options are
            ['instance_norm', 'group_norm', 'none']
        query_siren_layers: int, number of layers in SirenMLP, by default 3
        query_fourier_scale: float, scale (variance) of the Gaussian Fourier Feature Transform, by default 2.0
    """

    def __init__(
            self,
            n_dim,
            in_channels,
            out_channels,
            hidden_channels,
            num_heads,
            head_n_channels,
            query_basis='siren',
            use_mlp=True,
            mlp_dropout=0,
            mlp_expansion=2.0,
            non_linearity=F.gelu,
            norm='layer_norm',
            query_siren_layers=3,
            query_fourier_scale=2.0,
            **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_n_channels = head_n_channels
        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        self.query_basis = query_basis
        self.query_siren_layers = query_siren_layers
        self.query_fourier_scale = query_fourier_scale

        self.lifting = nn.Linear(self.in_channels, self.hidden_channels) \
            if self.in_channels != self.hidden_channels else nn.Identity()

        self.out_norm = get_normalization(self.norm, self.hidden_channels)
        self.to_out = MLPLinear([self.hidden_channels, self.hidden_channels, self.out_channels],
                                non_linearity=self.non_linearity)

        # build basis for decoder
        if self.query_basis == 'siren':
            self.query_basis_fn = SirenMLP(dim_in=self.n_dim,
                                           dim_hidden=self.hidden_channels,
                                           dim_out=self.num_heads * self.head_n_channels,
                                           num_layers=self.query_siren_layers)
        elif self.query_basis == 'fourier':
            self.query_basis_fn = nn.Sequential(
                GaussianFourierEmbedding(self.n_dim,
                                        mapping_size=self.head_n_channels,
                                        scale=self.query_fourier_scale),
                nn.Linear(self.head_n_channels * 2, self.num_heads * self.head_n_channels))
        elif self.query_basis == 'linear':
            self.query_basis_fn = nn.Linear(self.n_dim, num_heads * self.head_n_channels)
        else:
            raise ValueError(f'query_basis must be one of ["siren", "fourier", "linear"], got {self.query_basis}')

        self.attention_layer = AttentionKernelIntegral(in_channels=self.hidden_channels,
                                                        out_channels=self.hidden_channels,
                                                        n_heads=self.num_heads,
                                                        head_n_channels=self.head_n_channels,
                                                        project_query=False)

    def forward(self,
                u,
                pos_src,
                pos_emb_module=None,
                pos_qry=None,
                **kwargs
                ):
        """
           Project the input function u from the source grid to the query grid using the Transformer Decoder Block.

          Parameters:
                u: torch.Tensor, input tensor of shape [batch_size, num_src_grid_points, channels]
                pos_src: torch.Tensor, grid point coordinates of shape [batch_size, num_src_grid_points, channels]
                pos_emb_module: nn.Module, positional embedding module, by default None
                pos_qry: torch.Tensor, grid point coordinates of shape [batch_size, num_sry_grid_points, channels],
                         by default None and is set to pos_src, where input and output function will be sampled on
                         the same grid (the input grid specified by pos_src).
                         If pos_qry is provided, the output function will be sampled on query grid whose coordinates
                         are specified by pos_qry. This allows the output function to be sampled on arbitrary
                         discretization.

        """
        u = self.lifting(u)
        if pos_qry is None:
            pos_qry = pos_src  # assume that the query points are the same as the source points
        query_emb = self.query_basis_fn(pos_qry)
        query_emb = query_emb.view(pos_qry.shape[0], -1, self.num_heads * self.head_n_channels)
        if query_emb.shape[0] != u.shape[0]:
            query_emb = query_emb.expand(u.shape[0], -1, -1)

        u_out = self.attention_layer(u_src=u,
                                     pos_src=pos_src,
                                     u_qry=query_emb,
                                     pos_qry=pos_qry,
                                     positional_embedding_module=pos_emb_module,
                                     **kwargs)
        u_out = self.to_out(normalize(u_out, self.out_norm))
        return u_out








