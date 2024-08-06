from functools import partialmethod

import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..layers.transformer_block import TransformerEncoderBlock, TransformerDecoderBlock
from ..layers.embeddings import RotaryEmbedding2D


class TransformerNO(BaseModel, name='transformer_no'):
    """N-Dimensional Transformer-based Neural Operator (currently does not support N>2)
        using softmax-free attention to compute the kernel integral.

        Each layer in the encoder part is organized as follow (a.k.a pre-norm version of Transformer layer):
            u = attn(norm(u)) + u
            u = mlp(norm(u)) + u
        where u is the input function to the layer.

        For the decoder (cross-attention), given query bases q and src function u:
            u_out = attn(q, u)   # u_out will has the same shape as q

        Parameters
        ----------
        n_dim : int
            Number of dimensions of the domain
        in_channels : int, optional
            Number of input channels, by default 1
        out_channels : int, optional
            Number of output channels, by default 1
        encoder_hidden_channels : int
            Width of the encoder (i.e. number of channels in attention layer and MLP)
        decoder_hidden_channels : int
            Width of the decoder (i.e. number of channels in attention layer and MLP)
        encoder_num_heads: int, optional
            Number of heads in the encoder attention, by default 1
        decoder_num_heads: int, optional
            Number of heads in the decoder cross-attention, by default 8
        encoder_head_n_channels: int, optional
            Dimension of each attention head in the encoder, by default equals to encoder_hidden_channels
        decoder_head_n_channels: int, optional
            Dimension of each attention head in the decoder, by default equals to decoder_hidden_channels
        encoder_n_layers : int, optional
            Number of Transformer layer in the encoder, by default 3
        query_basis: string, optional
            Type of coordinate-based network to compute query basis function in the decoder,
            by default 'siren', other options are ['fourier', 'linear']
        query_siren_layers: int, optional
            Number of layers in SirenNet, by default 4
        query_fourier_scale: float, optional
            Scale of the Gaussian Fourier Feature Transform in random Fourier Feature, by default 2.0
        use_mlp : bool, optional
            Whether to use an MLP layer after each attention block, by default True
        mlp_dropout : float , optional
            droupout parameter of MLP layer, by default 0
        mlp_expansion : float, optional
            expansion parameter of MLP layer, by default 2.0
        non_linearity : nn.Module, optional
            Non-Linearity module to use, by default F.gelu
        norm: string, optional
            Normalization module to modeluse, by default layernorm

        """
    def __init__(self,
                 n_dim,
                 in_channels=1,
                 out_channels=1,
                 encoder_hidden_channels=128,
                 decoder_hidden_channels=128,
                 encoder_num_heads=1,
                 decoder_num_heads=8,
                 encoder_head_n_channels=None,
                 decoder_head_n_channels=None,
                 encoder_n_layers=3,
                 query_basis='siren',
                 query_siren_layers=4,          # number of layers in SirenNet
                 query_fourier_scale=2.0,       # scale of the Gaussian Fourier Feature Transform
                 pos_emb='rotary',              # ['rotary', 'none']
                 use_mlp=True,
                 mlp_dropout=0,
                 mlp_expansion=2.0,
                 non_linearity=F.gelu,
                 norm='layer_norm',      # ['layer_norm', 'instance_norm', ''group_norm', 'none']
                 attention_skip="identity",
                 mlp_skip="identity",
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.encoder_head_n_channels = encoder_head_n_channels if encoder_head_n_channels is not None else encoder_hidden_channels
        self.decoder_head_n_channels = decoder_head_n_channels if decoder_head_n_channels is not None else decoder_hidden_channels
        self.encoder_n_layers = encoder_n_layers
        self.query_basis = query_basis
        self.query_siren_layers = query_siren_layers
        self.query_fourier_scale = query_fourier_scale
        self.pos_emb = pos_emb

        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm
        self.attention_skip = attention_skip
        self.mlp_skip = mlp_skip

        if self.pos_emb not in ['rotary', 'none']:
            raise ValueError(f'pos_emb must be one of ["rotary", "none"], got {self.pos_emb}')

        if self.pos_emb == 'rotary':
            self.enc_pos_emb_module = RotaryEmbedding2D(self.encoder_head_n_channels // self.n_dim)
            self.dec_pos_emb_module = RotaryEmbedding2D(self.decoder_head_n_channels // self.n_dim)
        else:
            self.enc_pos_emb_module = None
            self.dec_pos_emb_module = None

        self.encoder = TransformerEncoderBlock(
                            in_channels=self.in_channels,
                            out_channels=self.decoder_hidden_channels,
                            hidden_channels=self.encoder_hidden_channels,
                            num_heads=self.encoder_num_heads,
                            head_n_channels=self.encoder_head_n_channels,
                            n_layers=self.encoder_n_layers,
                            use_mlp=self.use_mlp,
                            mlp_dropout=self.mlp_dropout,
                            mlp_expansion=self.mlp_expansion,
                            non_linearity=self.non_linearity,
                            norm=self.norm,
                            attention_skip=self.attention_skip,
                            mlp_skip=self.mlp_skip,

                        )

        self.decoder = TransformerDecoderBlock(
            n_dim=self.n_dim,
            in_channels=self.decoder_hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.decoder_hidden_channels,
            num_heads=self.decoder_num_heads,
            head_n_channels=self.decoder_head_n_channels,
            query_basis=self.query_basis,
            use_mlp=self.use_mlp,
            mlp_dropout=self.mlp_dropout,
            mlp_expansion=self.mlp_expansion,
            non_linearity=self.non_linearity,
            query_siren_layers=self.query_siren_layers,
            query_fourier_scale=self.query_fourier_scale,
            norm=self.norm,
        )

    def forward(self,
                u,
                pos_src,
                pos_qry=None,
                **kwargs):
        """Transformer NO's forward pass,
           please note that coordinates must be normalized to [-1, 1] interval when using siren"""

        # encoder part, use self-attention to process input function
        u = self.encoder(u, pos_src, self.enc_pos_emb_module, **kwargs)

        # decoder part, use cross-attention to query the solution function
        u_out = self.decoder(u, pos_src, self.dec_pos_emb_module, pos_qry, **kwargs)

        return u_out
