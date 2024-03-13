# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
from functools import partialmethod

import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from ..layers.resample import resample
from ..layers.afno_block import AFNOBlock, PatchEmbed, TimeAggregator
from ..layers.cno_block import CNOBlock
from .base_model import BaseModel





ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class DPOT(BaseModel, name='DPOT'):
    def __init__(self, n_dim = 2,  in_size=64, patch_size=16,in_channels=1, out_channels = 1, temporal=False, in_timesteps=1, out_timesteps=1, n_heads=4, embed_dim=768,out_layer_dim=32, n_layers=4,modes=32,num_norm_groups=8,
                 mlp_ratio=1., load_grid=True, double_skip=True, non_linearity='gelu',time_agg='exp_mlp', use_cno_block=False):
        '''
        Args:
        n_dim: int
            dimension of model, int {1, 2, 3}
        in_size: int
            tuple or int, resolution
        patch_size: int
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        temporal: bool
            if True, the data has temporal dimension, use in_timesteps and out_timesteps t
        in_timesteps: int
            number of input timesteps
        out_timesteps: int
            number of output timesteps
        n_heads: int
            number of heads int Fourier Mixer
        embed_dim: int
            dimension of latent embedding
        out_layer_dim: int
            latent dimension of decoding layer
        n_layers: int
            number of layers
        modes: int
            number of Fourier modes
        num_norm_groups: int
            number of groups for the GroupNorm layers, by default 8
        mlp_ratio: int
            the ratio for MLP dimension (mlp_ratio * embed_dim)
        load_grid: bool
            whether load uniform grid in the model, by default True
        double_skip: bool
            whether use double residual connection in AFNOBlock
        non_linearity: str
            type of activation function, by default gelu
        time_agg: str
            type of temporal aggregator
        use_cno_block: bool
            whether use CNO layers for anti-aliasing
        '''
        super(DPOT, self).__init__()

        # self.num_classes = num_classes
        self.n_dim = n_dim
        self.in_size = [in_size for _ in range(n_dim)] if isinstance(in_size, int) else list(in_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.temporal = temporal
        self.load_grid = load_grid


        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim
        self.time_agg = time_agg
        self.mlp_ratio = mlp_ratio
        self.num_norm_groups = num_norm_groups
        self.double_skip = double_skip
        self.non_linearity = ACTIVATION[non_linearity]
        self.use_cno_block = use_cno_block


        self.sp_dim = n_dim + 1

        if self.load_grid:
            self.patch_embed = PatchEmbed(n_dim = self.n_dim, res=self.in_size, patch_size=patch_size, in_chans=in_channels + self.sp_dim, embed_dim=out_channels * patch_size + self.sp_dim, out_dim=embed_dim,non_linearity=non_linearity,use_cno_block=use_cno_block)
        else:
            self.patch_embed = PatchEmbed(n_dim = self.n_dim, res=self.in_size, patch_size=patch_size, in_chans=in_channels, embed_dim=out_channels * patch_size, out_dim=embed_dim,non_linearity=non_linearity,use_cno_block=use_cno_block)


        self.latent_size = self.patch_embed.out_size
        self.get_grid = getattr(self, f"get_grid_{self.sp_dim}d") if self.temporal else getattr(self, f"get_grid_{self.sp_dim}d")
        self.lifting = getattr(self, f"lifting_{n_dim}d")

        if self.n_dim == 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0]))
        elif self.n_dim == 2:
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        elif self.n_dim == 3:
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1], self.patch_embed.out_size[2]))
        else:
            raise NotImplementedError





        self.blocks = nn.ModuleList([
            AFNOBlock(n_dim = self.n_dim, modes = modes, width = embed_dim, mlp_ratio = mlp_ratio, n_blocks=n_heads,double_skip=double_skip,non_linearity = non_linearity,num_norm_groups=num_norm_groups)
            for i in range(n_layers)])


        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)


        Conv = getattr(nn, f"Conv{n_dim}d")
        if use_cno_block:
            ConvOut = CNOBlock(n_dim, embed_dim, out_layer_dim, self.latent_size, self.in_size, sampling_rate=patch_size,conv_kernel=patch_size)
        else:
            ConvOut = getattr(nn, f"ConvTranspose{n_dim}d")(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size)
        self.out_layer = nn.Sequential(
            ConvOut,
            self.non_linearity,
            Conv(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.non_linearity,
            Conv(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )




        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)    # .02
            if m.bias is not None:
            # if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_grid_1d(self, x):
        '''
        :param x: [B, X, C]
        :return: [B, X, C]
        '''
        batchsize, size_x = x.shape[0], x.shape[1]
        grid = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid = grid.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return grid

    def get_grid_2d(self, x):
        '''
        :param x: [B, X, Y, C]
        :return: [B, X, Y, C]
        '''
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_grid_3d(self, x):
        '''
        :param x: [B, X, Y, Z, C]
        :return: [B, X, Y, Z, C]
        '''
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid

    def get_grid_4d(self, x):
        '''
        :param x: [B, X, Y, Z, T, C]
        :return: [B, X, Y, Z, T, C]
        '''
        batchsize, size_x, size_y, size_z, size_t = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, size_t, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, 1, size_t, 1).to(x.device).repeat([batchsize, size_x, size_y, size_z, 1, 1])
        grid = torch.cat((gridx, gridy, gridz, gridt), dim=-1)
        return grid

    def lifting_1d(self, x):
        '''
        :param x: input tensor, [B, C, X] or [B, T, C, X]
        :return: latent embeddings, [B, X, T, C]
        '''
        if not self.temporal:
            x = x.unsqueeze(1)
        x = rearrange(x, 'b t c x -> b x t c')
        B, _, T, _ = x.shape
        if self.load_grid:
            grid = self.get_grid(x)
            x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, T, C+1
        x = rearrange(x, 'b x t c -> (b t) c x')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = rearrange(x, '(b t) c x -> b x t c', b=B, t=T)
        return x

    def lifting_2d(self, x):
        '''
        :param x: input tensor, [B, C, X, Y] or [B, T, C, X, Y]
        :return: latent embeddings, [B, X, Y, T, C]
        '''
        if not self.temporal:
            x = x.unsqueeze(1)
        x = rearrange(x, 'b t c x y -> b x y t c')
        B, _, _, T, _ = x.shape
        if self.load_grid:
            grid = self.get_grid(x)
            x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)
        return x

    def lifting_3d(self, x):
        '''
        :param x: input tensor, [B, C, X, Y, Z] or [B, T, C, X, Y, Z]
        :return: latent embeddings, [B, X, Y, Z, T, C]
        '''
        if not self.temporal:
            x = x.unsqueeze(1)
        x = rearrange(x, 'b t c x y z -> b x y z t c')
        B, _, _, _, T, _ = x.shape
        if self.get_grid:
            grid = self.get_grid(x)
            x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, T, C+3
        x = rearrange(x, 'b x y z t c -> (b t) c x y z')
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = rearrange(x, '(b t) c x y z -> b x y z t c', b=B, t=T)
        return x

    def reshape_channels(self, x, channel_first=True):
        if channel_first:
            if self.n_dim == 1:
                x = rearrange(x, 'b c x -> b x c')
            elif self.n_dim == 2:
                x = rearrange(x, 'b c x y -> b x y c')
            elif self.n_dim == 3:
                x = rearrange(x, 'b c x y z -> b x y z c')
            else:
                raise NotImplementedError
        else:
            if self.n_dim == 1:
                x = rearrange(x, 'b x c -> b c x')
            elif self.n_dim == 2:
                x = rearrange(x, 'b x y c -> b c x y')
            elif self.n_dim == 3:
                x = rearrange(x, 'b x y z c -> b c x y z')
            else:
                raise NotImplementedError
        return x


    def recompute_output_shape(self, output_shape=None):
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        for i in range(self.n_layers):
            if output_shape[i] is not None:
                output_shape[i] = tuple([shape//self.patch_size for shape in output_shape[i]])
        return output_shape


    def resample_data(self, x, output_shape, temporal=True):
        if temporal:
            B, T, orig_size = x.shape[0], x.shape[1], x.shape[3:]

            x = rearrange(x, 'b t ... -> (b t) ...')
            x = resample(x, 1, axis=list(range(2,x.ndim)), output_shape=output_shape)
            x = rearrange(x, '(b t) ... -> b t ...', b=B, t=T)
        else:
            x = resample(x, 1, axis=list(range(2, x.ndim)), output_shape=output_shape)

        return x


    def forward(self, x, output_shape=None, **kwargs):
        '''
        :param x: input tensor of shape [B, T, C, X, Y ] or [B, C, X, Y]
        :return: tensor of shape [B, T, C, X, Y ] or [B, C, X, Y]
        '''
        orig_shape = list(x.shape[x.ndim - self.n_dim:])
        if orig_shape != self.in_size:
            x = self.resample_data(x, self.in_size, temporal=self.temporal)
        output_shapes = self.recompute_output_shape(output_shape)

        x = self.lifting(x)
        x = self.time_agg_layer(x)

        x = self.reshape_channels(x, channel_first=False)

        for i, blk in enumerate(self.blocks):
            x = blk(x,output_shape=output_shapes[i])

        x = self.out_layer(x)   # 2d: B, C, X, Y

        if (output_shapes[-1] is None) and (orig_shape != self.in_size):
            x = self.resample_data(x, output_shape=orig_shape, temporal=False)

        if self.temporal:
            x = x.reshape(x.shape[0], self.out_timesteps, self.out_channels, *x.shape[2:]).contiguous()


        return x





def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )
    return new_class

DPOT1d = partialclass('DPOT1d', DPOT, n_dim = 1)
DPOT2d = partialclass('DPOT2d', DPOT, n_dim = 2)
DPOT3d = partialclass('DPOT3d', DPOT, n_dim = 3)

if __name__ == "__main__":
    x = torch.rand(4, 6, 3, 20)
    net = DPOT1d(in_size=20, temporal=True, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32, num_norm_groups=4)
    y = net(x)
    print('1d temporal', y.shape)

    ## 1d steady
    x = torch.rand(4, 3, 20)
    net = DPOT1d(in_size=20, temporal=False, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=1,out_timesteps=1, embed_dim=32, num_norm_groups=4)
    y = net(x)
    print('1d steady', y.shape)

    ## 2d temporal
    x = torch.rand(4, 6, 3, 20, 20)
    net = DPOT2d(in_size=20, temporal=True, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32,num_norm_groups=4)
    y = net(x)
    print('2d temporal',y.shape)

    x = torch.rand(4, 3, 20, 20)
    net = DPOT2d(in_size=20, temporal=False, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=1, out_timesteps=1, embed_dim=32,num_norm_groups=4)
    y = net(x)
    print('2d steady', y.shape)

    ## 3d temporal
    x = torch.rand(4, 6, 3, 20, 20, 20)
    net = DPOT3d(in_size=20, temporal=True, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=6, out_timesteps=1, embed_dim=32,num_norm_groups=4)
    y = net(x)
    print('3d temporal',y.shape)

    x = torch.rand(4, 3, 20, 20, 20)
    net = DPOT3d(in_size=20, temporal=False, load_grid=True, patch_size=5, in_channels=3, out_channels=3, in_timesteps=1, out_timesteps=1, embed_dim=32,num_norm_groups=4)
    y = net(x)
    print('3d steady', y.shape)