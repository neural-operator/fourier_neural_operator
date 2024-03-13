# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


import math
from einops import rearrange, repeat
from .cno_block import CNOBlock, LReLu_torch
from .resample import resample


ACTIVATION = {
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus(),
    "ELU": nn.ELU(),
    "silu": nn.SiLU(),
}


class AFNO1D(nn.Module):
    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=True,
        sparsity_threshold=0.01,
        modes=32,
        hidden_size_factor=1,
        non_linearity="gelu",
    ):
        super().__init__()
        assert (
            width % num_blocks == 0
        ), f"hidden_size {width} should be divisble by num_blocks {num_blocks}"
        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)
        self.non_linearity = ACTIVATION[non_linearity]

        self.w1 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        """
        :param x: [N, C, X]
        :return:  [N, C, X]
        """
        if self.channel_first:
            B, C, H = x.shape
            x = rearrange(x, "b c x -> b x c")
        else:
            B, H, C = x.shape

        x_orig = x

        x = torch.fft.rfft(x, dim=(1), norm="ortho")

        x = x.reshape(B, x.shape[1], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [B, x.shape[1], self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = self.modes

        o1_real[:, :kept_modes] = self.non_linearity(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[1])
            + self.b1[0]
        )

        o1_imag[:, :kept_modes] = self.non_linearity(
            torch.einsum("...bi,bio->...bo", x[:, :kept_modes].imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x[:, :kept_modes].real, self.w1[1])
            + self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[1])
            + self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum("...bi,bio->...bo", o1_imag[:, :kept_modes], self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real[:, :kept_modes], self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], C)
        x = torch.fft.irfft(x, n=H, dim=(1), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = rearrange(x, "b x c -> b c x")

        return x


class AFNO2D(nn.Module):
    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=True,
        sparsity_threshold=0.01,
        modes=32,
        hidden_size_factor=1,
        non_linearity="gelu",
    ):
        super().__init__()
        assert (
            width % num_blocks == 0
        ), f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)
        self.non_linearity = ACTIVATION[non_linearity]

        self.w1 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        """
        :param x: [N, C, X, Y]
        :return: [N, C, X, Y]
        """
        if self.channel_first:
            B, C, H, W = x.shape
            x = rearrange(x, "b c x y -> b x y c")
        else:
            B, H, W, C = x.shape

        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.non_linearity(
            torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[0]
            )
            - torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[1]
            )
            + self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.non_linearity(
            torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].imag, self.w1[0]
            )
            + torch.einsum(
                "...bi,bio->...bo", x[:, :kept_modes, :kept_modes].real, self.w1[1]
            )
            + self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[0]
            )
            - torch.einsum(
                "...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[1]
            )
            + self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum(
                "...bi,bio->...bo", o1_imag[:, :kept_modes, :kept_modes], self.w2[0]
            )
            + torch.einsum(
                "...bi,bio->...bo", o1_real[:, :kept_modes, :kept_modes], self.w2[1]
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        # x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = rearrange(x, "b x y c -> b c x y")

        return x


class AFNO3D(nn.Module):
    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=True,
        sparsity_threshold=0.01,
        modes=32,
        temporal_modes=8,
        hidden_size_factor=1,
        non_linearity="gelu",
    ):
        super(AFNO3D, self).__init__()
        assert (
            width % num_blocks == 0
        ), f"hidden_size {width} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = ACTIVATION[non_linearity]
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.w1 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size,
                self.block_size * self.hidden_size_factor,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.rand(
                2,
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
                self.block_size,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

    def forward(self, x):
        """
        :param x: [N, C, X, Y, Z]
        :return: [N, C, X, Y, Z]
        """
        if self.channel_first:
            x = rearrange(x, "b c x y z -> b x y z c")
        B, H, W, L, C = x.shape
        x_orig = x

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")

        x = x.reshape(
            B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size
        )

        o1_real = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [
                B,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                self.num_blocks,
                self.block_size * self.hidden_size_factor,
            ],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes] = F.gelu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].real,
                self.w1[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].imag,
                self.w1[1],
            )
            + self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes] = F.gelu(
            torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].imag,
                self.w1[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                x[:, :kept_modes, :kept_modes, : self.temporal_modes].real,
                self.w1[1],
            )
            + self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes, : self.temporal_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[0],
            )
            - torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[1],
            )
            + self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes, : self.temporal_modes] = (
            torch.einsum(
                "...bi,bio->...bo",
                o1_imag[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[0],
            )
            + torch.einsum(
                "...bi,bio->...bo",
                o1_real[:, :kept_modes, :kept_modes, : self.temporal_modes],
                self.w2[1],
            )
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        # x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        x = x + x_orig
        if self.channel_first:
            x = rearrange(x, "b x y t c -> b c x y t")
        return x


class AFNOBlock(nn.Module):
    def __init__(
        self,
        n_dim=2,
        double_skip=True,
        width=32,
        n_blocks=4,
        mlp_ratio=1.0,
        channel_first=True,
        modes=32,
        non_linearity="gelu",
        num_norm_groups=8,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.norm1 = torch.nn.GroupNorm(num_norm_groups, width)
        self.width = width
        self.modes = modes
        self.non_linearity = ACTIVATION[non_linearity]

        if self.n_dim == 1:
            self.filter = AFNO1D(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hidden_size_factor=1,
                non_linearity=non_linearity,
            )
        elif self.n_dim == 2:
            self.filter = AFNO2D(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hidden_size_factor=1,
                non_linearity=non_linearity,
            )
        elif self.n_dim == 3:
            self.filter = AFNO3D(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hidden_size_factor=1,
                non_linearity=non_linearity,
            )
        else:
            raise NotImplementedError

        self.norm2 = torch.nn.GroupNorm(num_norm_groups, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        Conv = getattr(nn, f"Conv{n_dim}d")
        self.mlp = nn.Sequential(
            Conv(
                in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1
            ),
            self.non_linearity,
            Conv(
                in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1
            ),
        )

        self.double_skip = double_skip

    def forward(self, x, output_shape=None):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        if output_shape is not None:
            x = resample(
                x,
                res_scale=1,
                axis=list(range(2, x.ndim)),
                output_shape=output_shape,
            )
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        n_dim,
        res=64,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        out_dim=128,
        non_linearity="gelu",
        use_cno_block=False,
    ):
        super().__init__()
        self.n_dim = n_dim
        res = [res for _ in range(n_dim)] if isinstance(res, int) else res
        patch_size = [patch_size for _ in range(n_dim)]
        num_patches = np.prod([res[i] // patch_size[i] for i in range(n_dim)])
        self.res = res
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = [res[i] // patch_size[i] for i in range(n_dim)]
        self.out_dim = out_dim
        self.use_cno_block = use_cno_block
        if use_cno_block:
            self.non_linearity = LReLu_torch(
                n_dim, embed_dim, embed_dim, self.out_size, self.out_size
            )
        else:
            self.non_linearity = ACTIVATION[non_linearity]

        Conv = getattr(nn, f"Conv{n_dim}d")
        self.proj = nn.Sequential(
            Conv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.non_linearity,
            Conv(embed_dim, out_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        assert tuple(self.res) == tuple(
            x.shape[2:]
        ), f"Input image size {tuple(x.shape[2:])} does not match model {tuple(self.res)}"
        x = self.proj(x)
        return x


class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type="mlp"):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == "mlp":
            self.w = nn.Parameter(
                1
                / (n_timesteps * out_channels**0.5)
                * torch.randn(n_timesteps, out_channels, out_channels),
                requires_grad=True,
            )  # initialization could be tuned
        elif self.type == "exp_mlp":
            self.w = nn.Parameter(
                1
                / (n_timesteps * out_channels**0.5)
                * torch.randn(n_timesteps, out_channels, out_channels),
                requires_grad=True,
            )  # initialization could be tuned
            self.gamma = nn.Parameter(
                2 ** torch.linspace(-10, 10, out_channels).unsqueeze(0),
                requires_grad=True,
            )  # 1, C

    def forward(self, x):
        """
        :param x: B, (xyz), T, C
        :return: B, (xyz), C
        """
        if self.type == "mlp":
            x = torch.einsum("tij, ...ti->...j", self.w, x)
        elif self.type == "exp_mlp":
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device)  # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum("tij,...ti->...j", self.w, x * t_embed)

        return x
