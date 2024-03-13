#!/usr/bin/env python
# -*- coding:utf-8 _*-
import torch
import torch.nn as nn


import math
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class LReLu_torch(torch.nn.Module):
    def __init__(
        self,
        n_dim,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        in_size,  # Input spatial size: int or [width, height].
        out_size,  # Output spatial size: int or [width, height].
        sampling_rate=None,  # Input sampling rate (s). None or int
    ):
        super().__init__()
        self.n_dim = n_dim
        self.activation = nn.LeakyReLU()

        self.in_channels = in_channels
        self.out_channels = out_channels

        assert in_channels == out_channels

        # self.in_size = torch.LongTensor([in_size for _ in range(self.n_dim)]) if isinstance(in_size, int) else torch.LongTensor(in_size)
        # self.out_size = torch.LongTensor([out_size for _ in range(self.n_dim)]) if isinstance(out_size, int) else torch.LongTensor(out_size)
        self.in_size = (
            [in_size for _ in range(self.n_dim)]
            if isinstance(in_size, int)
            else in_size
        )
        self.out_size = (
            [out_size for _ in range(self.n_dim)]
            if isinstance(out_size, int)
            else out_size
        )

        self.sampling_rate = (
            sampling_rate  ## if specify sampling rate, will ignore in_size/out_size
        )

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        if self.n_dim == 1:
            self.mode = "linear"
            self.antialias_option = False
        elif self.n_dim == 2:
            self.mode = "bilinear"
            self.antialias_option = True
        elif self.n_dim == 3:
            self.mode = "trilinear"
            self.antialias_option = False
        else:
            raise NotImplementedError

        # ------------------------------------------------------------------------------------------------

    def forward(self, x):
        in_size = self.in_size if self.sampling_rate is None else x.shape[2:]
        out_size = (
            self.out_size
            if self.sampling_rate is None
            else [self.sampling_rate * _ for _ in x.shape[2:]]
        )
        x = nn.functional.interpolate(
            x,
            size=[2 * s for s in in_size],
            mode=self.mode,
            antialias=self.antialias_option,
        )
        x = self.activation(x)
        x = nn.functional.interpolate(
            x, size=in_size, mode=self.mode, antialias=self.antialias_option
        )
        if self.in_size != self.out_size:
            x = nn.functional.interpolate(
                x, size=out_size, mode=self.mode, antialias=self.antialias_option
            )

        if self.n_dim == 1:
            x = rearrange(x, "b c x -> b x c")
            x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
            x = rearrange(x, "b x c -> b c x")
        elif self.n_dim == 2:
            x = rearrange(x, "b c x y -> b x y c")
            x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
            x = rearrange(x, "b x y c -> b c x y")
        elif self.n_dim == 3:
            x = rearrange(x, "b c x y z -> b x y z c")
            x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
            x = rearrange(x, "b x y z c -> b c x y z")
        else:
            raise NotImplementedError
        return x


class CNOBlock(nn.Module):
    def __init__(
        self,
        n_dim,
        in_channels,
        out_channels,
        in_size,
        out_size,
        sampling_rate=None,
        conv_kernel=3,
    ):
        super(CNOBlock, self).__init__()

        self.n_dim = n_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = (
            [in_size for _ in range(self.n_dim)]
            if isinstance(in_size, int)
            else in_size
        )
        self.out_size = (
            [out_size for _ in range(self.n_dim)]
            if isinstance(out_size, int)
            else out_size
        )

        self.sampling_rate = sampling_rate
        self.conv_kernel = conv_kernel

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        pad = (self.conv_kernel - 1) // 2

        Conv = getattr(nn, f"Conv{n_dim}d")
        self.convolution = Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.conv_kernel,
            padding=pad,
        )

        self.filter_frequency = getattr(self, f"filter_frequency{n_dim}d")

        self.activation = LReLu_torch(
            n_dim=n_dim,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            in_size=self.in_size,
            out_size=self.out_size,
            sampling_rate=self.sampling_rate,
        )

    def filter_frequency1d(self, x, K):
        """
        Apply low-pass filter to a 1D tensor with batch and channel dimensions.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, C, M).
            K (int): The ratio of frequency components to retain.

        Returns:
            torch.Tensor: The filtered tensor.
        """
        # Step 1: FFT along the last dimension
        fft_tensor = torch.fft.fft(x, dim=-1)

        # Step 2: Create low-pass filter mask
        N, C, M = x.shape
        mask = torch.zeros(M, dtype=torch.bool).to(x.device)
        cutoff = M // K  # Define the cutoff frequency based on the ratio K
        mask[:cutoff] = 1  # Retain 1/K of the frequency components
        mask = mask[None, None, :]  # Adjust dimensions to match the tensor (N, C, M)
        mask = mask.expand(N, C, M)  # Expand mask to match tensor dimensions

        # Step 3: Apply filter
        filtered_fft_tensor = fft_tensor * mask

        # Step 4: Inverse FFT
        filtered_tensor = torch.fft.ifft(
            filtered_fft_tensor, dim=-1
        ).real  # Get the real part after inverse FFT

        return filtered_tensor

    def filter_frequency2d(self, x, K):
        """
        Apply low-pass filter to a tensor with shape (B, C, M, M).

        Parameters:
            tensor (torch.Tensor): Input tensor of shape (B, C, M, M).
            K (int): The ratio of frequency components to retain.

        Returns:
            torch.Tensor: The filtered tensor.
        """
        # Step 1: FFT
        fft_tensor = torch.fft.fft2(x)

        # Step 2: Create low-pass filter mask
        B, C, M, N = x.shape
        mask = torch.zeros((M, N), dtype=torch.bool).to(x.device)
        cutoff = M // K  # Define the cutoff frequency based on the ratio K
        mask[:cutoff, :cutoff] = 1  # Retain 1/K of the frequency components
        mask = mask[
            None, None, :, :
        ]  # Adjust dimensions to match the tensor (B, C, M, M)
        mask = mask.expand(B, C, M, N)  # Expand mask to match tensor dimensions

        # Step 3: Apply filter
        filtered_fft_tensor = fft_tensor * mask

        # Step 4: Inverse FFT
        filtered_tensor = torch.fft.ifft2(
            filtered_fft_tensor
        ).real  # Get the real part after inverse FFT

        return filtered_tensor

    def filter_frequency3d(self, x, K):
        """
        Apply low-pass filter to a 3D tensor with shape (B, C, X, Y, Z).

        Parameters:
            x (torch.Tensor): Input tensor of shape (B, C, X, Y, Z).
            K (int): The ratio of frequency components to retain.

        Returns:
            torch.Tensor: The filtered tensor.
        """
        # Step 1: FFT
        fft_tensor = torch.fft.fftn(x, dim=(-3, -2, -1))

        # Step 2: Create low-pass filter mask
        B, C, X, Y, Z = x.shape
        mask = torch.zeros((X, Y, Z), dtype=torch.bool).to(x.device)
        cutoff_x = X // K
        cutoff_y = Y // K
        cutoff_z = (
            Z // K
        )  # Define the cutoff frequency for each dimension based on the ratio K
        mask[:cutoff_x, :cutoff_y, :cutoff_z] = (
            1  # Retain 1/K of the frequency components
        )
        mask = mask[
            None, None, :, :, :
        ]  # Adjust dimensions to match the tensor (B, C, X, Y, Z)
        mask = mask.expand(B, C, X, Y, Z)  # Expand mask to match tensor dimensions

        # Step 3: Apply filter
        filtered_fft_tensor = fft_tensor * mask

        # Step 4: Inverse FFT
        filtered_tensor = torch.fft.ifftn(
            filtered_fft_tensor, dim=(-3, -2, -1)
        ).real  # Get the real part after inverse FFT

        return filtered_tensor

    def forward(self, x):
        x = self.filter_frequency(x, self.conv_kernel)
        x = self.convolution(x)
        return self.activation(x)


if __name__ == "__main__":
    # 1d
    x = torch.rand(20, 3, 32)
    net = CNOBlock(n_dim=1, in_channels=3, out_channels=5, in_size=32, out_size=32)
    print(net(x).shape)
    ## 2d
    x = torch.rand(20, 3, 32, 32)
    net = CNOBlock(
        n_dim=2, in_channels=3, out_channels=5, in_size=(32, 32), out_size=(32, 48)
    )
    print(net(x).shape)
    ## 3d
    x = torch.rand(20, 3, 32, 32, 32)
    net = CNOBlock(n_dim=3, in_channels=3, out_channels=5, in_size=32, out_size=32)
    print(net(x).shape)
