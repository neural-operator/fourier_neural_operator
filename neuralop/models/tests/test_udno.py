import pytest
import torch
from neuralop.models import UDNO
import pprint
import torch
from torch import nn
from torch.nn import functional as F


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize(
    "in_shape",
    [(32, 32), (33, 33), (64, 64), (123, 123), (512, 512)],
    ids=lambda x: f"in_shape={x}",
)
@pytest.mark.parametrize(
    "hidden_channels",
    [2, 16, 17, 24, 64],
    ids=lambda x: f"chans={x}",
)
@pytest.mark.parametrize(
    "num_pool_layers",
    [1, 2, 4],
    ids=lambda x: f"num_pool_layers={x}",
)
def test_udno(in_shape, hidden_channels, num_pool_layers):
    try:
        sel_device = get_device()
        in_batch_size = 2
        in_channels = 3
        out_channels = 1
        sample_input = torch.randn(in_batch_size, in_channels, *in_shape).to(sel_device)

        # create models
        unet = Unet(
            in_channels,
            out_channels,
            hidden_channels,
            num_pool_layers,
            drop_prob=0.1,
        ).to(sel_device)
        udno = UDNO(
            in_channels,
            out_channels,
            in_shape,
            disco_radius_cutoff=None,
            hidden_channels=hidden_channels,
            num_pool_layers=num_pool_layers,
            disco_kernel_shape=[2, 3],
            disco_kernel_bias=False,
            drop_prob=0.1,
        ).to(sel_device)

        unet.eval()
        udno.eval()

        # test forward pass
        unet_output = unet(sample_input)
        udno_output = udno(sample_input)

        # test backward pass
        loss = udno_output.sum()
        loss.backward()

        # assert unused params = 0
        n_unused_params = 0
        for param in udno.parameters():
            if param.grad is None:
                n_unused_params += 1
        assert n_unused_params == 0, f"{n_unused_params} parameters were unused!"

        del sample_input, unet, udno
        # assert that the output shapes are the same
        assert (
            unet_output.shape == udno_output.shape
        ), f"Output of UDNO is: {udno_output.shape} vs. expected U-Net output: {unet_output.shape}"
    except RuntimeError as e:
        if "can't allocate memory" in str(e):
            pytest.skip(f"Skipped due to memory allocation error: {e}")
        else:
            raise


"""
Reference U-Net implementation from Meta/Facebook: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py
"""


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
