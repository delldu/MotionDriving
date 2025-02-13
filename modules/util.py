"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from typing import List

import pdb


class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs

    During evaluation, this running mean/variance is used for normalization.

    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm

    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)



@torch.jit.script
def make_coordinate_grid(template):
    """
    template: BxCxHxW
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h = template.size(2)
    w = template.size(3)
    y = torch.arange(-1.0, 1.0, 2.0/h) + 1.0/h
    x = torch.arange(-1.0, 1.0, 2.0/w) + 1.0/w
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    return torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

@torch.jit.script
def region2gaussian(center, covar, template):
    """
    Transform a region parameters into gaussian like heatmap
    """
    # spatial_size = torch.Size([64, 64])
    # center.size() -- torch.Size([1, 10, 2])

    mean = center

    coordinate_grid = make_coordinate_grid(template).to(mean.device)
    number_of_leading_dimensions = len(mean.shape) - 1
    # number_of_leading_dimensions -- 2

    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    # coordinate_grid = coordinate_grid.view(*shape)
    coordinate_grid = coordinate_grid.view(shape)

    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    # coordinate_grid = coordinate_grid.repeat(*repeats)
    coordinate_grid = coordinate_grid.repeat(repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    # mean = mean.view(*shape)
    mean = mean.view(shape)

    mean_sub = (coordinate_grid - mean)
    # type(covar) -- <class 'torch.Tensor'>
    # ==> type(covar) == float, False
    # if type(covar) == float:
    #     out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / covar)
    # else:
    #     shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
    #     covar_inverse = torch.inverse(covar).view(*shape)
    #     under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), covar_inverse), mean_sub.unsqueeze(-1))
    #     out = torch.exp(-0.5 * under_exp.sum(dim=(-1, -2)))

    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2, 2)
    covar_inverse = torch.inverse(covar).view(shape)
    under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), covar_inverse), mean_sub.unsqueeze(-1))
    out = torch.exp(-0.5 * under_exp.sum(dim=(-1, -2)))

    return out

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)
        # in_features = 256
        # kernel_size = (3, 3)
        # padding = (1, 1)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        # x.size() -- torch.Size([1, 1024, 2, 2])
        out = F.interpolate(x, scale_factor=2.0)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        # ==> pdb.set_trace()
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        # ==> pdb.set_trace()
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """
    # __constants__ = ['down_blocks']

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x.size() -- torch.Size([1, 3, 64, 64])
        outs: List[torch.Tensor] = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """
    # __constants__ = ['up_blocks']

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
        # (Pdb) self.up_blocks
        # ModuleList(
        #   (0): UpBlock2d(
        #     (conv): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm): SynchronizedBatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        #   (1): UpBlock2d(
        #     (conv): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm): SynchronizedBatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        #   (2): UpBlock2d(
        #     (conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm): SynchronizedBatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        #   (3): UpBlock2d(
        #     (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm): SynchronizedBatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        #   (4): UpBlock2d(
        #     (conv): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #     (norm): SynchronizedBatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   )
        # )

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # x[0].size(), x[1].size(), x[2].size(), x[3].size(), x[4].size(), x[5].size()
        # (torch.Size([1, 3, 64, 64]), torch.Size([1, 64, 32, 32]), 
        # torch.Size([1, 128, 16, 16]), torch.Size([1, 256, 8, 8]), 
        # torch.Size([1, 512, 4, 4]), torch.Size([1, 1024, 2, 2]))
        out = x.pop()
        # out.size() -- torch.Size([1, 1024, 2, 2])
        # len(x) -- 5, from 6 to 5

        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)

        # out.size() -- torch.Size([1, 35, 64, 64])
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        # block_expansion = 64
        # in_features = 44
        # num_blocks = 5
        # max_features = 1024

        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters
        # self.out_filters -- 108

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        # channels = 3
        # scale = 0.25
        sigma = (1 / scale - 1) / 2
        # ==> sigma -- 1.5
        kernel_size = 2 * round(sigma * 4) + 1
        # kernel_size -- 13
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        # self.ka, self.kb -- (6, 6)

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)
        # pp self.int_inv_scale -- 4

    def forward(self, input):
        # self.scale -- 0.25
        # if self.scale == 1.0:
        #     return input
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
