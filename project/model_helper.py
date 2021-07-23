"""Model Define."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#

import os
import pdb
import collections
from typing import List, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


# Only for typing annotations
RegionParams = collections.namedtuple("RegionParams", ["shift", "covar", "affine"])
MotionParams = collections.namedtuple("MotionParams", ["optical_flow", "occlusion_map"])

# xxxx8888
# from torch.onnx.symbolic_helper import parse_args
# from torch.onnx.symbolic_registry import register_op

# @parse_args('v')
# def motion_svd(g, input):
#     return g.op('onnxservice::svd', input)

# register_op('svd', motion_svd, '', 11)


def svd(covar) -> Tuple[Tensor, Tensor, Tensor]:
    # xxxx8888
    u, s, v = torch.svd(covar)
    s = s.to(covar.device)
    u = u.to(covar.device)
    v = v.to(covar.device)
    # covar.size() -- [10, 2, 2]
    # u.size(), s.size(), v.size() -- [10, 2, 2], [10, 2], [10, 2, 2]

    return u, s, v


class RegionPredictor(nn.Module):
    """
    Region estimating. Estimate affine parameters of the region.
    """

    def __init__(self):
        super(RegionPredictor, self).__init__()

        block_expansion = 32
        num_regions = 10
        num_channels = 3
        max_features = 1024
        num_blocks = 5
        temperature = 0.1
        estimate_affine = True
        scale_factor = 0.25
        pca_based = True

        self.predictor = Hourglass(
            block_expansion,
            in_features=num_channels,
            max_features=max_features,
            num_blocks=num_blocks,
        )

        self.regions = nn.Conv2d(
            in_channels=self.predictor.out_filters,
            out_channels=num_regions,
            kernel_size=(7, 7),
            padding=3,
        )

        # FOMM-like regression based representation
        if estimate_affine and not pca_based:
            self.jacobian = nn.Conv2d(
                in_channels=self.predictor.out_filters,
                out_channels=4,
                kernel_size=(7, 7),
                padding=3,
            )
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1], dtype=torch.float))
        else:
            self.jacobian = None
        # ==> pp self.jacobian == None

        # temperature = 0.1
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.pca_based = pca_based

        # scale_factor = 0.25
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        # block_expansion = 32
        # num_regions = 10
        # num_channels = 3
        # max_features = 1024
        # num_blocks = 5
        # estimate_affine = True
        # scale_factor = 0.25
        # pca_based = True

    def region2affine(self, region) -> Tuple[Tensor, Tensor]:
        # (Pdb) region.shape -- torch.Size([1, 10, 96, 96])
        shape = region.shape
        region = region.unsqueeze(-1)
        # region.size() -- torch.Size([1, 10, 96, 96, 1]) ?

        # region.type() -- 'torch.cuda.FloatTensor'
        grid = (
            make_coordinate_grid(region).unsqueeze_(0).unsqueeze_(0).to(region.device)
        )
        mean = (region * grid).sum(dim=(2, 3))

        # region_params = {'shift': mean}
        shift = mean

        # self.pca_based == True
        # if self.pca_based:
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        covar = covar * region.unsqueeze(-1)
        covar = covar.sum(dim=(2, 3))
        # region_params['covar'] = covar

        # (Pdb) region_params.keys() -- dict_keys(['shift', 'covar'])
        # (Pdb) region_params['shift'].size() -- torch.Size([1, 10, 2])
        # (Pdb) region_params['covar'].size() -- torch.Size([1, 10, 2, 2])

        return shift, covar

    def forward(self, x) -> RegionParams:
        # x.size() -- torch.Size([1, 3, 256, 256])
        # scale_factor = 0.25
        # if self.scale_factor != 1:
        #     x = self.down(x)
        x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.regions(feature_map)

        final_shape = prediction.shape
        region = prediction.view(final_shape[0], final_shape[1], -1)
        region = F.softmax(region / self.temperature, dim=2)
        # region = region.view(*final_shape)
        region = region.view(final_shape)

        shift, covar = self.region2affine(region)
        # region_params['heatmap'] = region
        heatmap = region

        # Regression-based estimation
        # self.jacobian is None
        # if self.jacobian is not None:
        #     jacobian_map = self.jacobian(feature_map)
        #     jacobian_map = jacobian_map.reshape(final_shape[0], 1, 4, final_shape[2],
        #                                         final_shape[3])
        #     region = region.unsqueeze(2)

        #     jacobian = region * jacobian_map
        #     jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
        #     jacobian = jacobian.sum(dim=-1)
        #     jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
        #     region_params['affine'] = jacobian
        #     region_params['covar'] = torch.matmul(jacobian, jacobian.permute(0, 1, 3, 2))
        # elif self.pca_based:
        #     # self.pca_based == True
        #     covar = region_params['covar']
        #     shape = covar.shape
        #     covar = covar.view(-1, 2, 2)
        #     u, s, v = svd(covar)
        #     d = torch.diag_embed(s ** 0.5)
        #     sqrt = torch.matmul(u, d)
        #     sqrt = sqrt.view(*shape)
        #     region_params['affine'] = sqrt
        #     region_params['u'] = u
        #     region_params['d'] = d

        shape = covar.shape
        covar = covar.view(-1, 2, 2)
        u, s, v = svd(covar)
        # xxxx8888
        d = torch.diag_embed(s ** 0.5)

        # d = s ** 0.5
        sqrt = torch.matmul(u, d)
        sqrt = sqrt.view(shape)
        # region_params['affine'] = sqrt
        # region_params['u'] = u
        # region_params['d'] = d

        return RegionParams(shift=shift, covar=covar, affine=sqrt)


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
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


# @torch.jit.script
def make_coordinate_grid(template):
    """
    template: BxCxHxW
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """

    # h = template.size(2)
    # w = template.size(3)
    h = 64
    w = 64

    y = torch.arange(-1.0, 1.0, 2.0 / h) + 1.0 / h
    x = torch.arange(-1.0, 1.0, 2.0 / w) + 1.0 / w
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    return torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)


# @torch.jit.script
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

    mean_sub = coordinate_grid - mean
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
    # xxxx8888
    covar_inverse = torch.inverse(covar).view(shape)
    # covar_inverse.size() -- [1, 10, 1, 1, 2, 2]
    # covar.size() -- [1, 10, 2, 2]

    under_exp = torch.matmul(
        torch.matmul(mean_sub.unsqueeze(-2), covar_inverse), mean_sub.unsqueeze(-1)
    )
    out = torch.exp(-0.5 * under_exp.sum(dim=(-1, -2)))

    return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
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

        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
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
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
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
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
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
            down_blocks.append(
                DownBlock2d(
                    in_features
                    if i == 0
                    else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                )
            )
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
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(
                UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1)
            )

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
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
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
        out = out[:, :, :: self.int_inv_scale, :: self.int_inv_scale]

        return out


class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and transform_region_params
    """

    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        num_regions,
        num_channels,
        estimate_occlusion_map=False,
        scale_factor=1,
        region_var=0.01,
        use_covar_heatmap=False,
        use_deformed_source=True,
        revert_axis_swap=False,
    ):
        super(PixelwiseFlowPredictor, self).__init__()
        # pdb.set_trace()
        # block_expansion = 64
        # num_blocks = 5
        # max_features = 1024
        # num_regions = 10
        # num_channels = 3
        # estimate_occlusion_map = True
        # scale_factor = 0.25
        # region_var = 0.01
        # use_covar_heatmap = True
        # use_deformed_source = True
        # revert_axis_swap = True

        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_regions + 1) * (num_channels * use_deformed_source + 1),
            max_features=max_features,
            num_blocks=num_blocks,
        )

        self.mask = nn.Conv2d(
            self.hourglass.out_filters,
            num_regions + 1,
            kernel_size=(7, 7),
            padding=(3, 3),
        )

        # estimate_occlusion_map = True
        self.occlusion = nn.Conv2d(
            self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3)
        )

        self.num_regions = num_regions
        self.scale_factor = scale_factor
        self.region_var = region_var
        self.use_covar_heatmap = use_covar_heatmap
        self.use_deformed_source = use_deformed_source
        self.revert_axis_swap = revert_axis_swap

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(
        self,
        source_image,
        transform_region_params: RegionParams,
        source_region_params: RegionParams,
    ):
        """
        Eq 6. in the paper H_k(z)
        """
        # (Pdb) source_image.shape -- torch.Size([1, 3, 64, 64])
        # spatial_size = source_image.shape[2:]
        # h = int(source_image.shape[2])
        # w = int(source_image.shape[3])

        # use_covar_heatmap = True
        # covar = self.region_var if not self.use_covar_heatmap else transform_region_params['covar']
        covar = transform_region_params.covar
        gaussian_driving = region2gaussian(
            transform_region_params.shift, covar, source_image
        )

        # use_covar_heatmap = True
        # covar = self.region_var if not self.use_covar_heatmap else source_region_params['covar']
        covar = source_region_params.covar
        gaussian_source = region2gaussian(
            source_region_params.shift, covar, source_image
        )

        heatmap = gaussian_driving - gaussian_source
        # (Pdb) heatmap.size() -- torch.Size([1, 10, 64, 64])

        # adding background feature
        # zeros = torch.zeros(heatmap.shape[0], 1, h, w).to(heatmap.device)
        zeros = torch.zeros_like(source_image)[:, 0:1, :, :]
        heatmap = torch.cat([zeros, heatmap], dim=1).unsqueeze(2)

        # (Pdb) heatmap.size() -- torch.Size([1, 11, 1, 64, 64])

        return heatmap

    def create_sparse_motions(
        self,
        source_image,
        transform_region_params: RegionParams,
        source_region_params: RegionParams,
    ):
        bs, _, h, w = source_image.shape

        identity_grid = make_coordinate_grid(source_image).to(
            source_region_params.shift.device
        )

        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - transform_region_params.shift.view(
            bs, self.num_regions, 1, 1, 2
        )

        # 'affine' in transform_region_params -- True
        # if 'affine' in transform_region_params:
        affine = torch.matmul(
            source_region_params.affine, torch.inverse(transform_region_params.affine)
        )

        # self.revert_axis_swap == True
        # if self.revert_axis_swap:
        #     affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
        affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
        affine = affine.unsqueeze(-3).unsqueeze(-3)
        affine = affine.repeat(1, 1, h, w, 1, 1)
        coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
        coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params.shift.view(
            bs, self.num_regions, 1, 1, 2
        )

        # adding background feature
        bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)
        # (Pdb) driving_to_source.size() -- torch.Size([1, 10, 64, 64, 2])
        # pp sparse_motions.size() -- torch.Size([1, 11, 64, 64, 2])

        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = (
            source_image.unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, self.num_regions + 1, 1, 1, 1, 1)
        )
        source_repeat = source_repeat.view(bs * (self.num_regions + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_regions + 1), h, w, -1))

        # (Pdb) source_repeat.size() -- torch.Size([11, 3, 64, 64])
        # (Pdb) sparse_motions.size() -- torch.Size([11, 64, 64, 2])
        sparse_deformed = F.grid_sample(
            source_repeat, sparse_motions, align_corners=False
        )
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))

        # (Pdb) sparse_deformed.size() -- torch.Size([1, 11, 3, 64, 64])

        return sparse_deformed

    def forward(
        self,
        source_image,
        transform_region_params: RegionParams,
        source_region_params: RegionParams,
    ) -> MotionParams:
        # self.scale_factor == 0.25
        # if self.scale_factor != 1:
        #     source_image = self.down(source_image)
        source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        # out_dict: Dict[str, torch.Tensor] = dict()
        heatmap_representation = self.create_heatmap_representations(
            source_image, transform_region_params, source_region_params
        )
        sparse_motion = self.create_sparse_motions(
            source_image, transform_region_params, source_region_params
        )
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)

        # self.use_deformed_source == True
        # if self.use_deformed_source:
        #     predictor_input = torch.cat([heatmap_representation, deformed_source], dim=2)
        # else:
        #     predictor_input = heatmap_representation
        predictor_input = torch.cat([heatmap_representation, deformed_source], dim=2)

        predictor_input = predictor_input.view(bs, -1, h, w)

        prediction = self.hourglass(predictor_input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1).unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        # out_dict['optical_flow'] = deformation

        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        # out_dict['occlusion_map'] = occlusion_map

        return MotionParams(optical_flow=deformation, occlusion_map=occlusion_map)


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    # __constants__ = ['up_blocks', 'down_blocks']

    def __init__(self):
        super(Generator, self).__init__()

        num_channels = 3
        num_regions = 10
        block_expansion = 64
        max_features = 512
        num_down_blocks = 2
        num_bottleneck_blocks = 6
        pixelwise_flow_predictor_params = {
            "block_expansion": 64,
            "max_features": 1024,
            "num_blocks": 5,
            "scale_factor": 0.25,
            "use_deformed_source": True,
            "use_covar_heatmap": True,
            "estimate_occlusion_map": True,
        }
        skips = True
        revert_axis_swap = True

        self.pixelwise_flow_predictor = PixelwiseFlowPredictor(
            num_regions=num_regions,
            num_channels=num_channels,
            revert_axis_swap=revert_axis_swap,
            **pixelwise_flow_predictor_params
        )

        self.first = SameBlock2d(
            num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3)
        )

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(
                    in_features, out_features, kernel_size=(3, 3), padding=(1, 1)
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(
                max_features, block_expansion * (2 ** (num_down_blocks - i))
            )
            out_features = min(
                max_features, block_expansion * (2 ** (num_down_blocks - i - 1))
            )
            up_blocks.append(
                UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1))
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()

        # max_features == 512, block_expansion == 64, num_down_blocks == 2 ==> in_features == 256
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))

        # num_bottleneck_blocks = 6
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module(
                "r" + str(i),
                ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)),
            )

        self.final = nn.Conv2d(
            block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3)
        )
        self.num_channels = num_channels
        self.skips = skips

    def deform_input(self, inp, optical_flow):
        # Remove "if" for trace model
        # _, h_old, w_old, _ = optical_flow.shape
        # _, _, h, w = inp.shape
        # if h_old != h or w_old != w:
        #     # [b, h, w, 2] ==> [b, 2, h, w]
        #     optical_flow = optical_flow.permute(0, 3, 1, 2)
        #     # Flow smoothing ...
        #     optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear', align_corners=False)
        #     optical_flow = optical_flow.permute(0, 2, 3, 1)
        optical_flow = optical_flow.permute(0, 3, 1, 2)
        # Flow smoothing ...
        optical_flow = F.interpolate(
            optical_flow, size=inp.shape[2:], mode="bilinear", align_corners=False
        )
        optical_flow = optical_flow.permute(0, 2, 3, 1)

        return F.grid_sample(inp, optical_flow, align_corners=False)

    def apply_optical_with_prev(
        self, input_previous, input_skip, motion_params: MotionParams
    ):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params.occlusion_map
        deformation = motion_params.optical_flow
        input_skip = self.deform_input(input_skip, deformation)
        # Remove "if" for trace model
        # if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
        #     occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        occlusion_map = F.interpolate(
            occlusion_map,
            size=input_skip.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        # input_previous != None
        return input_skip * occlusion_map + input_previous * (1 - occlusion_map)

    def apply_optical_without_prev(self, input_skip, motion_params: MotionParams):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params.occlusion_map
        deformation = motion_params.optical_flow
        input_skip = self.deform_input(input_skip, deformation)
        # Remove "if" for trace model
        # if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
        #     occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        occlusion_map = F.interpolate(
            occlusion_map,
            size=input_skip.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return input_skip * occlusion_map

    def forward(
        self,
        source_image,
        source_region_params: RegionParams,
        transform_region_params: RegionParams,
    ):
        # Remove "if" for trace model

        out = self.first(source_image)
        skips: List[Tensor] = [out]
        # for i in range(len(self.down_blocks)):
        #     out = self.down_blocks[i](out)
        #     skips.append(out)
        for mod in self.down_blocks:
            out = mod(out)
            skips.append(out)

        # output_dict: Dict[str, Tensor] = {}

        motion_params = self.pixelwise_flow_predictor(
            source_image=source_image,
            transform_region_params=transform_region_params,
            source_region_params=source_region_params,
        )
        # output_dict["deformed"] = self.deform_input(source_image, motion_params.optical_flow)
        # if 'occlusion_map' in motion_params:
        #     output_dict['occlusion_map'] = motion_params['occlusion_map']
        # output_dict['occlusion_map'] = motion_params.occlusion_map

        out = self.apply_optical_without_prev(out, motion_params)
        out = self.bottleneck(out)
        i = 0
        for mod in self.up_blocks:
            # self.skips -- True
            # if self.skips:
            #     out = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = self.apply_optical_with_prev(out, skips[-(i + 1)], motion_params)
            out = mod(out)
            i = i + 1

        # self.skips -- True
        # if self.skips:
        #     out = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out = self.apply_optical_with_prev(out, skips[0], motion_params)

        out = self.final(out)
        out = torch.sigmoid(out)

        # self.skips -- True
        # if self.skips:
        #     out = self.apply_optical(input_skip=source_image, input_previous=out, motion_params=motion_params)
        out = self.apply_optical_with_prev(out, source_image, motion_params)

        # output_dict["prediction"] = out

        # return output_dict
        return out


class AVDNetwork(nn.Module):
    """
    Animation via Disentanglement network
    """

    def __init__(self):
        super(AVDNetwork, self).__init__()
        num_regions = 10
        id_bottle_size = 64
        pose_bottle_size = 64
        revert_axis_swap = True

        input_size = (2 + 4) * num_regions
        self.num_regions = num_regions
        self.revert_axis_swap = revert_axis_swap

        self.id_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, id_bottle_size),
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, pose_bottle_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(pose_bottle_size + id_bottle_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size),
        )

    def region_params_to_emb(self, shift, affine):
        # affine.size() -- torch.Size([1, 10, 2, 2])
        emb = torch.cat(
            [shift, affine.view(affine.shape[0], affine.shape[1], -1)], dim=-1
        )
        emb = emb.view(emb.shape[0], -1)
        # emb.size() -- torch.Size([1, 60])
        return emb

    def emb_to_region_params(self, emb) -> Tuple[Tensor, Tensor]:
        emb = emb.view(emb.shape[0], self.num_regions, 6)
        shift = emb[:, :, :2]
        affine = emb[:, :, 2:].view(emb.shape[0], emb.shape[1], 2, 2)
        return shift, affine

    def forward(self, x_id: RegionParams, x_pose: RegionParams) -> RegionParams:
        # (Pdb) pp x_id.keys()
        # dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
        # (Pdb) pp x_pose.keys()
        # dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])

        # self.revert_axis_swap -- True
        # if self.revert_axis_swap:
        #     affine = torch.matmul(x_id['affine'], torch.inverse(x_pose['affine']))
        #     sign = torch.sign(affine[:, :, 0:1, 0:1])
        #     x_id = {'affine': x_id['affine'] * sign, 'shift': x_id['shift']}
        affine = torch.matmul(x_id.affine, torch.inverse(x_pose.affine))
        sign = torch.sign(affine[:, :, 0:1, 0:1])
        # x_id = {'affine': x_id['affine'] * sign, 'shift': x_id['shift']}

        pose_emb = self.pose_encoder(
            self.region_params_to_emb(x_pose.shift, x_pose.affine)
        )
        id_emb = self.id_encoder(
            self.region_params_to_emb(x_id.shift, x_id.affine * sign)
        )

        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

        shift, affine = self.emb_to_region_params(rec)
        covar = torch.matmul(affine, affine.permute(0, 1, 3, 2))

        # ['shift', 'covar', 'affine']
        return RegionParams(shift=shift, covar=covar, affine=affine)


class MotionDriving(nn.Module):
    def __init__(self):
        super(MotionDriving, self).__init__()
        self.generator = Generator()
        self.region_predictor = RegionPredictor()
        self.avd_network = AVDNetwork()

    def load_weights(self, checkpoint: str):
        state = torch.load(checkpoint, map_location=torch.device("cpu"))
        self.generator.load_state_dict(state["generator"])
        self.region_predictor.load_state_dict(state["region_predictor"])
        self.avd_network.load_state_dict(state["avd_network"])

        self.generator.eval()
        self.region_predictor.eval()
        self.avd_network.eval()

    def forward(self, source_image, driving_image):
        source_params: RegionParams = self.region_predictor(source_image)

        # pp source_image.shape -- torch.Size([1, 3, 384, 384])
        # source_params.keys() -- dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
        # (Pdb) source_params['shift'].size() -- torch.Size([1, 10, 2])
        # (Pdb) source_params['covar'].size() -- torch.Size([1, 10, 2, 2])
        # (Pdb) source_params['affine'].size() -- torch.Size([1, 10, 2, 2])

        # now image is driving frame
        driving_params: RegionParams = self.region_predictor(driving_image)
        transform_params: RegionParams = self.avd_network(source_params, driving_params)

        return self.generator(source_image, source_params, transform_params)


if __name__ == "__main__":
    model = MotionDriving()
    model.load_weights("models/image_motion.pth")
    model = model.eval()
    print(model)
