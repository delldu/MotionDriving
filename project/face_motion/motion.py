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

import collections
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

# Only for typing annotations
RegionParams = collections.namedtuple("RegionParams", ["shift", "covar", "affine"])
MotionParams = collections.namedtuple("MotionParams", ["optical_flow", "occlusion_map"])


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
        scale_factor = 0.25

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

        self.temperature = 0.1
        self.scale_factor = scale_factor

        # scale_factor = 0.25
        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        # block_expansion = 32
        # num_regions = 10
        # num_channels = 3
        # max_features = 1024
        # num_blocks = 5
        self.grid = make_coordinate_grid()

    def region2affine(self, region) -> Tuple[Tensor, Tensor]:
        # region.shape -- [1, 10, 64, 64]
        region = region.unsqueeze(-1)
        # region.size() -- [1, 10, 64, 64, 1]

        grid = self.grid.unsqueeze(0).unsqueeze(0).to(region.device)

        # grid.size() -- [1, 1, 64, 64, 2]
        mean = (region * grid).sum(dim=(2, 3))

        shift = mean
        # mean.size() -- [1, 10, 2]

        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        # mean.unsqueeze(-2).unsqueeze(-2) -- [1, 10, 1, 1, 2]
        # mean_sub.size() -- [1, 1, 64, 64, 2]

        covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        covar = covar * region.unsqueeze(-1)
        # covar.size() -- [1, 10, 64, 64, 2, 2]

        covar = covar.sum(dim=(2, 3))

        # shift'.size() -- [1, 10, 2], covar'.size() -- [1, 10, 2, 2]
        return shift, covar

    def forward(self, x) -> RegionParams:
        # x.size() -- torch.Size([1, 3, 256, 256])
        x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.regions(feature_map)

        final_shape = prediction.shape
        region = prediction.view(final_shape[0], final_shape[1], -1)
        region = F.softmax(region / self.temperature, dim=2)
        region = region.view(final_shape)

        shift, covar = self.region2affine(region)

        # Regression-based estimation
        shape = covar.shape
        # shape == [1, 10, 2, 2]
        covar = covar.view(-1, 2, 2)
        # covar.size() == [10, 2, 2]

        # svd python prototype:
        #   torch.svd(input, some=True, compute_uv=True, *, out=None) -> (Tensor, Tensor, Tensor)
        u, s, v = torch.svd(covar)

        d = torch.diag_embed(s ** 0.5)

        # s ** 0.5.size() == [10, 2] ==> d.size() == [10, 2, 2]
        sqrt = torch.matmul(u, d)
        sqrt = sqrt.view(shape)

        # sqrt.size() -- [1, 10, 2, 2]

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


def make_coordinate_grid():
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h = 64
    w = 64

    y = torch.arange(-1.0, 1.0, 2.0 / h) + 1.0 / h
    x = torch.arange(-1.0, 1.0, 2.0 / w) + 1.0 / w
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    return torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)


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
                    in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
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
        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
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
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, :: self.int_inv_scale, :: self.int_inv_scale]

        return out


class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and trans_region_params
    """

    def __init__(
        self,
        block_expansion,
        num_blocks,
        max_features,
        num_regions,
        num_channels,
    ):
        super(PixelwiseFlowPredictor, self).__init__()
        # block_expansion = 64
        # num_blocks = 5
        # max_features = 1024
        # num_regions = 10
        # num_channels = 3

        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_regions + 1) * (num_channels + 1),
            max_features=max_features,
            num_blocks=num_blocks,
        )

        self.mask = nn.Conv2d(
            self.hourglass.out_filters,
            num_regions + 1,
            kernel_size=(7, 7),
            padding=(3, 3),
        )

        self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_regions = num_regions
        self.scale_factor = 0.25

        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)
        self.grid = make_coordinate_grid()

    def region2gaussian(self, mean, covar):
        """
        Transform a region parameters into gaussian like heatmap
        """
        # mean.size() -- [1, 10, 2]

        coordinate_grid = self.grid.to(mean.device)
        # coordinate_grid.size() -- [64, 64, 2]
        leading_dimensions = len(mean.shape) - 1
        # leading_dimensions -- 2

        shape = (1,) * leading_dimensions + coordinate_grid.shape
        # shape ==> (1, 1, 64, 64, 2)
        coordinate_grid = coordinate_grid.view(shape)

        repeats = mean.shape[:leading_dimensions] + (1, 1, 1)
        # repeats ==> (1, 10, 1, 1, 1)
        coordinate_grid = coordinate_grid.repeat(repeats)
        # ==> (1, 10, 64, 64, 2)

        # Preprocess kp shape
        shape = mean.shape[:leading_dimensions] + (1, 1, 2)
        # ==> (1, 10,   1, 1, 2)
        mean = mean.view(shape)

        mean_sub = coordinate_grid - mean

        shape = mean.shape[:leading_dimensions] + (1, 1, 2, 2)
        # ==> (1, 10,   1, 1, 2, 2,)
        covar_inverse = torch.inverse(covar).view(shape)
        # covar_inverse.size() -- [1, 10, 1, 1, 2, 2]
        # covar.size() -- [1, 10, 2, 2]

        # mean_sub.size() -- [1, 10, 64, 64, 2]
        under_exp = torch.matmul(torch.matmul(mean_sub.unsqueeze(-2), covar_inverse), mean_sub.unsqueeze(-1))
        # under_exp.size() -- [1, 10, 64, 64, 1, 1]

        out = torch.exp(-0.5 * under_exp.sum(dim=(-1, -2)))
        # out.size() -- [1, 10, 64, 64]
        return out

    def create_heatmap(
        self,
        source_image,
        trans_region_params: RegionParams,
        source_region_params: RegionParams,
    ):
        """
        Eq 6. in the paper H_k(z)
        """
        # source_image.shape -- [1, 3, 64, 64]

        covar = trans_region_params.covar
        gaussian_driving = self.region2gaussian(trans_region_params.shift, covar)

        covar = source_region_params.covar
        gaussian_source = self.region2gaussian(source_region_params.shift, covar)

        heatmap = gaussian_driving - gaussian_source
        # heatmap.size() -- [1, 10, 64, 64]

        # adding background feature
        # zeros = torch.zeros(heatmap.shape[0], 1, h, w).to(heatmap.device)
        zeros = torch.zeros_like(source_image)[:, 0:1, :, :]
        heatmap = torch.cat([zeros, heatmap], dim=1).unsqueeze(2)

        # heatmap.size() -- [1, 11, 1, 64, 64]
        return heatmap

    def create_sparse_motions(
        self,
        source_image,
        trans_region_params: RegionParams,
        source_region_params: RegionParams,
    ):
        bs, _, h, w = source_image.shape

        id_grid = self.grid.to(source_region_params.shift.device)

        id_grid = id_grid.view(1, 1, h, w, 2)
        coordinate_grid = id_grid - trans_region_params.shift.view(bs, self.num_regions, 1, 1, 2)

        # 'affine' in trans_region_params -- True
        affine = torch.matmul(source_region_params.affine, torch.inverse(trans_region_params.affine))

        affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
        affine = affine.unsqueeze(-3).unsqueeze(-3)
        affine = affine.repeat(1, 1, h, w, 1, 1)
        coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
        coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params.shift.view(bs, self.num_regions, 1, 1, 2)

        # adding background feature
        bg_grid = id_grid.repeat(bs, 1, 1, 1, 1)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)
        # driving_to_source.size() -- [1, 10, 64, 64, 2]
        # sparse_motions.size() -- [1, 11, 64, 64, 2]

        return sparse_motions

    def create_deformed_source(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_regions + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_regions + 1), -1, h, w)
        sparse_motions = sparse_motions.view(bs * (self.num_regions + 1), h, w, -1)

        # source_repeat.size() -- [11, 3, 64, 64]
        # sparse_motions.size() -- [11, 64, 64, 2]
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))
        # sparse_deformed.size() -- [1, 11, 3, 64, 64]

        return sparse_deformed

    def forward(
        self,
        source_image,
        trans_region_params: RegionParams,
        source_region_params: RegionParams,
    ) -> MotionParams:
        source_image = self.down(source_image)
        bs, _, h, w = source_image.shape

        heatmap = self.create_heatmap(source_image, trans_region_params, source_region_params)
        sparse_motion = self.create_sparse_motions(source_image, trans_region_params, source_region_params)
        # sparse_motion.size() -- [1, 11, 64, 64, 2]
        deformed_source = self.create_deformed_source(source_image, sparse_motion)

        # heatmap.size()         -- [1, 11, 1, 64, 64]
        # deformed_source.size() -- [1, 11, 3, 64, 64]
        predictor_input = torch.cat([heatmap, deformed_source], dim=2)
        predictor_input = predictor_input.view(bs, -1, h, w)
        # predictor_input.size() -- [1, 44, 64, 64]

        prediction = self.hourglass(predictor_input)
        # prediction.size() -- [1, 108, 64, 64]

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1).unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        # sparse_motion.size() -- [1, 11, 2, 64, 64]
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        occlusion_map = torch.sigmoid(self.occlusion(prediction))

        # deformation.size() -- [1, 64, 64, 2]
        # occlusion_map.size() -- [1, 1, 64, 64]
        return MotionParams(optical_flow=deformation, occlusion_map=occlusion_map)


class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform
    image according to movement trajectories induced by region parameters.
    Generator follows Johnson architecture.
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
        }
        skips = True

        self.pixelwise_flow_predictor = PixelwiseFlowPredictor(
            num_regions=num_regions, num_channels=num_channels, **pixelwise_flow_predictor_params
        )

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
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

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels
        self.skips = skips

    def deform_input(self, input, optical_flow):
        # input.size() -- [1, 256, 64, 64]
        # optical_flow.size() -- [1, 64, 64, 2]

        optical_flow = optical_flow.permute(0, 3, 1, 2)

        # Flow smoothing ...
        optical_flow = F.interpolate(optical_flow, size=input.shape[2:], mode="bilinear", align_corners=False)
        optical_flow = optical_flow.permute(0, 2, 3, 1)

        return F.grid_sample(input, optical_flow, align_corners=False)

    def apply_optical_with_prev(self, input_previous, input_skip, motion_params: MotionParams):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params.occlusion_map
        deformation = motion_params.optical_flow
        input_skip = self.deform_input(input_skip, deformation)
        # Remove "if" for trace model
        occlusion_map = F.interpolate(
            occlusion_map,
            size=input_skip.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return input_skip * occlusion_map + input_previous * (1 - occlusion_map)

    def apply_optical_without_prev(self, input_skip, motion_params: MotionParams):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params.occlusion_map
        deformation = motion_params.optical_flow
        input_skip = self.deform_input(input_skip, deformation)
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
        trans_region_params: RegionParams,
    ):
        out = self.first(source_image)
        skips: List[Tensor] = [out]
        for mod in self.down_blocks:
            out = mod(out)
            skips.append(out)

        motion_params = self.pixelwise_flow_predictor(
            source_image=source_image,
            trans_region_params=trans_region_params,
            source_region_params=source_region_params,
        )

        out = self.apply_optical_without_prev(out, motion_params)
        out = self.bottleneck(out)
        i = 0
        for mod in self.up_blocks:
            out = self.apply_optical_with_prev(out, skips[-(i + 1)], motion_params)
            out = mod(out)
            i = i + 1

        out = self.apply_optical_with_prev(out, skips[0], motion_params)

        out = self.final(out)
        out = torch.sigmoid(out)

        out = self.apply_optical_with_prev(out, source_image, motion_params)

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

        input_size = (2 + 4) * num_regions
        self.num_regions = num_regions

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
        emb = torch.cat([shift, affine.view(affine.shape[0], affine.shape[1], -1)], dim=-1)
        emb = emb.view(emb.shape[0], -1)
        # emb.size() -- torch.Size([1, 60])
        return emb

    def emb_to_region_params(self, emb) -> Tuple[Tensor, Tensor]:
        emb = emb.view(emb.shape[0], self.num_regions, 6)
        shift = emb[:, :, :2]
        affine = emb[:, :, 2:].view(emb.shape[0], emb.shape[1], 2, 2)
        return shift, affine

    def forward(self, x_id: RegionParams, x_pose: RegionParams) -> RegionParams:
        # x_id - ['shift', 'covar', 'affine']
        # x_pose - ['shift', 'covar', 'affine']
        # shift.size(), covar.size(), affine.size()
        # [1, 10, 2], [10, 2, 2], [1, 10, 2, 2]

        affine = torch.matmul(x_id.affine, torch.inverse(x_pose.affine))
        sign = torch.sign(affine[:, :, 0:1, 0:1])

        pose_emb = self.pose_encoder(self.region_params_to_emb(x_pose.shift, x_pose.affine))
        id_emb = self.id_encoder(self.region_params_to_emb(x_id.shift, x_id.affine * sign))

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

        # pp source_image.shape -- [1, 3, 256, 256]
        # source_params.keys() -- dict_keys(['shift', 'covar', 'affine'])
        # (Pdb) source_params['shift'].size() -- [1, 10, 2]
        # (Pdb) source_params['covar'].size() -- [1, 10, 2, 2]
        # (Pdb) source_params['affine'].size() -- [1, 10, 2, 2]

        # now image is driving frame
        driving_params: RegionParams = self.region_predictor(driving_image)
        transform_params: RegionParams = self.avd_network(source_params, driving_params)

        return self.generator(source_image, source_params, transform_params)
