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
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, region2gaussian
from typing import Dict, List
import collections

import pdb
# Only for typing annotations
Tensor = torch.Tensor
RegionParams = collections.namedtuple('RegionParams', ['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
TransformParams = collections.namedtuple('TransformParams', ['shift', 'covar', 'affine'])

MotionParams = collections.namedtuple('MotionParams', ['optical_flow', 'occlusion_map'])


class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and transform_region_params
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_regions, num_channels,
                 estimate_occlusion_map=False, scale_factor=1, region_var=0.01,
                 use_covar_heatmap=False, use_deformed_source=True, revert_axis_swap=False):
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

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=(num_regions + 1) * (num_channels * use_deformed_source + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_regions + 1, kernel_size=(7, 7), padding=(3, 3))

        # estimate_occlusion_map = True
        self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))

        self.num_regions = num_regions
        self.scale_factor = scale_factor
        self.region_var = region_var
        self.use_covar_heatmap = use_covar_heatmap
        self.use_deformed_source = use_deformed_source
        self.revert_axis_swap = revert_axis_swap

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def create_heatmap_representations(self, source_image, 
        transform_region_params: TransformParams, source_region_params: RegionParams):
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
        gaussian_driving = region2gaussian(transform_region_params.shift, covar, source_image)

        # use_covar_heatmap = True
        # covar = self.region_var if not self.use_covar_heatmap else source_region_params['covar']
        covar = source_region_params.covar
        gaussian_source = region2gaussian(source_region_params.shift, covar, source_image)

        heatmap = gaussian_driving - gaussian_source
        # (Pdb) heatmap.size() -- torch.Size([1, 10, 64, 64])

        # adding background feature
        # zeros = torch.zeros(heatmap.shape[0], 1, h, w).to(heatmap.device)
        zeros = torch.zeros_like(source_image)[:, 0:1, :, :]
        heatmap = torch.cat([zeros, heatmap], dim=1).unsqueeze(2)

        # (Pdb) heatmap.size() -- torch.Size([1, 11, 1, 64, 64])

        return heatmap

    def create_sparse_motions(self, source_image, transform_region_params: TransformParams, 
        source_region_params: RegionParams):
        bs, _, h, w = source_image.shape

        identity_grid = make_coordinate_grid(source_image).to(source_region_params.shift.device)

        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - transform_region_params.shift.view(bs, self.num_regions, 1, 1, 2)

        # 'affine' in transform_region_params -- True
        # if 'affine' in transform_region_params:
        affine = torch.matmul(source_region_params.affine, torch.inverse(transform_region_params.affine))

        # self.revert_axis_swap == True
        # if self.revert_axis_swap:
        #     affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
        affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
        affine = affine.unsqueeze(-3).unsqueeze(-3)
        affine = affine.repeat(1, 1, h, w, 1, 1)
        coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
        coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params.shift.view(bs, self.num_regions, 1, 1, 2)

        # adding background feature
        bg_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        sparse_motions = torch.cat([bg_grid, driving_to_source], dim=1)
        # (Pdb) driving_to_source.size() -- torch.Size([1, 10, 64, 64, 2])
        # pp sparse_motions.size() -- torch.Size([1, 11, 64, 64, 2])

        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_regions + 1, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * (self.num_regions + 1), -1, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_regions + 1), h, w, -1))

        # (Pdb) source_repeat.size() -- torch.Size([11, 3, 64, 64])
        # (Pdb) sparse_motions.size() -- torch.Size([11, 64, 64, 2])
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))

        # (Pdb) sparse_deformed.size() -- torch.Size([1, 11, 3, 64, 64])

        return sparse_deformed

    def forward(self, source_image, transform_region_params: TransformParams, 
        source_region_params: RegionParams) -> MotionParams:
        # self.scale_factor == 0.25
        # if self.scale_factor != 1:
        #     source_image = self.down(source_image)
        source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        # out_dict: Dict[str, torch.Tensor] = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, transform_region_params,
                                                                     source_region_params)
        sparse_motion = self.create_sparse_motions(source_image, transform_region_params, source_region_params)
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

        return MotionParams(optical_flow = deformation, occlusion_map=occlusion_map)


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
        pixelwise_flow_predictor_params = {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25, 'use_deformed_source': True, 'use_covar_heatmap': True, 'estimate_occlusion_map': True}
        skips = True
        revert_axis_swap = True

        self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_channels=num_channels,
                                                                   revert_axis_swap=revert_axis_swap,
                                                                   **pixelwise_flow_predictor_params)

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
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
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
        optical_flow = F.interpolate(optical_flow, size=inp.shape[2:], mode='bilinear', align_corners=False)
        optical_flow = optical_flow.permute(0, 2, 3, 1)

        return F.grid_sample(inp, optical_flow, align_corners=False)


    def apply_optical_with_prev(self, input_previous, input_skip, motion_params: MotionParams):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params.occlusion_map
        deformation = motion_params.optical_flow
        input_skip = self.deform_input(input_skip, deformation)
        # Remove "if" for trace model
        # if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
        #     occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
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
        occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        return input_skip * occlusion_map

    def forward(self, source_image, source_region_params: RegionParams, transform_region_params: TransformParams):
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

        motion_params = self.pixelwise_flow_predictor(source_image=source_image,
                                                      transform_region_params=transform_region_params,
                                                      source_region_params=source_region_params)
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
