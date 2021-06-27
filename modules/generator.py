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
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from typing import Dict, List

import pdb
# Only for typing annotations
Tensor = torch.Tensor

class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """
    # __constants__ = ['up_blocks', 'down_blocks']

    def __init__(self, num_channels, num_regions, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, pixelwise_flow_predictor_params=None, skips=False, revert_axis_swap=True):
        super(Generator, self).__init__()

        # num_channels = 3
        # num_regions = 10
        # block_expansion = 64
        # max_features = 512
        # num_down_blocks = 2
        # num_bottleneck_blocks = 6
        # pixelwise_flow_predictor_params = {'block_expansion': 64, 'max_features': 1024, 'num_blocks': 5, 'scale_factor': 0.25, 'use_deformed_source': True, 'use_covar_heatmap': True, 'estimate_occlusion_map': True}
        # skips = True
        # revert_axis_swap = True

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
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            # [b, h, w, 2] ==> [b, 2, h, w]
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            # Flow smoothing ...
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear', align_corners=False)
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow, align_corners=False)


    def apply_optical_with_prev(self, input_previous, input_skip, motion_params: Dict[str, Tensor]):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params['occlusion_map']
        deformation = motion_params['optical_flow']
        input_skip = self.deform_input(input_skip, deformation)
        if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        # input_previous != None
        return input_skip * occlusion_map + input_previous * (1 - occlusion_map)

    def apply_optical_without_prev(self, input_skip, motion_params: Dict[str, Tensor]):
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        occlusion_map = motion_params['occlusion_map']
        deformation = motion_params['optical_flow']
        input_skip = self.deform_input(input_skip, deformation)
        if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        return input_skip * occlusion_map

    def forward(self, source_image, driving_region_params: Dict[str, Tensor], source_region_params: Dict[str, Tensor]):
        out = self.first(source_image)
        skips: List[Tensor] = [out]
        # for i in range(len(self.down_blocks)):
        #     out = self.down_blocks[i](out)
        #     skips.append(out)
        for mod in self.down_blocks:
            out = mod(out)
            skips.append(out)

        output_dict: Dict[str, Tensor] = {}

        motion_params = self.pixelwise_flow_predictor(source_image=source_image,
                                                      driving_region_params=driving_region_params,
                                                      source_region_params=source_region_params)
        output_dict["deformed"] = self.deform_input(source_image, motion_params['optical_flow'])
        # motion_params.keys() -- dict_keys(['optical_flow', 'occlusion_map'])
        # if 'occlusion_map' in motion_params:
        #     output_dict['occlusion_map'] = motion_params['occlusion_map']
        output_dict['occlusion_map'] = motion_params['occlusion_map']

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

        output_dict["prediction"] = out

        return output_dict
