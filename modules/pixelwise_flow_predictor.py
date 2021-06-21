"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, region2gaussian
from modules.util import to_homogeneous, from_homogeneous
from typing import Dict

import pdb

class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and driving_region_params
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

    def create_heatmap_representations(self, source_image, driving_region_params: Dict[str, torch.Tensor], source_region_params: Dict[str, torch.Tensor]):
        """
        Eq 6. in the paper H_k(z)
        """
        # (Pdb) source_image.shape -- torch.Size([1, 3, 64, 64])
        spatial_size = source_image.shape[2:]

        # use_covar_heatmap = True
        # covar = self.region_var if not self.use_covar_heatmap else driving_region_params['covar']
        covar = driving_region_params['covar']
        gaussian_driving = region2gaussian(driving_region_params['shift'], covar=covar, spatial_size=spatial_size)


        # use_covar_heatmap = True
        # covar = self.region_var if not self.use_covar_heatmap else source_region_params['covar']
        covar = source_region_params['covar']
        gaussian_source = region2gaussian(source_region_params['shift'], covar=covar, spatial_size=spatial_size)

        heatmap = gaussian_driving - gaussian_source
        # (Pdb) heatmap.size() -- torch.Size([1, 10, 64, 64])

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1])
        heatmap = torch.cat([zeros.type(heatmap.type()), heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)

        # (Pdb) heatmap.size() -- torch.Size([1, 11, 1, 64, 64])

        return heatmap

    def create_sparse_motions(self, source_image, driving_region_params: Dict[str, torch.Tensor], source_region_params: Dict[str, torch.Tensor]):
        bs, _, h, w = source_image.shape

        # identity_grid = make_coordinate_grid((h, w), type=source_region_params['shift'].type())
        identity_grid = make_coordinate_grid(h, w).to(source_region_params['shift'].device)

        identity_grid = identity_grid.view(1, 1, h, w, 2)
        coordinate_grid = identity_grid - driving_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)

        # 'affine' in driving_region_params -- True
        if 'affine' in driving_region_params:
            affine = torch.matmul(source_region_params['affine'], torch.inverse(driving_region_params['affine']))

            # self.revert_axis_swap == True
            # if self.revert_axis_swap:
            #     affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
            affine = affine * torch.sign(affine[:, :, 0:1, 0:1])
            affine = affine.unsqueeze(-3).unsqueeze(-3)
            affine = affine.repeat(1, 1, h, w, 1, 1)
            coordinate_grid = torch.matmul(affine, coordinate_grid.unsqueeze(-1))
            coordinate_grid = coordinate_grid.squeeze(-1)

        driving_to_source = coordinate_grid + source_region_params['shift'].view(bs, self.num_regions, 1, 1, 2)

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
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions + 1, -1, h, w))

        # (Pdb) sparse_deformed.size() -- torch.Size([1, 11, 3, 64, 64])

        return sparse_deformed

    def forward(self, source_image, driving_region_params: Dict[str, torch.Tensor], source_region_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # self.scale_factor == 0.25
        # if self.scale_factor != 1:
        #     source_image = self.down(source_image)
        source_image = self.down(source_image)

        bs, _, h, w = source_image.shape

        out_dict: Dict[str, torch.Tensor] = dict()
        heatmap_representation = self.create_heatmap_representations(source_image, driving_region_params,
                                                                     source_region_params)
        sparse_motion = self.create_sparse_motions(source_image, driving_region_params, source_region_params)
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
        mask = F.softmax(mask, dim=1)
        mask = mask.unsqueeze(2)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)

        out_dict['optical_flow'] = deformation

        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        out_dict['occlusion_map'] = occlusion_map

        return out_dict
