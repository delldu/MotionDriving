"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d, Encoder

from typing import Dict, List

import pdb
# Only for typing annotations
Tensor = torch.Tensor

def svd(covar):
    u, s, v = torch.svd(covar.cpu())
    s = s.to(covar.device)
    u = u.to(covar.device)
    v = v.to(covar.device)
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

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.regions = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_regions, 
                                kernel_size=(7, 7),
                                padding=3)

        # FOMM-like regression based representation
        if estimate_affine and not pca_based:
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4, kernel_size=(7, 7), padding=3)
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

    def region2affine(self, region):
        # (Pdb) region.shape -- torch.Size([1, 10, 96, 96])
        shape = region.shape
        region = region.unsqueeze(-1)
        # region.size() -- torch.Size([1, 10, 96, 96, 1]) ?

        # region.type() -- 'torch.cuda.FloatTensor'
        grid = make_coordinate_grid(int(shape[2]), int(shape[3])).unsqueeze_(0).unsqueeze_(0).to(region.device)
        mean = (region * grid).sum(dim=(2, 3))

        region_params = {'shift': mean}

        # self.pca_based == True
        # if self.pca_based:
        mean_sub = grid - mean.unsqueeze(-2).unsqueeze(-2)
        covar = torch.matmul(mean_sub.unsqueeze(-1), mean_sub.unsqueeze(-2))
        covar = covar * region.unsqueeze(-1)
        covar = covar.sum(dim=(2, 3))
        region_params['covar'] = covar

        # (Pdb) region_params.keys() -- dict_keys(['shift', 'covar'])
        # (Pdb) region_params['shift'].size() -- torch.Size([1, 10, 2])
        # (Pdb) region_params['covar'].size() -- torch.Size([1, 10, 2, 2])

        return region_params

    def forward(self, x) -> Dict[str, Tensor]:
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

        region_params = self.region2affine(region)
        region_params['heatmap'] = region

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

        covar = region_params['covar']
        shape = covar.shape
        covar = covar.view(-1, 2, 2)
        u, s, v = svd(covar)
        d = torch.diag_embed(s ** 0.5)
        sqrt = torch.matmul(u, d)
        sqrt = sqrt.view(shape)
        region_params['affine'] = sqrt
        region_params['u'] = u
        region_params['d'] = d

        return region_params
