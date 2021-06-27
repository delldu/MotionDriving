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
from typing import Dict

import pdb

# Only for typing annotations
Tensor = torch.Tensor


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
            nn.Linear(1024, id_bottle_size)
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
            nn.Linear(1024, pose_bottle_size)
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
            nn.Linear(256, input_size)
        )

    def region_params_to_emb(self, x: Dict[str, Tensor]):
        mean = x["shift"]
        jac = x["affine"]
        # jac.size() -- torch.Size([1, 10, 2, 2])
        emb = torch.cat([mean, jac.view(jac.shape[0], jac.shape[1], -1)], dim=-1)
        emb = emb.view(emb.shape[0], -1)
        # emb.size() -- torch.Size([1, 60])
        return emb

    def emb_to_region_params(self, emb) -> Dict[str, Tensor]:
        emb = emb.view(emb.shape[0], self.num_regions, 6)
        mean = emb[:, :, :2]
        jac = emb[:, :, 2:].view(emb.shape[0], emb.shape[1], 2, 2)
        return {'shift': mean, 'affine': jac}

    def forward(self, x_id: Dict[str, Tensor], x_pose: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # (Pdb) pp x_id.keys()
        # dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
        # (Pdb) pp x_pose.keys()
        # dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])

        # self.revert_axis_swap -- True
        # if self.revert_axis_swap:
        #     affine = torch.matmul(x_id['affine'], torch.inverse(x_pose['affine']))
        #     sign = torch.sign(affine[:, :, 0:1, 0:1])
        #     x_id = {'affine': x_id['affine'] * sign, 'shift': x_id['shift']}
        affine = torch.matmul(x_id['affine'], torch.inverse(x_pose['affine']))
        sign = torch.sign(affine[:, :, 0:1, 0:1])
        x_id = {'affine': x_id['affine'] * sign, 'shift': x_id['shift']}

        pose_emb = self.pose_encoder(self.region_params_to_emb(x_pose))
        id_emb = self.id_encoder(self.region_params_to_emb(x_id))

        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

        rec = self.emb_to_region_params(rec)
        rec['covar'] = torch.matmul(rec['affine'], rec['affine'].permute(0, 1, 3, 2))

        return rec


if __name__ == '__main__':
    """Onnx tools ..."""
    import argparse
    import os
    import time
    import numpy as np
    import onnx
    import onnxruntime


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--build', help="build torch script", action='store_true')
    parser.add_argument('-e', '--export', help="export onnx model", action='store_true')
    parser.add_argument('-v', '--verify', help="verify onnx model", action='store_true')
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #

    dummy_input = torch.randn(1, 1, 1, 512)
    onnx_file_name = "{}/motion_driving_avd_network.onnx".format(args.output)

    def get_model():
        # num_regions = 10
        model = AVDNetwork(num_regions=10)

        source_state = torch.load("../output/motion_driving_avd_network.pth")
        target_state = {}
        for k, v in source_state.items():
            k = k.replace("module.", "")
            target_state[k] = v

        model.load_state_dict(target_state)

        return model


    def build_script():
        model = AVDNetwork(num_regions=10)

        script_model = torch.jit.script(model)
        print(script_model.code)
        print(script_model.graph)


    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_model()
        torch_model.eval()


        pdb.set_trace()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        # dynamic_axes = {'input': {0: "batch"},'output': {0: "batch"}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          export_params=True)

        # 3. Visual model
        # python -c "import netron; netron.start('output/model.onnx')"

    def verify_onnx():
        """Verify onnx model."""

        torch_model = get_model()
        torch_model.eval()

        onnxruntime_engine = onnx_load(onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_input)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(onnx_file_name))
    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    if args.build:
        build_script()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()

