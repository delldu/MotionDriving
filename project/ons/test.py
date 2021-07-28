"""Test ons."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 07月 17日
# ***
# ************************************************************************************/
#
import torch
import ons
import argparse
import os
import pdb  # For debug
import time

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


import math
import torch

# Our module!
import ons

class GetSubWindowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, pos):
        output = ons.GetSubWindow(input, pos)
        return output

    # @staticmethod
    # def backward(ctx, grad_h, grad_cell):
    #     outputs = lltm_cpp.backward(
    #         grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
    #     d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
    #     return d_input, d_weights, d_bias, d_old_h, d_old_cell


class GetSubWindow(torch.nn.Module):
    def __init__(self):
        super(GetSubWindow, self).__init__()

    def forward(self, input, pos):
        return GetSubWindowFunction.apply(input, pos)



# class TestModel(nn.Module):
#     def __init__(self):
#         super(TestModel, self).__init__()

#     def forward(self, input, x1, y1, x2, y2):
#     	return ons.GetSubWindow(input, x1, y1, x2, y2)


if __name__ == "__main__":
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="output folder"
    )

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
    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        model = GetSubWindow()
        model.eval()

        onnx_file_name = "/tmp/test.onnx"

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]

        torch.onnx.export(
            model,
            (torch.randn(10, 3, 256, 256), torch.Tensor([10, 20, 30, 40])),
            onnx_file_name,
            input_names=input_names,
            output_names=output_names,
            verbose=True,
            opset_version=11,
            keep_initializers_as_inputs=False,
            export_params=True,
        )

        # 3. Optimize model
        print("Checking model ...")
        onnx_model = onnx.load(onnx_file_name)
        onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/image_motion.onnx')"

    export_onnx()
