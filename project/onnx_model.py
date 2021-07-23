"""Onnx Model Tools."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 06月 17日 星期四 14:09:56 CST
# ***
# ************************************************************************************/
#

import argparse
import os
import pdb  # For debug
import time

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Import Model Method
# ***
# ************************************************************************************/
#
from model import get_model


def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx Model Engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


def onnx_forward(onnx_model, input):
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


if __name__ == "__main__":
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--export", help="export onnx model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify onnx model", action="store_true")
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

    dummy_source = torch.randn(1, 3, 256, 256)
    dummy_driving = torch.randn(1, 3, 256, 256)

    checkpoint = "models/image_motion.pth"
    onnx_file_name = "{}/image_motion.onnx".format(args.output)

    def build_script():
        print("Building script model ...")
        model = get_model(checkpoint)
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save("{}/image_motion.pt".format(args.output))

        model = get_model(checkpoint).generator
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save("{}/image_motion_generator.pt".format(args.output))

        model = get_model(checkpoint).region_predictor
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save("{}/image_motion_regin_predictor.pt".format(args.output))

        model = get_model(checkpoint).avd_network
        model.eval()
        script_model = torch.jit.script(model)
        script_model.save("{}/image_motion_avd_network.pt".format(args.output))

        print("Building OK.")

    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        model = get_model(checkpoint)
        # model = model.cuda()
        model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["source", "driving"]
        output_names = ["output"]

        torch.onnx.export(
            model,
            (dummy_source, dummy_driving),
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

    def verify_onnx():
        """Verify onnx model."""

        model = get_model(checkpoint)
        model.eval()

        onnxruntime_engine = onnx_load(onnx_file_name)

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        with torch.no_grad():
            torch_output = model(dummy_source, dummy_driving)

        onnxruntime_inputs = {
            onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_source),
            onnxruntime_engine.get_inputs()[1].name: to_numpy(dummy_driving),
        }
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print(
            "Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(
                onnx_file_name
            )
        )

    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    # build_script()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
