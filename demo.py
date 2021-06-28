"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from torch import nn

# from sync_batchnorm import DataParallelWithCallback

from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from modules.avd_network import AVDNetwork

import os
import pdb
import collections

# Only for typing annotations
Tensor = torch.Tensor
RegionParams = collections.namedtuple('RegionParams', ['shift', 'covar', 'affine'])

class MotionDriving(nn.Module):
    def __init__(self):
        super(MotionDriving, self).__init__()
        self.generator = Generator()
        self.region_predictor = RegionPredictor()
        self.avd_network = AVDNetwork()

    def load_weights(self, checkpoint: str):
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.generator.load_state_dict(state['generator'])
        self.region_predictor.load_state_dict(state['region_predictor'])
        self.avd_network.load_state_dict(state['avd_network'])

        self.generator.eval()
        self.region_predictor.eval()
        self.avd_network.eval()

    def forward(self, source_image, driving_image):
        source_params: RegionParams = self.region_predictor(source_image)

        # pp source_image.shape -- torch.Size([1, 3, 384, 384])
        # source_params.keys() -- dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
        # (Pdb) source_params['shift'].size() -- torch.Size([1, 10, 2])
        # (Pdb) source_params['covar'].size() -- torch.Size([1, 10, 2, 2])
        # (Pdb) source_params['heatmap'].size() -- torch.Size([1, 10, 96, 96])
        # (Pdb) source_params['affine'].size() -- torch.Size([1, 10, 2, 2])
        # (Pdb) source_params['u'].size() -- torch.Size([10, 2, 2])
        # (Pdb) source_params['d'].size() -- torch.Size([10, 2, 2])

        # now image is driving frame
        driving_params: RegionParams = self.region_predictor(driving_image)
        transform_params: RegionParams = self.avd_network(source_params, driving_params)
        
        return self.generator(source_image, source_params, transform_params)

# vox256.pth-- dict_keys(['generator', 'bg_predictor', 'region_predictor', 'optimizer_reconstruction',
# 'avd_network', 'optimizer_avd', 'epoch_avd'])

# ted384.pth -- dict_keys(['epoch_reconstruction', 
#'generator', 'bg_predictor', 'region_predictor', 'avd_network'])

# ==> generator, bg_predictor, region_predictor, avd_network

def make_animation(model, source_image, driving_video):
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    source = source.cuda()
    driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

    total = len(range(driving.shape[2]))
    progress_bar = tqdm(total = total)
    # (Pdb) driving.shape -- torch.Size([1, 3, 265, 384, 384])
    progress_bar.set_description("Processing total %03d" % total)

    for frame_idx in range(driving.shape[2]):
        progress_bar.update(1)

        driving_frame = driving[:, :, frame_idx]
        driving_frame = driving_frame.cuda()

        with torch.no_grad():
            out = model(source, driving_frame)

        predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])

    # type(predictions), predictions[0].shape-- (<class 'list'>, (384, 384, 3))

    return predictions


def main(opt):
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    reader.close()
    driving_video = imageio.mimread(opt.driving_video, memtest=False)

    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    
    model = MotionDriving()
    model.load_weights(opt.checkpoint)
    model = model.eval()
    model = model.cuda()

    print("Building script model ...")
    torch.jit.script(model)
    traced_model = torch.jit.trace(model, 
        (torch.randn(1, 3, 256, 256).cuda(),  torch.randn(1, 3, 256, 256).cuda()))
    # pdb.set_trace()

    print("Building OK.")

    predictions = make_animation(model, source_image, driving_video)

    result_filename = opt.output + "/" + os.path.basename(opt.driving_video)
    imageio.mimsave(result_filename, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='ted384.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")

    parser.add_argument("--mode", default='avd', choices=['standard', 'relative', 'avd'],
                        help="Animation mode")
    parser.add_argument("--img_shape", default="256,256", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    main(parser.parse_args())
