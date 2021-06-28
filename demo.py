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

class MotionDriving(nn.Module):
    def __init__(self):
        super(MotionDriving, self).__init__()
        self.generator = Generator()
        self.region_predictor = RegionPredictor()
        self.avd_network = AVDNetwork()

        self.source = None
        self.source_params = None

        self.load_weights("checkpoints/vox256.pth")

    def load_weights(self, checkpoint):
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.generator.load_state_dict(state['generator'])
        self.region_predictor.load_state_dict(state['region_predictor'])
        self.avd_network.load_state_dict(state['avd_network'])

        self.generator.eval()
        self.region_predictor.eval()
        self.avd_network.eval()

    def forward(self, image, is_driving):
        if not is_driving:
            # image is source
            self.source = image
            self.source_params = self.region_predictor(image)
            return image

        # now image is driving frame
        driving_params = self.region_predictor(image)
        transform_params = self.avd_network(self.source_params, driving_params)
        output = self.generator(self.source, self.source_params, transform_params)

        return output

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

# vox256.pth-- dict_keys(['generator', 'bg_predictor', 'region_predictor', 'optimizer_reconstruction',
# 'avd_network', 'optimizer_avd', 'epoch_avd'])

# ted384.pth -- dict_keys(['epoch_reconstruction', 
#'generator', 'bg_predictor', 'region_predictor', 'avd_network'])

# ==> generator, bg_predictor, region_predictor, avd_network

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator()

    # print("Building generator ...")
    # script_model = torch.jit.script(generator)
    # script_model.save("output/motion_generator.pt")
    # print(script_model.code)
    # print("Building OK")

    # pdb.set_trace()

    if not cpu:
        generator.cuda()

    region_predictor = RegionPredictor()

    # print("Building region predictor ...")
    # script_model = torch.jit.script(region_predictor)
    # script_model.save("output/motion_predictor.pt")
    # print(script_model.code)
    # print("Building OK")

    if not cpu:
        region_predictor.cuda()

    avd_network = AVDNetwork()

    # print("Building avd_network ...")
    # script_model = torch.jit.script(avd_network)
    # script_model.save("output/motion_avdnetwork.pt")
    # print(script_model.code)
    # print("Building OK")
    # pdb.set_trace()
    
    if not cpu:
        avd_network.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    # if not cpu:
    #     generator = DataParallelWithCallback(generator)
    #     region_predictor = DataParallelWithCallback(region_predictor)
    #     avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()

    return generator, region_predictor, avd_network


def make_animation(source_image, driving_video, generator, region_predictor, avd_network,
                   animation_mode='standard', cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

        source_params = region_predictor(source)
        # pp source.shape -- torch.Size([1, 3, 384, 384])
        # source_params.keys() -- dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])
        # (Pdb) source_params['shift'].size() -- torch.Size([1, 10, 2])
        # (Pdb) source_params['covar'].size() -- torch.Size([1, 10, 2, 2])
        # (Pdb) source_params['heatmap'].size() -- torch.Size([1, 10, 96, 96])
        # (Pdb) source_params['affine'].size() -- torch.Size([1, 10, 2, 2])
        # (Pdb) source_params['u'].size() -- torch.Size([10, 2, 2])
        # (Pdb) source_params['d'].size() -- torch.Size([10, 2, 2])

        # driving_region_params_initial = region_predictor(driving[:, :, 0])
        # driving[:, :, 0].size() -- torch.Size([1, 3, 384, 384])
        # driving_region_params_initial.keys() -- dict_keys(['shift', 'covar', 'heatmap', 'affine', 'u', 'd'])

        # (Pdb) driving.shape -- torch.Size([1, 3, 265, 384, 384])
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_params = region_predictor(driving_frame)
            # only for avd
            transform_params = avd_network(source_params, driving_params)

            out = generator(source, source_params, transform_params)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    # type(predictions), predictions[0].shape-- (<class 'list'>, (384, 384, 3))

    return predictions


def do_predict(model, source_image, driving_video):
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

    source = source.cuda()
    driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)

    with torch.no_grad():
        model(source, False)

    # (Pdb) driving.shape -- torch.Size([1, 3, 265, 384, 384])
    for frame_idx in tqdm(range(driving.shape[2])):
        driving_frame = driving[:, :, frame_idx]
        driving_frame = driving_frame.cuda()

        with torch.no_grad():
            out = model(driving_frame, True)

        predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

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
    
    # type(source_image), source_image.shape -- (<class 'numpy.ndarray'>, (384, 384, 3))
    # (Pdb) type(driving_video), len(driving_video), driving_video[0].shape
    # (<class 'list'>, 265, (384, 384, 3))

    # generator, region_predictor, avd_network = load_checkpoints(config_path=opt.config,
    #                                                             checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    # # torch.save(generator.state_dict(), "output/motion_driving_generator.pth")
    # # torch.save(region_predictor.state_dict(), "output/motion_driving_region_predictor.pth")
    # # torch.save(avd_network.state_dict(), "output/motion_driving_avd_network.pth")

    # predictions = make_animation(source_image, driving_video, generator, region_predictor, avd_network,
    #                              animation_mode=opt.mode, cpu=opt.cpu)



    model = MotionDriving()
    model = model.eval()
    model = model.cuda()
    
    predictions = do_predict(model, source_image, driving_video)

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
