"""Model predict."""# coding=utf-8
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
import glob
import os
import pdb  # For debug

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import get_model, model_device, model_setenv

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', type=str, default="models/image_motion.pth", help="checkpint file")
    parser.add_argument("--source_image", default='images/feynman.jpeg', help="path to source image")
    parser.add_argument("--driving_video", default='videos/2/*.png', help="path to driving video")
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_setenv()
    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)
    model.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    source_image = Image.open(args.source_image).convert("RGB").resize((256, 256))
    source_tensor = totensor(source_image).unsqueeze(0).to(device)

    video_filenames = sorted(glob.glob(args.driving_video))
    progress_bar = tqdm(total = len(video_filenames))

    for index, filename in enumerate(video_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB").resize((256, 256))
        driving_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(source_tensor, driving_tensor).clamp(0, 1.0).squeeze()

        output_filename ="{}/{}".format(args.output, os.path.basename(filename))

        toimage(output_tensor.cpu()).save(output_filename)
