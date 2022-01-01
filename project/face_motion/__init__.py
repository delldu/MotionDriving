"""Face Motion Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os

import redos
import todos
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from . import motion


def get_model():
    """Create model."""
    checkpoint = os.path.dirname(__file__) + "/models/image_motion.pth"

    model = motion.MotionDriving()
    model.load_weights(checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    return model, device


def model_forward(model, device, face_tensor, driving_tensor):
    face_tensor = face_tensor.to(device)
    driving_tensor = driving_tensor.to(device)

    with torch.no_grad():
        output_tensor = model(face_tensor, driving_tensor)
    return output_tensor


def video_service(input_file, output_file, targ):
    face_file = redos.taskarg_search(targ, "face_file")

    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # load face image
    face_image = Image.open(face_file).convert("RGB").resize((256, 256))
    face_tensor = transforms.ToTensor()(face_image).unsqueeze(0)

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"{input_file} driving {face_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        driving_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        driving_tensor = driving_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, face_tensor, driving_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def client(name, input_file, face_file, output_file):
    cmd = redos.video.Command()
    context = cmd.face(input_file, face_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_face", video_service, HOST, port)

def video_predict(input_file, face_file, output_file):
    targ = redos.taskarg_parse(
        f"video_face(input_file={input_file},face_file={face_file},output_file={output_file})"
    )
    video_service(input_file, output_file, targ)
