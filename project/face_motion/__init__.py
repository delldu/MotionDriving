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


def model_forward(model, device, face_tensor, input_tensor):
    face_tensor = face_tensor.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output_tensor = model(face_tensor, input_tensor).clamp(0, 1.0)

    return output_tensor


def predict(input_file, face_file, output_file):
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

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, face_tensor, input_tensor)

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


def server(name, HOST="localhost", port=6379):
    # return redos.video.service(name, "video_face", predict, HOST, port)
    mkey = "video_face"
    print(f"Start {mkey} service ...")

    client = redos.Redos(name, HOST=HOST, port=port)
    targ = client.get_queue_task(mkey)
    if targ is None:
        return False

    qkey = targ["key"]
    if not redos.taskarg_check(targ):
        client.set_task_state(qkey, -100)
        return False

    client.set_task_state(qkey, 0)

    input_file = redos.taskarg_search(targ, "input_file")
    face_file = redos.taskarg_search(targ, "face_file")
    output_file = redos.taskarg_search(targ, "output_file")

    ret = predict(input_file, face_file, output_file)

    # update state
    client.set_task_state(qkey, 100 if ret else -100)

    return ret
