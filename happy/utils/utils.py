import time
from collections import OrderedDict
from pathlib import Path

import GPUtil
import numpy as np
import skimage.color
import skimage.io
import torch


def set_gpu_device():
    print(GPUtil.showUtilization())
    device_ids = GPUtil.getAvailable(
        order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
    )
    while not device_ids:
        print("No GPU avail.")
        time.sleep(10)
        device_ids = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=0.3, maxMemory=0.3
        )
    device_id = str(device_ids[0])
    print(f"Using GPU number {device_id}")
    return device_id


def get_device(get_cuda_device_num=False):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if get_cuda_device_num:
            return f"cuda:{set_gpu_device()}"
        else:
            return "cuda"
    else:
        return "cpu"


def get_project_dir(project_name):
    root_dir = Path(__file__).absolute().parent.parent.parent
    return root_dir / "projects" / project_name


def load_weights(state_dict, model):
    # Removes the module string from the keys if it's there.
    if "module.conv1.weight" in state_dict.keys():
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    return model


def process_image(img):
    if img.shape[2] == 2:
        img = skimage.color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = np.array(img)[:, :, 0:3]
    elif img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected number of channels {img.shape[2]}")
    # returns in the image as a uin8 (to be converted to float32 later)
    return img


def load_image(image_path):
    img = skimage.io.imread(image_path)
    return process_image(img)
