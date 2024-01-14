import time
from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import skimage.color
import skimage.io
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
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


def send_graph_to_device(data, device, tissue_class=None):
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    if not tissue_class is None:
        tissue_class = torch.Tensor(tissue_class).type(torch.LongTensor).to(device)
    return x, edge_index, tissue_class


def save_model(model, save_path):
    torch.save(model, save_path)
