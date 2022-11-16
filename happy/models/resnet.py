from collections import OrderedDict

import torchvision.models as models
import torch
import torch.nn as nn


def build_resnet(out_features=5, depth=50):
    if depth == 34:
        model = models.resnet34(pretrained=True)
        in_features = 512
        first_out = 384
        second_out = 192
    elif depth == 50:
        model = models.resnet50(pretrained=True)
        in_features = 2048
        first_out = 768
        second_out = 384
    elif depth == 101:
        model = models.resnet101(pretrained=True)
        in_features = 2048
        first_out = 768
        second_out = 384
    else:
        raise ValueError(f"No such depth {depth}")

    layers = OrderedDict()
    layers["linear_layer_1"] = torch.nn.Linear(
        in_features=in_features, out_features=first_out, bias=True
    )
    layers["linear_Re_lu_1"] = torch.nn.ReLU()
    layers["linear_dropout_1"] = torch.nn.Dropout(p=0.3)
    layers["linear_layer_2"] = torch.nn.Linear(
        in_features=first_out, out_features=second_out, bias=True
    )
    layers["linear_Re_lu_2"] = torch.nn.ReLU()
    layers["linear_dropout_2"] = torch.nn.Dropout(p=0.3)
    layers["embeddings_layer"] = torch.nn.Linear(
        in_features=second_out, out_features=64, bias=True
    )
    layers["output_layer"] = torch.nn.Linear(in_features=64, out_features=out_features)

    new_module = nn.Sequential(layers)
    model.fc = new_module

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

    return model
