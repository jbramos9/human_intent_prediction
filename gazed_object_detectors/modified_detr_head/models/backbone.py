# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torchvision.models import ResNet50_Weights

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 typed: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        
        if typed == "head":
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=FrozenBatchNorm2d)
            
            gaze_weight_path = "data/pretrained_weights/gaze360_model.pth"
            state_dict = torch.load(gaze_weight_path)["state_dict"]
            backbone.fc1 = nn.Linear(1000, 256)
            weights = {".".join(name.split(".")[2:]): value for name, value in state_dict.items() if "base_model" in name}
            weights["fc.weight"] = weights.pop("fc1.weight")
            weights["fc.bias"] = weights.pop("fc1.bias")
            weights["fc1.weight"] = weights.pop("fc2.weight")
            weights["fc1.bias"] = weights.pop("fc2.bias")
            backbone.load_state_dict(weights, strict=True)

        else:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=ResNet50_Weights.IMAGENET1K_V2,
                norm_layer=FrozenBatchNorm2d)
            
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
        if typed == "scene":
            # Change the first layer, and clone the weight of the first channel to the fourth
            # weights of channel0 copied to channel3 in case the backbone is pretrained
            weight = self.body.conv1.weight.clone()

            self.body.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                self.body.conv1.weight[:, :3] = weight
                self.body.conv1.weight[:, 3] = weight.mean(1)


class Joiner(nn.Sequential):
    def __init__(self, scene_backbone, position_embedding, head_backbone=None):
        if head_backbone:
            super().__init__(scene_backbone, head_backbone, position_embedding)
        else:
            super().__init__(scene_backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, head_list: NestedTensor=None):
        xs = self[0](tensor_list)
        hs = self[1](head_list) if head_list is not None else None
        out1: List[NestedTensor] = []
        out2: List[NestedTensor] = []
        pos1 = []
        pos2 = []

        if hs is not None:
            for ((_, x), (_, h)) in zip(xs.items(), hs.items()):
                out1.append(x)
                out2.append(h)
                # position encoding
                pos1.append(self[-1](x).to(x.tensors.dtype))
                pos2.append(self[-1](h).to(x.tensors.dtype))
        else:
            for name, x in xs.items():
                out1.append(x)
                # position encoding
                pos1.append(self[-1](x).to(x.tensors.dtype))

        return out1, out2, pos1, pos2


def build_backbone(config):

    hidden_dim = config["model"]["transformer"]["d_model"]
    position_embedder = config["model"]["position_encoder"]

    lr = config["model"]["backbone"]["lr"]
    backbone_net = config["model"]["backbone"]["name"]
    masks = config["model"]["backbone"].get("masks", False)
    dilation = config["model"]["backbone"].get("dilation", False)

    position_embedding = build_position_encoding(position_embedder, hidden_dim)
    train_backbone = lr > 0
    return_interm_layers = masks
    scene_backbone = Backbone(backbone_net, "scene", train_backbone, return_interm_layers, dilation)
    head_backbone = Backbone("resnet18", "head", train_backbone, return_interm_layers, dilation)

    model = Joiner(scene_backbone, position_embedding, head_backbone)
    model.scene_num_channels = scene_backbone.num_channels
    model.head_num_channels = head_backbone.num_channels
    return model
