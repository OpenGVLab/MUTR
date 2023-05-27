# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
from torch import nn
from typing import Dict, List
import torch.nn.functional as F

from util.misc import NestedTensor
from torch import Tensor
import math

from .position_encoding import build_position_encoding

from timm.models import convnext
from einops import rearrange

def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', ''): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        if 'visual.head.proj.weight' in state_dict:
            out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        elif 'visual.head.mlp.fc1.weight' in state_dict:
            out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
            out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
            out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
            out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        return out_dict

    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        if 'grn' in k:
            k = k.replace('grn.beta', 'mlp.grn.bias')
            k = k.replace('grn.gamma', 'mlp.grn.weight')
            v = v.reshape(v.shape[-1])
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


class ConvNeXt(convnext.ConvNeXt):
    def __init__(self, out_indices,  **kwargs):
        super(ConvNeXt, self).__init__(**kwargs)
        self.out_indices = out_indices
        del self.norm_pre
        del self.head

    def forward_features(self, x):
        x = self.stem(x)
        outputs = {}
        for stage_idx in range(len(self.stages)):
            x = self.stages[stage_idx](x)
            if stage_idx in self.out_indices:
                outputs[stage_idx] = x
        return outputs


def convnext_large(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location='cpu')
        ckpt = checkpoint_filter_fn(ckpt, model)
        load_logs = model.load_state_dict(ckpt, strict=False)
        print(load_logs)
    return model


def convnext_xxlarge(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 4, 30, 3], dims=[384, 768, 1536, 3072], norm_eps=kwargs.pop('norm_eps', 1e-5),
                     **kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location='cpu')
        ckpt = checkpoint_filter_fn(ckpt, model)
        load_logs = model.load_state_dict(ckpt, strict=False)
        print(load_logs)
    return model


class ConvnextBackbone(nn.Module):
    def __init__(
        self, backbone: str, train_backbone: bool, return_interm_layers: bool, args
    ):
        super().__init__()
        if args.num_feature_levels == 4:
            out_indices = (0, 1, 2, 3)
        else:
            out_indices = (1, 2, 3)

        if backbone == 'convnext_large':
            backbone = convnext_large(args.backbone_pretrained, out_indices=out_indices,
                                      drop_path_rate=0.5)
            embed_dim = 192
        elif backbone == 'convnext_xxlarge':
            backbone = convnext_xxlarge(args.backbone_pretrained, out_indices=out_indices,
                                        drop_path_rate=0.5)
            embed_dim = 384
        else:
            raise NotImplementedError

        self.train_backbone = train_backbone
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)

        if return_interm_layers:

            if args.num_feature_levels == 4:
                self.strides = [4, 8, 16, 32]
                self.num_channels = [
                    embed_dim,
                    embed_dim * 2,
                    embed_dim * 4,
                    embed_dim * 8,
                ]
            else:
                self.strides = [8, 16, 32]
                self.num_channels = [
                    embed_dim * 2,
                    embed_dim * 4,
                    embed_dim * 8,
                ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]

        self.norm_layers = nn.ModuleList([nn.LayerNorm(ndim) for ndim in self.num_channels])

        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        
        if self.train_backbone:
            xs = self.body.forward_features(tensor_list.tensors)
        else:
            with torch.no_grad():
                xs = self.body.forward_features(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for layer_idx, (name, x) in enumerate(xs.items()):
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            b, c, h, w = x.shape
            x = self.norm_layers[layer_idx](x.view(b, c, -1).transpose(1, 2))
            x = x.transpose(1, 2).view(b, c, h, w)
            out[name] = NestedTensor(x, mask)
        return out



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor, no_norm=None):

        tensor_list.tensors = rearrange(tensor_list.tensors, 'b t c h w -> (b t) c h w')
        tensor_list.mask = rearrange(tensor_list.mask, 'b t h w -> (b t) h w')

        if no_norm is None:
            xs = self[0](tensor_list)
        else:
            xs = self[0](tensor_list, no_norm)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_convnext_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0

    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = ConvnextBackbone(args.backbone, train_backbone, return_interm_layers, args)
    model = Joiner(backbone, position_embedding)
    return model
