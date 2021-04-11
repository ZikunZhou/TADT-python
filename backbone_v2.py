import logging
import os
from collections import OrderedDict
from functools import partial

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv3x3(in_planes, out_planes, padding = 1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=padding,
        dilation=dilation)


def make_vgg_layer(inplanes, planes, num_blocks, dilation=1, with_bn=False,
                   ceil_mode=False, with_pool = False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    if with_pool:
        #ceil_mode – when True, will use ceil instead of floor to compute the output shape
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers

class VGG(nn.Module):
    '''
     Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    '''
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }
    def __init__(self,
                depth,
                with_bn = False,
                with_pools = (True,True,True,True,True),
                num_stages = 5,
                dilations = (1,1,1,1,1),
                frozen_stages = 5,
                bn_eval=False,
                bn_frozen=False,
                ceil_mode = False,
                out_indices = None,
                pretrain = "/your/model/path/imagenet-vgg-verydeep-16.mat"
    ):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for vgg'.format(depth))
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.inplanes = 3
        self.module_name = 'features'
        stage_blocks = self.arch_settings[16]
        self.stage_blocks = stage_blocks[:num_stages]
        self.out_indices = out_indices
        self.range_sub_modules = []
        self.out_indices_available = {}
        self.channels_of_outs = {}
        self.missing_layer = []
        vgg_layers = []
        start_idx = 0
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            with_pool = with_pools[i]
            planes = 64*(2**i) if i < 4 else 512
            vgg_layer = make_vgg_layer(
                inplanes = self.inplanes,
                planes = planes,
                num_blocks = num_blocks,
                dilation = dilation,
                with_bn = with_bn,
                ceil_mode = ceil_mode,
                with_pool = True
            )
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            if not with_pool:
                self.missing_layer.append(end_idx - 1)

            for j in range(num_blocks):
                if j < num_blocks-1:
                    out_indice = start_idx + j * (2+with_bn) + 1
                if j == num_blocks-1:
                    out_indice = start_idx + j * (2+with_bn) + 1 + with_pool
                self.out_indices_available['conv{}_{}'.format(i+1,j+1)] = out_indice
                self.channels_of_outs['conv{}_{}'.format(i+1,j+1)] = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx

        self.add_module(self.module_name,nn.Sequential(*vgg_layers))
        self.init_weights(pretrain)

    def forward(self, x, out_indices = None):

        outs = []
        vgg_layers = getattr(self,self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                if j not in self.missing_layer:
                    vgg_layer = vgg_layers[j]
                    x = vgg_layer(x)
                for key in self.out_indices:
                    if self.out_indices_available[key] == j:
                        outs.append(x)
        return outs
    def init_weights(self, pretrain = None):
        logger = logging.getLogger()
        load_params_from_mat(self, pretrain, strict = False, logger = logger)

    def train(self, mode=True):

        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False

def load_params_from_mat(model, net_path, strict = False, logger = None):
    #model_conv_layers = ['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'features.5.weight', 'features.5.bias', 'features.7.weight', 'features.7.bias', 'features.10.weight', 'features.10.bias', 'features.12.weight', 'features.12.bias', 'features.14.weight', 'features.14.bias', 'features.17.weight', 'features.17.bias', 'features.19.weight', 'features.19.bias', 'features.21.weight', 'features.21.bias', 'features.24.weight', 'features.24.bias', 'features.26.weight', 'features.26.bias', 'features.28.weight', 'features.28.bias']
    layers_dict = {}
    layers, conv_layers, state_dict = load_matconvnet(net_path)
    own_state = model.state_dict()
    own_layers = list(own_state.keys())

    for i in range(len(own_layers)):
        layers_dict[conv_layers[i]] = own_layers[i]

    unexpected_keys = []
    for name, param in state_dict.items():
        if name not in layers_dict.keys():
            unexpected_keys.append(name)
            continue
        try:
            own_state[layers_dict[name]].copy_(torch.from_numpy(param))

        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(layers_dict[name], own_state[name].size(),
                                       param.size()))
    missing_keys = set(own_state.keys()) - set(layers_dict.values())
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            pass
            logger.warn(err_msg)
        else:
            #pass
            print(err_msg)


def load_matconvnet(net_path):

    mat = scipy.io.loadmat(net_path)
    state_dict = {}

    net_dot_mat = mat.get('layers')

    layers = []
    conv_layers = []
    for i in range(len(net_dot_mat[0])):
        layers.append(net_dot_mat[0][i][0][0][0][0])
        if net_dot_mat[0][i][0][0][1][0] == 'conv':
            param = {}
            layer_name = net_dot_mat[0][i][0][0][0][0]
            conv_layers.append(layer_name + '-weight')
            conv_layers.append(layer_name + '-bias')
            weight = net_dot_mat[0][i][0][0][2][0][0]
            bias = net_dot_mat[0][i][0][0][2][0][1]

            state_dict[layer_name + '-weight'] = weight.transpose((3,2,0,1))
            state_dict[layer_name + '-bias'] = np.squeeze(bias)

    return layers, conv_layers, state_dict

def build_vgg16(config):
    vgg16 = VGG(depth = config.BACKBONE.VGG16.DEPTH,
                with_bn = config.BACKBONE.VGG16.WITH_BN,
                with_pools =config.BACKBONE.VGG16.WITH_POOLS,
                num_stages = config.BACKBONE.VGG16.NUM_STAGES,
                dilations = config.BACKBONE.VGG16.DILATIONS,
                frozen_stages = config.BACKBONE.VGG16.FROZEN_STAGE,
                bn_eval = config.BACKBONE.VGG16.BN_EVAL,
                bn_frozen = config.BACKBONE.VGG16.BN_FROZEN,
                ceil_mode = config.BACKBONE.VGG16.CEIL_MODEL,
                out_indices = config.BACKBONE.VGG16.OUT_INDICES,
                pretrain = config.BACKBONE.VGG16.PRETRAIN_MAT
            )
    return vgg16
