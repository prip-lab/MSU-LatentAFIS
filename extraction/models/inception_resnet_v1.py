import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import math
from collections import OrderedDict
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Block35(nn.Module):

    def __init__(self, in_planes, planes, stride=1, scale=1.0, if_activate=True):
        super(Block35, self).__init__()

        self.scale = scale
        self.if_activate = if_activate

        self.layer1 = nn.Sequential()
        self.layer1.add_module('Branch_0',
            self._make_layers(in_planes, planes, [1]))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('Branch_1',
            self._make_layers(in_planes, planes, [1,3]))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('Branch_2',
            self._make_layers(in_planes, planes, [1,3,3]))

        self.layer = nn.Conv2d(planes*3, in_planes, kernel_size=1, stride=stride, bias=True)

    def _make_layers(self, in_planes, planes, kSizes):
        layers = []
        for i in range(len(kSizes)):
            layers.append(nn.Conv2d(in_planes, planes, kernel_size=kSizes[i], stride=1,
                bias=False, padding=math.floor(kSizes[i]/2)))
            in_planes = planes
            layers.append(nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):

        tower1 = self.layer1(x)
        tower2 = self.layer2(x)
        tower3 = self.layer3(x)
        mixed = torch.cat([tower1, tower2, tower3], 1)
        y = self.layer(mixed)
        z = x+y*self.scale

        if self.if_activate:
            z = F.relu(z)

        return z

class Block17(nn.Module):

    def __init__(self, in_planes, planes, stride=1, scale=1.0, if_activate=True):
        super(Block17, self).__init__()

        self.scale = scale
        self.if_activate = if_activate

        self.layer1 = nn.Sequential()
        self.layer1.add_module('Branch_0',
            self._make_layers(in_planes, planes, [1,1]))

        tmp_layer = nn.Sequential(
            self._make_layers(in_planes, planes, [1,1]),
            self._make_layers(planes, planes, [1,7]),
            self._make_layers(planes, planes, [7,1]))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('Branch_1',
            tmp_layer)

        self.layer = nn.Conv2d(planes*2, in_planes, kernel_size=1, stride=stride, bias=True)

    def _make_layers(self, in_planes, planes, kSize):
        mylayer = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=kSize, stride=1,
                bias=False, padding=(math.floor(kSize[0]/2), math.floor(kSize[1]/2))),
            nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995),
            nn.ReLU())
        return mylayer

    def forward(self, x):
        tower1 = self.layer1(x)
        tower2 = self.layer2(x)
        mixed = torch.cat([tower1, tower2], 1)
        y = self.layer(mixed)
        z = x+y*self.scale

        if self.if_activate:
            z = F.relu(z)

        return z

class Block8(nn.Module):

    def __init__(self, in_planes, planes, stride=1, scale=1.0, if_activate=True):
        super(Block8, self).__init__()

        self.scale = scale
        self.if_activate = if_activate

        self.layer1 = nn.Sequential()
        self.layer1.add_module('Branch_0',
            self._make_layers(in_planes, planes, [1,1]))

        tmp_layer = nn.Sequential(
            self._make_layers(in_planes, planes, [1,1]),
            self._make_layers(planes, planes, [1,3]),
            self._make_layers(planes, planes, [3,1]))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('Branch_1',
            tmp_layer)

        self.layer = nn.Conv2d(planes*2, in_planes, kernel_size=1, stride=stride, bias=True)

    def _make_layers(self, in_planes, planes, kSize):
        mylayer = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=kSize, stride=1,
                bias=False, padding=(math.floor(kSize[0]/2), math.floor(kSize[1]/2))),
            nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995),
            nn.ReLU())
        return mylayer

    def forward(self, x):
        tower1 = self.layer1(x)
        tower2 = self.layer2(x)
        mixed = torch.cat([tower1, tower2], 1)
        y = self.layer(mixed)
        z = x+y*self.scale

        if self.if_activate:
            z = F.relu(z)

        return z

class Reduction_a(nn.Module):
    def __init__(self, in_planes, planes):
        super(Reduction_a, self).__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module('Branch_0',
            self._make_convlayer(in_planes, planes[0], 3, 2, 0))

        tmp_layer = nn.Sequential(
            self._make_convlayer(in_planes, planes[1], 1, 1, 0),
            self._make_convlayer(planes[1], planes[2], 3, 1, 1),
            self._make_convlayer(planes[2], planes[3], 3, 2, 0))
        self.layer2 = nn.Sequential()
        self.layer2.add_module('Branch_1',
            tmp_layer)

        self.layer3 = nn.Sequential()
        self.layer3.add_module('Branch_2',
            nn.MaxPool2d(3, stride=2, ceil_mode=False))

    def _make_convlayer(self, in_planes, planes, kSize, stride, padding):
        mylayer = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=kSize, stride=stride,
            bias=False, padding=padding),
        nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995),
        nn.ReLU())
        return mylayer

    def forward(self, x):
        tower1 = self.layer1(x)
        tower2 = self.layer2(x)
        tower3 = self.layer3(x)
        y = torch.cat([tower1, tower2, tower3], 1)

        return y

class Reduction_b(nn.Module):
    def __init__(self, in_planes):
        super(Reduction_b, self).__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module('Branch_0',
            self._make_layers(in_planes, [256,384], [1,3], [1,2]))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('Branch_1',
            self._make_layers(in_planes, [256,256], [1,3], [1,2]))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('Branch_2',
            self._make_layers(in_planes, [256,256,256], [1,3,3], [1,1,2]))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('Branch_3',
            nn.MaxPool2d(3, stride=2, ceil_mode=False))

    def _make_convlayer(self, in_planes, planes, kSize, stride, padding):
        mylayer = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=kSize, stride=stride,
            bias=False, padding=padding),
        nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995),
        nn.ReLU())
        return mylayer

    def _make_layers(self, in_planes, planes, kSizes, strides):
        layers = []
        for i in range(len(kSizes)):
            if i == len(kSizes)-1:
                layers.append(self._make_convlayer(in_planes, planes[i], kSizes[i], strides[i], 0))
            else:
                layers.append(self._make_convlayer(in_planes, planes[i], kSizes[i], strides[i], math.floor(kSizes[i]/2)))
            in_planes = planes[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        tower1 = self.layer1(x)
        tower2 = self.layer2(x)
        tower3 = self.layer3(x)
        tower4 = self.layer4(x)
        y = torch.cat([tower1, tower2, tower3, tower4], 1)

        return y

class Inception_resnet_v1(nn.Module):
    def __init__(self, in_planes, blocks, nfeatures=128, dropout_prob=0.0):
        super(Inception_resnet_v1, self).__init__()

        self.dropout_prob=dropout_prob
        self.nfeatures=nfeatures

        layer1 = self._make_convlayer(in_planes, 32, 3, 2, 0)
        layer2 = self._make_convlayer(32, 32, 3, 1, 0)
        layer3 = self._make_convlayer(32, 64, 3, 1, 1)
        layer4 = nn.MaxPool2d(3, stride=2)
        layer5 = self._make_convlayer(64, 80, 1, 1, 0)
        layer6 = self._make_convlayer(80, 192, 3, 1, 0)
        layer7 = self._make_convlayer(192, 256, 3, 2, 0)

        # block35
        layer8 = []
        in_planes = 256
        for i in range(5):
            layer8.append(blocks[0](in_planes, 32, scale=0.17))
        layer8 = nn.Sequential(*layer8)

        # reduction-A
        in_planes = 256
        layer9 = blocks[1](in_planes, [384, 192, 192, 256])

        # block17
        layer10 = []
        in_planes = 896
        for i in range(10):
            layer10.append(blocks[2](in_planes, 128, scale=0.10))
        layer10 = nn.Sequential(*layer10)

        # reduction-B
        in_planes = 896
        layer11 = blocks[3](in_planes)
        
        # block8
        layer12 = []
        in_planes = 1792
        for i in range(5):
            layer12.append(blocks[4](in_planes, 192, scale=0.20))
        layer12 = nn.Sequential(*layer12)

        # Block8
        in_planes = 1792
        layer13 = blocks[4](in_planes, 192, if_activate=False)

        self.bottleneck = nn.Sequential(
            nn.Linear(in_planes, self.nfeatures, bias=False),
            nn.BatchNorm1d(self.nfeatures),)

        self.model = nn.Sequential(OrderedDict([
          ('Conv2d_1a_3x3', layer1),
          ('Conv2d_2a_3x3', layer2),
          ('Conv2d_2b_3x3', layer3),
          ('MaxPool_3a_3x3', layer4),
          ('Conv2d_3b_1x1', layer5),
          ('Conv2d_4a_3x3', layer6),
          ('Conv2d_4b_3x3', layer7),
          ('Block35', layer8),
          ('Mixed_6a', layer9),
          ('Block17', layer10),
          ('Mixed_7a', layer11),
          ('Block8_1', layer12),
          ('Block8_2', layer13),
          ('AvgPool_1a_8x8', nn.AvgPool2d(3)),
        ]))



    def _make_convlayer(self, in_planes, planes, kSize, stride, padding):
        mylayer = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=kSize, stride=stride,
            bias=False, padding=padding),
        nn.BatchNorm2d(planes, affine=True, eps=1e-03, momentum=0.995),
        nn.ReLU()
        )
        return mylayer

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, self.dropout_prob)
        x = self.bottleneck(x)

        return [x]

class Inception_resnet_v1_multigpu(nn.Module):
    def __init__(self, network, in_planes, blocks, nfeatures=128, dropout_prob=0.0, ngpu=1):
        super(Inception_resnet_v1_multigpu, self).__init__()
        self.main = network(in_planes, blocks, nfeatures, dropout_prob)
        self.ngpu = ngpu

    def forward(self, input):
        output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        return output


def incep_resnetV1(nchannels,nfeatures,drop_prob=0.0):
    return Inception_resnet_v1(nchannels, [Block35, Reduction_a, 
        Block17, Reduction_b, Block8], nfeatures, dropout_prob=0.0)
# def incep_resnetV1(nchannels, nfeatures, drop_prob=0.0, ngpu=1):
#     return Inception_resnet_v1_multigpu(Inception_resnet_v1, nchannels, [Block35, Reduction_a, 
#         Block17, Reduction_b, Block8], nfeatures, drop_prob, ngpu)

# model = incep_resnetV1(3, 128)
# state_dict = model.state_dict()
# params = torch.load('new_state_dict.pth')
# params_val = list(params.values())

# for i, key in enumerate(state_dict):
#     state_dict[key] = params_val[i]

# torch.save(state_dict, 'multigpu_state_dict.pth')