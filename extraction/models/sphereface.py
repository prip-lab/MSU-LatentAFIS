# sphereface.py

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import pdb

__all__ = ['SphereFace', 'sphereface4', 'sphereface10', 'sphereface20',
           'sphereface36', 'sphereface64', 'sphere20a']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def myphi(x, m):
    x = x * m
    output = 1 - x**2 / math.factorial(2) + x**4 / math.factorial(4) - \
        x**6 / math.factorial(6) + x**8 / math.factorial(8) - \
        x**9 / math.factorial(9)
    return output


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]

    def forward(self, input):
        # size=(B,F)    F is feature len
        x = input

        # size=(F,Classnum) F=in_features Classnum=out_features
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class BasicUnit(nn.Module):
    def __init__(self, planes):
        super(BasicUnit, self).__init__()
        self.planes = planes
        self.main = nn.Sequential(
            conv3x3(self.planes, self.planes, stride=1),
            nn.PReLU(self.planes),
            conv3x3(self.planes, self.planes, stride=1),
            nn.PReLU(self.planes)
        )

    def forward(self, x):
        y = self.main(x)
        y += x
        return y


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, nlayers):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.nlayers = nlayers

        self.conv1 = conv3x3(inplanes, outplanes, stride=2)
        self.relu1 = nn.PReLU(outplanes)

        layers = []
        for i in range(nlayers):
            layers.append(BasicUnit(self.outplanes))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.main(x)
        return x


class SphereFace(nn.Module):
    def __init__(self, layers, nchannels=3, nfilters=64,
        ndim=512, nclasses=0, dropout_prob=0.0, features=False):
        super(SphereFace, self).__init__()
        self.nclasses = nclasses
        self.nfilters = nfilters
        self.nchannels = nchannels
        self.dropout_prob = dropout_prob
        self.features = features

        self.conv = nn.Conv2d(self.nchannels, self.nfilters, kernel_size=5,
                              stride=2, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(self.nfilters)
        self.relu = nn.PReLU(self.nfilters)

        self.layer1 = BasicBlock(1 * nfilters, 1 * nfilters, layers[0])
        self.layer2 = BasicBlock(1 * nfilters, 2 * nfilters, layers[1])
        self.layer3 = BasicBlock(2 * nfilters, 4 * nfilters, layers[2])
        self.layer4 = BasicBlock(4 * nfilters, 8 * nfilters, layers[3])

        self.fc = nn.Linear(8 * nfilters * 3 * 3, ndim)

        if self.nclasses > 0:
            self.fc2 = AngleLinear(ndim, nclasses)

    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        x = F.dropout(x, self.dropout_prob)
        x = self.fc(x)

        if self.nclasses > 0:
            if self.features is True:
                return [x]
            else:
                y = self.fc2(x)
                return [x, y]
        else:
            return [x]


def sphereface4(**kwargs):
    """Constructs a SphereFace-04 model."""
    model = SphereFace([0, 0, 0, 0], **kwargs)
    return model


def sphereface10(**kwargs):
    """Constructs a SphereFace-10 model."""
    model = SphereFace([0, 1, 2, 0], **kwargs)
    return model


def sphereface20(**kwargs):
    """Constructs a SphereFace-20 model."""
    model = SphereFace([1, 2, 4, 1], **kwargs)
    return model


def sphereface36(**kwargs):
    """Constructs a SphereFace-36 model."""
    model = SphereFace([2, 4, 8, 2], **kwargs)
    return model


def sphereface64(**kwargs):
    """Constructs a SphereFace-64 model."""
    model = SphereFace([3, 8, 16, 3], **kwargs)
    return model

class sphere20a(nn.Module):
    def __init__(self,classnum=10574,features=False):
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.features = features
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512)
        self.fc6 = AngleLinear(512,self.classnum)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)
        if self.features: return [x]

        y = self.fc6(x)
        return [x,y]
