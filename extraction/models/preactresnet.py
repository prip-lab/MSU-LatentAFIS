# preactresnet.py

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['PreActResNet', 'preactresnet18', 'preactresnet34',
           'preactresnet50', 'preactresnet101', 'preactresnet152']


def conv3x3(inplanes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.PReLU(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += shortcut

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.PReLU(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)

        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))
        out += shortcut

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, layers, nchannels=3, nfilters=64,
                 ndim=512, nclasses=0, dropout_prob=0.0, features=False):
        super(PreActResNet, self).__init__()
        self.ndim = ndim
        self.nclasses = nclasses
        self.nchannels = nchannels
        self.nfilters = nfilters
        self.inplanes = nfilters
        self.dropout_prob = dropout_prob
        self.features = features

        self.conv1 = nn.Conv2d(self.nchannels, self.nfilters, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nfilters)
        self.relu1 = nn.PReLU(self.nfilters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 1 * self.nfilters, layers[0],
                                       stride=1)
        self.layer2 = self._make_layer(block, 2 * self.nfilters, layers[1],
                                       stride=2)
        self.layer3 = self._make_layer(block, 4 * self.nfilters, layers[2],
                                       stride=2)
        self.layer4 = self._make_layer(block, 8 * self.nfilters, layers[3],
                                       stride=2)
        self.fc = nn.Linear(8 * self.nfilters * block.expansion, self.ndim)

        if self.nclasses > 0:
            self.fc2 = nn.Linear(self.ndim, self.nclasses)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu1(self.bn1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
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


def preactresnet18(**kwargs):
    """Constructs a PreActResNet-18 model.
    """
    model = PreActResNet(PreActBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def preactresnet34(**kwargs):
    """Constructs a PreActResNet-34 model.
    """
    model = PreActResNet(PreActBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def preactresnet50(**kwargs):
    """Constructs a PreActResNet-50 model.
    """
    model = PreActResNet(PreActBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def preactresnet101(**kwargs):
    """Constructs a PreActResNet-101 model.
    """
    model = PreActResNet(PreActBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def preactresnet152(**kwargs):
    """Constructs a PreActResNet-152 model.
    """
    model = PreActResNet(PreActBottleneck, [3, 8, 36, 3], **kwargs)
    return model
