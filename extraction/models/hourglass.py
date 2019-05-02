# hourglass.py

import torch.nn as nn

__all__ = ['HourGlass']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Lin(nn.Module):
    def __init__(self, nin, nout):
        super(Lin, self).__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0)
        self.bnorm = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, inp):
        inp = self.conv(inp)
        inp = self.bnorm(inp)
        inp = self.relu(inp)
        return inp


class Residual(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Residual, self).__init__()

        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes

        self.conv1 = conv3x3(self.inplanes, self.planes, self.stride)
        self.bn1 = nn.BatchNorm2d(self.planes)
        self.relu1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = conv3x3(self.planes, self.planes)
        self.bn2 = nn.BatchNorm2d(self.planes)
        self.conv3 = nn.Conv2d(self.planes, self.inplanes, kernel_size=1,
                               stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.inplanes != self.planes:
            out = self.conv3(out)

        out += residual
        out = self.relu2(out)

        return out


class HourGlassUnit(nn.Module):
    def __init__(self, depth, nfilters, nmodules=1):
        super(HourGlassUnit, self).__init__()

        self.depth = depth
        self.nfilters = nfilters
        self.nmodules = nmodules
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.layer1 = self.make_layers_()
        self.layer2 = self.make_layers_()
        self.layer3 = self.make_layers_()
        self.layer4 = self.make_layers_()
        self.conv = nn.ConvTranspose2d(self.nfilters, self.nfilters,
                                       4, 2, 1, bias=False)

        if self.depth > 1:
            self.hg = HourGlassUnit(self.depth - 1, self.nfilters)

    def make_layers_(self):
        modules = []
        for i in range(self.nmodules):
            modules.append(Residual(self.nfilters, self.nfilters))
        return nn.Sequential(*modules)

    def forward(self, inp):
        # skip connection branch
        up1 = self.layer1(inp)

        # hourglass branch
        low1 = self.pool1(inp)
        low1 = self.layer2(low1)

        if self.depth > 1:
            low2 = self.hg(low1)
        else:
            low2 = self.layer3(low1)

        low3 = self.layer4(low2)
        up2 = self.conv(low3)

        up1 += up2
        return up1


class HourGlassStack(nn.Module):
    def __init__(self, nfilters, noutputs, nmodules=1):
        super(HourGlassStack, self).__init__()

        self.nfilters = nfilters
        self.noutputs = noutputs
        self.nmodules = nmodules

        self.layers = self.make_layers_()
        self.output = nn.Conv2d(self.nfilters, self.noutputs,
                                kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(self.nfilters, self.nfilters,
                               kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.noutputs, self.nfilters,
                               kernel_size=1, stride=1, padding=0)

    def make_layers_(self):
        modules = []
        modules.append(HourGlassUnit(4, self.nfilters))
        for i in range(self.nmodules):
            modules.append(Residual(self.nfilters, self.nfilters))
        modules.append(Lin(self.nfilters, self.nfilters))
        return nn.Sequential(*modules)

    def forward(self, inp):
        ll = self.layers(inp)
        tmpout = self.output(ll)

        ll_ = self.conv1(ll)
        tmpout_ = self.conv2(tmpout)
        inter = ll_ + tmpout_

        out = []
        out.append(inter)
        out.append(tmpout)
        return out


class HourGlassLast(nn.Module):
    def __init__(self, nfilters, noutputs, nmodules=1):
        super(HourGlassLast, self).__init__()

        self.nfilters = nfilters
        self.noutputs = noutputs
        self.nmodules = nmodules
        self.layers = self.make_layers_()

    def make_layers_(self):
        modules = []
        modules.append(HourGlassUnit(4, self.nfilters))
        for i in range(self.nmodules):
            modules.append(Residual(self.nfilters, self.nfilters))
        modules.append(Lin(self.nfilters, self.nfilters))
        modules.append(nn.Conv2d(self.nfilters, self.noutputs,
                                 kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*modules)

    def forward(self, inp):
        inp = self.layers(inp)
        return inp


class HourGlass(nn.Module):

    """ This is the appearance model based on the hourglass model. """

    def __init__(self, nchannels, nfilters, nstack, noutputs, nmodules=1):
        super(HourGlass, self).__init__()

        self.nchannels = nchannels
        self.nstack = nstack
        self.nfilters = nfilters
        self.noutputs = noutputs
        self.nmodules = nmodules

        self.conv1 = nn.Conv2d(self.nchannels, self.nfilters,
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.nfilters)
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.r1 = Residual(self.nfilters, self.nfilters)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.r2 = Residual(self.nfilters, self.nfilters)
        self.r3 = Residual(self.nfilters, self.noutputs)
        self.stack_hg = self.make_layers_()
        self.last_hg = HourGlassLast(self.nfilters, self.noutputs,
                                     self.nmodules)

        self.extracted_layers = ['HourGlassStack']

    def make_layers_(self):
        modules = []
        for i in range(self.nstack - 1):
            modules.append(HourGlassStack(self.nfilters, self.noutputs,
                                          self.nmodules))
        return nn.Sequential(*modules)

    def forward(self, inp):
        tmp = self.conv1(inp)
        tmp = self.bn1(tmp)
        tmp = self.relu(tmp)
        tmp = self.r1(tmp)
        tmp = self.pool(tmp)
        tmp = self.r2(tmp)
        tmp = self.r3(tmp)
        inp = []
        inp.append(tmp)
        out = []
        for name, module in self.stack_hg._modules.items():
            inp = module(inp[0])
            out.append(inp[1])

        out.append(self.last_hg(inp[0]))
        return out
