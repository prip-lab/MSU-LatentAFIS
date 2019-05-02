# net of comppression

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_dims, out_dims, shortcut=None, if_activate=True):
        super(BasicBlock, self).__init__()
        self.if_activate = if_activate
        self.layers = nn.Sequential(
            nn.Linear(in_dims, out_dims, bias=True),
            nn.BatchNorm1d(out_dims),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),
            nn.Linear(out_dims, out_dims, bias=True),
            nn.BatchNorm1d(out_dims),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        y += residual
        if self.if_activate:
            # y = F.relu(y)
            y = F.leaky_relu(y, 0.2, True)
        return y

class CompNet(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(CompNet, self).__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims

        self.layer1 = nn.Sequential(
            nn.Linear(in_dims, out_dims, bias=True),
            nn.BatchNorm1d(out_dims),
            nn.LeakyReLU(0.2, True),
            )
        self.layer2 = BasicBlock(out_dims, out_dims)
        self.layer3 = nn.Sequential(
            nn.Linear(out_dims, out_dims, bias=True),
            nn.BatchNorm1d(out_dims),
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x