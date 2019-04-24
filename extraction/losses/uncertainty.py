# uncertainty.py

import torch
from torch import nn

__all__ = ['Uncertainty']


class Uncertainty(nn.Module):

    def __init__(self, wsigma):
        super(Uncertainty, self).__init__()
        self.wsigma = wsigma

    def __call__(self, input, target):

        mu = input[0]
        cov = input[1]  # log (sigma^2)

        tmp1 = target - mu
        tmp2 = torch.mul(tmp1, tmp1)

        tmp4 = torch.mul(tmp2, torch.exp(-cov))
        tmp5 = cov

        # regularizing the norm of the covariance.
        tmp6 = self.wsigma * torch.norm(torch.exp(cov))
        loss = tmp4.mean() + tmp5.mean() + tmp6

        return loss
