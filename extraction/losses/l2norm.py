# L2Norm.py

import torch
from torch import nn

__all__ = ['L2NormLoss', 'BatchHardPairL2NormLoss']

class L2NormLoss(nn.Module):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, x1, x2, y1, y2):
        dist_in = torch.norm(x1 - x2, dim=1, keepdim=True)
        dist_out = torch.norm(y1 - y2, dim=1, keepdim=True)
        loss = torch.norm(dist_in - dist_out) / x1.size(0)
        return loss

class BatchHardPairL2NormLoss(nn.Module):
    def __init__(self, dist_metric='cosine', threshold=0.0):
        super(BatchHardPairL2NormLoss, self).__init__()
        self.dist_metric = dist_metric
        self.threshold = threshold

    def forward(self, x1, x2, y1, y2):
        if self.dist_metric == 'cosine':
            dist_in = torch.sum(torch.mul(x1,x2) / (torch.norm(x1,dim=1,keepdim=True)
                *torch.norm(x2,dim=1,keepdim=True)), dim=1)
            dist_out = torch.sum(torch.mul(y1,y2) / (torch.norm(y1,dim=1,keepdim=True)
                *torch.norm(y2,dim=1,keepdim=True)), dim=1)
        elif self.dist_metric == 'Euclidean':
            dist_in = torch.norm(x1-x2, dim=1, keepdim=True)
            dist_out = torch.norm(y1-y2, dim=1, keepdim=True)
        else:
            raise(RuntimeError('Metric does not support!'))
        diff = abs(dist_in - dist_out) - self.threshold
        ind_hard = diff > 0.0
        diff = (dist_in - dist_out)[ind_hard]
        if diff.size(0) != 0:
            loss = torch.norm(diff) / diff.size(0)
        else:
            loss = 0.0
        return loss