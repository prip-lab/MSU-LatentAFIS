# classification.py

from torch import nn

__all__ = ['Classification']


# REVIEW: does this have to inherit nn.Module?
class Classification(nn.Module):
    def __init__(self, if_cuda=False):
        super(Classification, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss
