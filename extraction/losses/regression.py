# regression.py

from torch import nn

__all__ = ['Regression']


class Regression(nn.Module):

    def __init__(self):
        super(Regression, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, inputs, target):
        loss = self.loss.forward(inputs, target)

        return loss
