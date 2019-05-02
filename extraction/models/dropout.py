# dropout.py

from torch.nn.modules import Module
from torch.autograd.function import InplaceFunction

__all__ = ['CustomDropout']


class Dropout(InplaceFunction):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if self.p > 0:
            self.noise = self._make_noise(input)
            self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
            if self.p == 1:
                self.noise.fill_(0)
            self.noise = self.noise.expand_as(input)
            output.mul_(self.noise)

        return output

    def backward(self, grad_output):
        if self.p > 0:
            return grad_output.mul(self.noise)
        else:
            return grad_output


def f_dropout(input, p):
    return Dropout(p)(input)


class CustomDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(CustomDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.dropout = Dropout(self.p, self.inplace)

    def forward(self, input):
        return f_dropout(input, self.p)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) \
            + inplace_str + ')'
