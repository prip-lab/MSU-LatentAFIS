# model.py

import math
import models
import losses
import evaluate
from torch import nn
import pdb

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.args = args
        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.model_type = args.model_type
        self.model_options = args.model_options
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options
        self.evaluation_type = args.evaluation_type
        self.evaluation_options = args.evaluation_options

    def setup(self, checkpoints):
        model = {}
        keys = [str(x['out_dims']) for x in self.model_options]
        for i,key in enumerate(keys):
            model[key] = getattr(models, self.model_type)(**self.model_options[i])
        criterion = getattr(losses, self.loss_type)(**self.loss_options)

        if self.cuda:
            for i,key in enumerate(keys):
                model[key] = model[key].cuda()
            criterion = criterion.cuda()

        if checkpoints.latest('resume') is None:
            pass
        else:
            model = checkpoints.load(model, checkpoints.latest('resume'))

        evaluation = None
        return model, criterion, evaluation
