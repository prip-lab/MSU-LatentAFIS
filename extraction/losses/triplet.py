# triplet.py

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['BatchHardTripletLoss', 'RandomBatchTripletLoss']


class BatchHardTripletLoss(nn.Module):
    def __init__(self, batch_size_class, batch_size_image,
                 margin=0.0, cuda=False):
        super(BatchHardTripletLoss, self).__init__()
        self.cuda = cuda
        self.margin = margin

        if self.margin == 0:
            self.activation = nn.Softplus()
        else:
            self.activation = None

        self.batch_size_class = batch_size_class
        self.batch_size_image = batch_size_image
        self.batch_size = self.batch_size_class * self.batch_size_image

        self.pos_mask = torch.zeros(self.batch_size, self.batch_size)
        for i in range(self.batch_size_class):
            start = i * self.batch_size_image
            stop = (i + 1) * self.batch_size_image
            self.pos_mask[start:stop, start:stop].fill_(1)
        self.pos_mask = Variable(self.pos_mask, requires_grad=False)
        self.neg_mask = torch.add(-self.pos_mask, 1)
        self.pos_dist_diag = torch.mul(self.pos_mask, 1e25)

        if self.cuda:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()
            if self.activation is not None:
                self.activation = self.activation.cuda()
            self.pos_dist_diag = self.pos_dist_diag.cuda()

    def forward(self, features):

        batch_size = features.size(0)
        batch_size_class = int(batch_size / self.batch_size_image)

        if batch_size_class != self.batch_size_class:
            self.pos_mask.data.resize_(batch_size, batch_size).fill_(0)
            self.batch_size_class = int(batch_size / self.batch_size_image)
            for i in range(self.batch_size_class):
                start = i * self.batch_size_image
                stop = (i + 1) * self.batch_size_image
                self.pos_mask[start:stop, start:stop].data.fill_(1)
            self.neg_mask.data = torch.add(-self.pos_mask.data, 1)
            self.pos_dist_diag = torch.mul(self.pos_mask, 1e25)

        feat_norm = torch.mul(
            features, features).sum(dim=1, keepdim=True).repeat(1, batch_size)
        r = torch.mm(features, features.transpose(1, 0))
        distance = torch.add(
            feat_norm, torch.add(feat_norm.transpose(1, 0), -torch.mul(r, 2)))
        distance = torch.clamp(distance, min=1e-12)
        avg_feat_norm = features.norm(dim=1).mean()

        dist_pos = torch.mul(distance, self.pos_mask)
        dist_neg = torch.mul(distance, self.neg_mask)
        avg_pos_dist = dist_pos[self.pos_mask == 1].sqrt().mean()
        avg_neg_dist = dist_neg[self.neg_mask == 1].sqrt().mean()
        dist_neg = torch.add(dist_neg, self.pos_dist_diag)

        score_pos, _ = dist_pos.max(dim=1)
        score_neg, _ = dist_neg.min(dim=1)

        score_pos = score_pos.sqrt()
        score_neg = score_neg.sqrt()

        if self.margin > 0:
            diff = self.margin + score_pos - score_neg
            avg_active = (diff > 0).sum() / batch_size
            loss = torch.clamp(diff, min=0.0)
        else:
            diff = 1e-6 + score_pos - score_neg
            avg_active = (diff > 0).sum() / batch_size
            loss = self.activation(score_pos - score_neg)

        return loss, avg_feat_norm, avg_active, avg_pos_dist, avg_neg_dist


class RandomBatchTripletLoss(nn.Module):
    def __init__(self, batch_size_class, batch_size_image,
                 margin=0.0, cuda=False):
        super(RandomBatchTripletLoss, self).__init__()
        self.cuda = cuda
        self.margin = margin

        if self.margin == 0:
            self.activation = nn.Softplus()
        else:
            self.activation = None

        self.batch_size_class = batch_size_class
        self.batch_size_image = batch_size_image
        self.batch_size = self.batch_size_class * self.batch_size_image

        self.pos_mask = torch.zeros(self.batch_size, self.batch_size)
        for i in range(self.batch_size_class):
            start = i * self.batch_size_image
            stop = (i + 1) * self.batch_size_image
            self.pos_mask[start:stop, start:stop].fill_(1)
        self.neg_mask = torch.add(-self.pos_mask, 1)
        self.pos_dist_diag = torch.mul(self.pos_mask, 1e25)
        self.pos_mask = torch.add(self.pos_mask,
                                  -torch.eye(self.batch_size, self.batch_size))
        self.pos_mask = Variable(self.pos_mask, requires_grad=False)
        self.neg_mask = Variable(self.neg_mask, requires_grad=False)
        self.pos_dist_diag = Variable(self.pos_dist_diag, requires_grad=False)

        if self.cuda:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()
            self.pos_dist_diag = self.pos_dist_diag.cuda()
            if self.activation is not None:
                self.activation = self.activation.cuda()

    def sample_from_dist(self, dist_pos, dist_neg):
        pos_ids = []
        neg_ids = []
        for i in range(dist_pos.size(0)):
            id1 = dist_pos[i, :].nonzero().squeeze()
            sample1 = torch.multinomial(
                torch.index_select(dist_pos[i], 0, id1), 1)
            pscore = torch.index_select(dist_pos[i], 0, id1[sample1])
            sample1 = id1[sample1][0]
            id2 = (dist_neg[i, :] < pscore + 1e-4).nonzero().squeeze()
            if id2.numel() == 0:
                sample2 = (dist_neg[i, :] > 0.0).nonzero().squeeze()[0]
            else:
                sample2 = torch.multinomial(
                    torch.exp(-torch.index_select(dist_neg[i], 0, id2)), 1)
                sample2 = id2[sample2][0]

            pos_ids.append(sample1)
            neg_ids.append(sample2)

        pos_ids = torch.Tensor(pos_ids).long()
        neg_ids = torch.Tensor(neg_ids).long()
        if self.cuda:
            pos_ids = pos_ids.cuda()
            neg_ids = neg_ids.cuda()
        return pos_ids, neg_ids

    def forward(self, features):
        batch_size = features.size(0)
        batch_size_class = int(batch_size / self.batch_size_image)

        if batch_size_class != self.batch_size_class:
            self.pos_mask.data.resize_(batch_size, batch_size).fill_(0)
            self.batch_size_class = int(batch_size / self.batch_size_image)
            for i in range(self.batch_size_class):
                start = i * self.batch_size_image
                stop = (i + 1) * self.batch_size_image
                self.pos_mask[start:stop, start:stop].data.fill_(1)
            self.neg_mask.data = torch.add(-self.pos_mask.data, 1)
            self.pos_dist_diag = torch.mul(self.pos_mask, 1e25)
            self.pos_dist_diag.requires_grad = False
            if self.cuda:
                self.pos_mask.data = torch.add(
                    self.pos_mask.data,
                    -torch.eye(batch_size, batch_size).cuda()
                )
            else:
                self.pos_mask = torch.add(
                    self.pos_mask, -torch.eye(batch_size, batch_size)
                )

        feat_norm = torch.mul(
            features, features).sum(dim=1, keepdim=True).repeat(1, batch_size)
        r = torch.mm(features, features.transpose(1, 0))
        distance = torch.add(feat_norm, torch.add(feat_norm.transpose(1, 0),
                                                  -torch.mul(r, 2)))
        distance = torch.clamp(distance, min=1e-8)

        dist_pos = torch.mul(distance, self.pos_mask)
        dist_neg = torch.mul(distance, self.neg_mask)
        dist_neg = torch.add(dist_neg, self.pos_dist_diag)

        score_pos1, _ = dist_pos.data.max(dim=1)
        score_neg1, _ = dist_neg.data.min(dim=1)

        score_pos1 = score_pos1.sqrt()
        score_neg1 = score_neg1.sqrt()
        accuracy = 100 * (score_pos1 < score_neg1).float().sum() / batch_size

        pos_ids, neg_ids = self.sample_from_dist(dist_pos.data, dist_neg.data)
        pos_ids = Variable(pos_ids, requires_grad=False)
        neg_ids = Variable(neg_ids, requires_grad=False)

        score_pos = torch.gather(dist_pos, 1, pos_ids.view(-1, 1))
        score_neg = torch.gather(dist_neg, 1, neg_ids.view(-1, 1))

        score_pos = score_pos.sqrt()
        score_neg = score_neg.sqrt()

        if self.margin > 0:
            diff = self.margin + score_pos - score_neg
            avg_active = (diff > 0.0).float().sum() / batch_size
            loss = torch.clamp(diff, min=0.0)
        else:
            diff = 1e-4 + score_pos - score_neg
            avg_active = (diff > 0.0).float().sum() / batch_size
            loss = self.activation(diff)

        return [loss, avg_active.data[0], accuracy]
