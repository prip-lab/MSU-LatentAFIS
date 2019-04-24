# triplet.py

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['BatchHardTripletLoss_ImgSearch', 'RandomBatchTripletLoss_ImgSearch']


class BatchHardTripletLoss_ImgSearch(nn.Module):
    def __init__(self, batch_size_class, batch_size_image,
                 margin=0.0, if_cuda=False):
        super(BatchHardTripletLoss_ImgSearch, self).__init__()
        self.if_cuda = if_cuda
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

        if self.if_cuda:
            self.pos_mask = self.pos_mask.cuda()
            self.neg_mask = self.neg_mask.cuda()
            if self.activation is not None:
                self.activation = self.activation.cuda()
            self.pos_dist_diag = self.pos_dist_diag.cuda()

    def __call__(self, feat_query, feat_gallery):

        batch_size = feat_query.size(0)
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

        norm_query = torch.mul(
            feat_query, feat_query).sum(dim=1, keepdim=True).repeat(1, batch_size)
        norm_gallery = torch.mul(
            feat_gallery, feat_gallery).sum(dim=1, keepdim=True).repeat(1, batch_size)
        r = torch.mm(feat_query, feat_gallery.transpose(1, 0))
        distance = torch.add(
            norm_query, torch.add(norm_gallery.transpose(1, 0), -torch.mul(r, 2)))
        distance = torch.clamp(distance, min=1e-12)

        dist_pos = torch.mul(distance, self.pos_mask)
        dist_neg = torch.mul(distance, self.neg_mask)
        avg_pos_dist = dist_pos[self.pos_mask == 1].sqrt().mean()
        avg_neg_dist = dist_neg[self.neg_mask == 1].sqrt().mean()
        dist_neg = torch.add(dist_neg, self.pos_dist_diag)

        score_pos, _ = dist_pos.max(dim=1)
        idx_pos = dist_pos.argmax(dim=1)[0]
        score_neg, _ = dist_neg.min(dim=1)
        idx_neg = dist_neg.argmin(dim=1)[0]

        score_pos = score_pos.sqrt()
        score_neg = score_neg.sqrt()

        if self.margin > 0:
            diff = self.margin + score_pos - score_neg
            avg_active = (diff > 0).sum() / batch_size
            loss = torch.mean(torch.clamp(diff, min=0.0))
        else:
            diff = 1e-6 + score_pos - score_neg
            avg_active = (diff > 0).sum() / batch_size
            loss = torch.mean(self.activation(score_pos - score_neg))

        return loss, avg_active.item(), idx_pos, idx_neg


class RandomBatchTripletLoss_ImgSearch(nn.Module):
    def __init__(self, batch_size_class, batch_size_image,
                 margin=0.0, if_cuda=False):
        super(RandomBatchTripletLoss_ImgSearch, self).__init__()
        self.if_cuda = if_cuda
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

        if self.if_cuda:
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
                sample2 = (dist_neg[i, :]).argmin()
            elif id2.numel() == 1:
                sample2 = id2
            else:
                sample2 = torch.randint(0,id2.size(0),(1,)).long()
                sample2 = id2[sample2][0]

            pos_ids.append(sample1)
            neg_ids.append(sample2)

        pos_ids = torch.Tensor(pos_ids).long()
        neg_ids = torch.Tensor(neg_ids).long()
        if self.if_cuda:
            pos_ids = pos_ids.cuda()
            neg_ids = neg_ids.cuda()
        return pos_ids, neg_ids

    def __call__(self, feat_query, feat_gallery):
        batch_size = feat_query.size(0)
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
            if self.if_cuda:
                self.pos_mask.data = torch.add(
                    self.pos_mask.data,
                    -torch.eye(batch_size, batch_size).cuda()
                )
            else:
                self.pos_mask = torch.add(
                    self.pos_mask, -torch.eye(batch_size, batch_size)
                )

        norm_query = torch.mul(
            feat_query, feat_query).sum(dim=1, keepdim=True).repeat(1, batch_size)
        norm_gallery = torch.mul(
            feat_gallery, feat_gallery).sum(dim=1, keepdim=True).repeat(1, batch_size)
        r = torch.mm(feat_query, feat_gallery.transpose(1, 0))
        distance = torch.add(
            norm_query, torch.add(norm_gallery.transpose(1, 0), -torch.mul(r, 2)))
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
        pos_ids = pos_ids.view(-1, 1)
        neg_ids = neg_ids.view(-1, 1)
        idx_pos = pos_ids[0][0]
        idx_neg = neg_ids[0][0]

        score_pos = torch.gather(dist_pos, 1, pos_ids)
        score_neg = torch.gather(dist_neg, 1, neg_ids)

        score_pos = score_pos.sqrt()
        score_neg = score_neg.sqrt()

        if self.margin > 0:
            diff = self.margin + score_pos - score_neg
            avg_active = (diff > 0.0).float().sum() / batch_size
            loss = torch.mean(torch.clamp(diff, min=0.0))
        else:
            diff = 1e-4 + score_pos - score_neg
            avg_active = (diff > 0.0).float().sum() / batch_size
            loss = torch.mean(self.activation(diff))

        return loss, avg_active.item(), idx_pos, idx_neg
