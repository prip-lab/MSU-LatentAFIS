# lfw verify -- standard protocol

import torch

import os
from sklearn.model_selection import KFold

class LFWVerification():
    def __init__(self, args,imagepaths_filename, pairs_filename,
        cuda=False, metric='cos_dist', nfolds=10):
        self.cuda = cuda
        self.metric = metric
        self.nfolds = 10

        pairfiles = self.read_pairfile(pairs_filename)
        index_dict = self.get_index_dict(imagepaths_filename)
        self.get_pair_and_label(pairfiles, index_dict)

    def get_index_dict(self, imagepaths_filename):
        index_dict = {}
        with open(imagepaths_filename, 'r') as f:
            lines = list(f)
            imagepaths = [x.split('\n')[0] for x in lines]
        for i,path in enumerate(imagepaths):
            index_dict[os.path.splitext(os.path.basename(path))[0]] = i
        return index_dict

    def read_pairfile(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return pairs

    def get_pair_and_label(self, pairfiles, index_dict):
        nrof_skipped_pairs = 0
        self.pair_indices = []
        self.issame_label = []
        for pair in pairfiles:
            if len(pair) == 3:
                path0 = pair[0] + '_' + '%04d' % int(pair[1])
                path1 = pair[0] + '_' + '%04d' % int(pair[2])
                issame = True
            elif len(pair) == 4:
                path0 = pair[0] + '_' + '%04d' % int(pair[1])
                path1 = pair[2] + '_' + '%04d' % int(pair[3])
                issame = False
            self.pair_indices.append((index_dict[path0],index_dict[path1]))
            self.issame_label.append(issame)
        self.pair_indices = torch.LongTensor(self.pair_indices)
        self.issame_label = torch.ByteTensor(self.issame_label)
        if self.cuda:
            self.issame_label = self.issame_label.cuda()
            self.pair_indices = self.pair_indices.cuda()

    def normalize(self, x, axis=None, epsilon=10e-12):
        ''' Devide the vectors in x by their norms.'''
        if axis is None:
            axis = len(x.shape) - 1
        norm = torch.mul(x,x).sum(dim=axis, keepdim=True).sqrt()
        x = x / (norm + epsilon)
        return x

    def get_scores(self):
        feats1 = self.feats[self.pair_indices[:,0]]
        feats2 = self.feats[self.pair_indices[:,1]]
        if self.metric == 'cos_dist':
            feats1 = self.normalize(feats1)
            feats2 = self.normalize(feats2)
            self.score = torch.mul(feats1, feats2).sum(dim=1)
        elif self.metric == 'l2_norm':
            self.score = -1*torch.norm(feats1 - feats2, p=2,dim=1)
        else:
            raise(RuntimeError('The disctnace metric does not support!'))

    def get_accuracy(self, score, label, threshold=None):
        assert(len(score.size()) == 1)
        assert(len(label.size()) == 1)
        assert(score.size() == label.size())
        if self.cuda:
            assert(label.data.type() == 'torch.cuda.ByteTensor')
        else:
            assert(label.data.type() == 'torch.ByteTensor')

        den = float(score.size()[0])

        if threshold is None:
            # get possible threshold
            score_pos = score[label == 1]
            thresholds,_ = torch.sort(score_pos)

            accuracies = torch.zeros(thresholds.size())
            if self.cuda:
                accuracies = accuracies.cuda()
            for i,threshold in enumerate(thresholds):
                pred = score>=threshold
                accuracies[i] = float(torch.sum(pred==label)) / den

            accuracy = torch.max(accuracies)
            threshold = torch.mean(thresholds[accuracies==accuracy])
        else:
            pred = score>=threshold
            accuracy = float(torch.sum(pred==label)) / den

        return accuracy,threshold

    def __call__(self, feats, labels=None):
        self.feats = feats
        self.get_scores()

        kfold = KFold(n_splits=self.nfolds, shuffle=False)

        accuracies = torch.zeros(self.nfolds)
        if self.cuda:
            accuracies = accuracies.cuda()

        for fold_idx, (train_set, test_set) in enumerate(kfold.split(self.pair_indices)):
            train_set = torch.LongTensor(train_set)
            test_set = torch.LongTensor(test_set)
            if self.cuda:
                train_set = train_set.cuda()
                test_set = test_set.cuda()
            _,threshold = self.get_accuracy(self.score[train_set],self.issame_label[train_set])
            accuracy,_ = self.get_accuracy(self.score[test_set], self.issame_label[test_set], threshold)
            accuracies[fold_idx] = accuracy

        avg = torch.mean(accuracies)
        std = torch.std(accuracies)

        return avg




