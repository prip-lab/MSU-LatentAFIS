# mAP and topk recall rate for image retrieval

import numpy as np
import torch
from torch.autograd import Variable
import pdb

def main():
    x_query = Variable(torch.rand(3,100))
    x_gallery = Variable(torch.rand(9,100))
    y_query = Variable(torch.LongTensor([0,1,2]))
    y_gallery = Variable(torch.LongTensor([0,0,1,1,1,1,2,2,2]))

    test=ImageRetrieval()
    result1=test(x_query,x_gallery,y_query,y_gallery)
    result2=test.getby_numpy(x_query.data.numpy(),x_gallery.data.numpy(),
        y_query.data.numpy(),y_gallery.data.numpy())
    print('p={},r={}'.format(result1[0],result1[1]))
    print('p={},r={}'.format(result2[0],result2[1]))

class ImageRetrieval:
    def __init__(self, topk=10, cuda=False):
        self.topk = topk
        self.cuda = cuda

    def normalize(self, x, tool, axis=None, epsilon=10e-12):
        ''' Devide the vectors in x by their norms.'''
        if axis is None:
            axis = len(x.shape) - 1
        if tool == 'numpy':
            norm = np.linalg.norm(x, axis=axis, keepdims=True)
        elif tool == 'torch':
            norm = torch.mul(x,x).sum(dim=axis, keepdim=True).sqrt()
        x = x / (norm + epsilon)
        return x

    def __call__(self, x_query, x_gallery, y_query, y_gallery):
        x_query = self.normalize(x_query, 'torch')
        x_gallery = self.normalize(x_gallery, 'torch')
        score_mat = torch.mm(x_query, x_gallery.transpose(1,0))
        temp1 = torch.eye(x_query.size(0))
        temp2 = torch.ones(x_query.size(0))
        score_mask = temp2 - temp1
        if self.cuda:
            score_mask = score_mask.cuda()
        if x_query.size(0) == x_gallery.size(0):
            score_mat = torch.mul(score_mask, score_mat)
        
        # compute label matrix
        y_query = y_query[:,None]
        y_gallery = y_gallery[:,None]
        label_mat = y_query==y_gallery.transpose(1,0)
        label_mat=label_mat.type(torch.FloatTensor)
        
        # sort scores and labels
        _,idx_sorted = torch.sort(-score_mat, dim=1)
        tmp_list = [(label_mat[x, idx_sorted[x]])[None,:] for x in range(label_mat.shape[0])]
        label_sorted = torch.zeros(label_mat.size())
        torch.cat(tmp_list, out=label_sorted)
        if self.cuda:
            label_sorted = label_sorted.cuda()
        if x_query.size(0) == x_gallery.size(0):
            label_sorted = torch.mul(score_mask, label_sorted)
        label_sorted = Variable(label_sorted, requires_grad=False)

        # check the number of matching images
        num_positive = torch.sum(label_sorted, dim=1)
        idx = num_positive.nonzero()

        # compute precision of top positives
        if idx.numel() != 0:
            precision = torch.zeros(idx.size(0))
            precision = Variable(precision, requires_grad=False)
            if self.cuda:
                precision = precision.cuda()
            for i,j in enumerate(idx):
                num = float(num_positive[j])
                temp = label_sorted[j].nonzero()
                den = float(temp[-1][-1])
                if den+1 == 0:
                    pdb.set_trace()
                precision[i] = num/(den+1)
            precision = torch.mean(precision).item()
        else:
            precision = 0.0

        # compute top k recall
        if idx.numel() != 0:
            if label_sorted.size(-1) < self.topk:
                topk = label_sorted.size(-1)
            else:
                topk = self.topk
            total = torch.sum(label_sorted[idx,:topk].view(-1,topk), dim=1)
            num = float(total.nonzero().size(0))
            den = float(idx.size(0))
            recall = num/den
        else:
            recall = 0.0

        return precision,recall

    def getby_numpy(self, x_query, x_gallery, y_query, y_gallery):
        x_query = self.normalize(x_query,'numpy')
        x_gallery = self.normalize(x_gallery,'numpy')
        score_mat = np.dot(x_query,x_gallery.T)
        
        # compute label matrix
        y_query = y_query[:,None]
        y_gallery = y_gallery[:,None]
        label_mat = y_query==y_gallery.T
        
        idx_sorted = np.argsort(-score_mat, axis=1)
        label_sorted = [label_mat[x, idx_sorted[x]] for x in range(label_mat.shape[0])]
        label_sorted = np.array(label_sorted)
        label_sorted = label_sorted.astype(float)
        
        # check the number of matching images
        num_positive = np.sum(label_sorted, axis=1)
        idx = num_positive.nonzero()

        # compute precision of top positives
        if len(idx[0]) != 0:
            precision = np.zeros((len(idx[0])))
            for i,j in enumerate(idx[0]):
                num = float(num_positive[j])
                temp = label_sorted[j].nonzero()
                den = float(temp[0][-1])
                precision[i] = num/(den+1)
            precision = float(np.mean(precision))
        else:
            precision = 0.0

        # compute top k recall
        if len(idx[0]) != 0:
            total = np.sum(label_sorted[idx,:self.topk].reshape(-1,self.topk), axis=1)
            num = float(len(total.nonzero()[0]))
            den = float(len(idx[0]))
            recall = num/den
        else:
            recall = 0.0

        return precision,recall
