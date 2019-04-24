# mAP and topk recall rate for image retrieval

import numpy as np
import torch
from torch.autograd import Variable
import pdb

class ImageRetrieval_strict:
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
        y_query = y_query[:,None]
        y_gallery = y_gallery[:,None]
        label_mat = y_query==y_gallery.transpose(1,0)
        label_mat=label_mat.type(torch.FloatTensor)
        
        _,idx_sorted = torch.sort(-score_mat, dim=1)
        tmp_list = [(label_mat[x, idx_sorted[x]])[None,:] for x in range(label_mat.shape[0])]
        label_sorted = torch.zeros(label_mat.size())
        torch.cat(tmp_list, out=label_sorted)
        label_sorted = Variable(label_sorted, requires_grad=False)
        
        # initialize varaible for mAP and recall computation
        AP = torch.zeros(x_query.shape[0])
        pre_nfound = torch.zeros(x_query.shape[0])
        recall = torch.zeros(x_query.shape[0])
        AP = Variable(AP, requires_grad=False)
        pre_nfound = Variable(pre_nfound, requires_grad=False)
        recall = Variable(recall, requires_grad=False)
        if self.cuda:
            label_sorted = label_sorted.cuda()
            AP = AP.cuda()
            pre_nfound = pre_nfound.cuda()
            recall  = recall.cuda()

        # compute mAP
        num_positive = torch.sum(label_sorted, dim=1)
        idx = num_positive.nonzero()
        if idx.numel() != 0:
            AP = AP[idx].view(-1)
            pre_nfound = pre_nfound[idx].view(-1)
            num_positive = num_positive[idx].view(-1)
            for k in range(x_gallery.shape[0]):
                if k != 0:
                    pre_nfound = num_found
                num_found = torch.sum(label_sorted[idx,0:k+1].view(-1,k+1), dim=1)
                p = num_found / float(k+1)                           
                r = (num_found - pre_nfound) / num_positive
                AP += p*r 
            mAP = torch.mean(AP)
        else:
            mAP = 0.0

        # compute top k recall
        if idx.numel() != 0:
            recall = recall[idx]
            recall = torch.sum(label_sorted[idx,:self.topk].view(-1,self.topk), dim=1) / num_positive
            recall = torch.mean(recall)
        else:
            recall = 0.0

        return mAP, recall

    def getby_numpy(self, x_query, x_gallery, y_query, y_gallery):
        x_query = self.normalize(x_query, 'numpy')
        x_gallery = self.normalize(x_gallery, 'numpy')
        score_mat = np.dot(x_query, x_gallery.T)
        y_query = y_query[:,None]
        y_gallery = y_gallery[:,None]
        label_mat = y_query==y_gallery.T
        
        idx_sorted = np.argsort(-score_mat, axis=1)
        label_sorted = [label_mat[x, idx_sorted[x]] for x in range(label_mat.shape[0])]
        label_sorted = np.array(label_sorted)
        label_sorted = label_sorted.astype(float)
        
        # compute mAP
        AP = np.zeros(x_query.shape[0])
        pre_nfound = np.zeros(x_query.shape[0])
        num_positive = np.sum(label_sorted, axis=1)
        idx = num_positive.nonzero()
        if len(idx[0]) != 0:
            AP = AP[idx].squeeze()
            pre_nfound = pre_nfound[idx].squeeze()
            num_positive = num_positive[idx].squeeze()
            for k in range(x_gallery.shape[0]):
                if k != 0:
                    pre_nfound = num_found
                num_found = np.sum(label_sorted[idx,0:k+1].reshape(-1,k+1), axis=1)
                p = num_found / float(k+1)
                r = (num_found - pre_nfound) / num_positive
                AP += p*r 
            mAP = np.mean(AP)
        else:
            mAP = 0.0

        # compute top k recall
        recall = np.zeros(x_query.shape[0])
        if len(idx[0]) != 0:
            recall = recall[idx]
            recall = np.sum(label_sorted[idx,:self.topk].reshape(-1,self.topk), axis=1) / num_positive
            recall = np.mean(recall)
        else:
            recall = 0.0

        return mAP, recall