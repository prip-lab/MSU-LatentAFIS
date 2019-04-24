# pathmat.py

import torch
import torch.utils.data as data

import os
import numpy as np
from sklearn.utils import shuffle
import scipy.io
import pickle


def normalize(x, ord=None, axis=None, epsilon=10e-12):
    ''' Devide the vectors in x by their norms.'''
    if axis is None:
        axis = len(x.shape) - 1
    norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
    x = x / (norm + epsilon)
    return x

class Featpair(data.Dataset):
    def __init__(self, feat_filename, pair_filename, if_norm,
        nimgs, ndim, template_filename=None):
        self.nimgs = nimgs
        self.ndim = ndim
        self.if_norm = if_norm
        if template_filename is not None:
            with open(template_filename, 'rb') as fp:
                self.temp_dict = pickle.load(fp)
        else:
             self.temp_dict = None
        
        if feat_filename.endswith('npm'):
            self.feats = np.memmap(feat_filename, dtype='float32', mode='r', shape=(nimgs,ndim))
        elif feat_filename.endswith('npy'):
            self.feats = np.load(feat_filename)
        elif feat_filename.endswith('mat'):
            self.feats = scipy.io.loadmat(feat_filename)
            self.feats = self.feats['feat']
        elif feat_filename.endswith('npz'):
            self.feats = np.load(feat_filename)
            self.feats = self.feats['feat']
        else:
            raise(RuntimeError('Feature format does not support!'))

        if pair_filename.endswith('pkl'):
            with open(pair_filename, 'rb') as fp:
                self.ind_pairs = pickle.load(fp)
        elif pair_filename.endswith('npy'):
            self.ind_pairs = np.load(pair_filename)
        elif pair_filename.endswith('csv'):
            with open(pair_filename, 'r') as f:
                lines = f.readlines()
                self.ind_pairs = [x.split('\n')[0] for x in lines]
        else:
            raise(RuntimeError('IDX file format does not support!'))

    def __len__(self):
        if self.temp_dict is None:
            # shape of ind_pairs: 2*N. (N is the number of pairs)
            return self.ind_pairs.shape[1]
        else:
            return len(self.ind_pairs)

    def get_feat(self, x):
        f = np.zeros((2,self.ndim))
        if self.temp_dict is None:            
            f[0,:] = self.feats[self.ind_pairs[0,x],:]
            f[1,:] = self.feats[self.ind_pairs[1,x],:]
            if self.if_norm:
                f = normalize(f)
        else:
            pair = self.ind_pairs[x]
            temp1 = pair.split(',')[0]
            temp2 = pair.split(',')[1]
            idA = self.temp_dict[temp1][1]
            idB = self.temp_dict[temp2][1]
            if self.if_norm:
                f[0,:] = np.mean(normalize(self.feats[idA,:]), axis=0)
                f[1,:] = np.mean(normalize(self.feats[idB,:]), axis=0)
            else:
                f[0,:] = np.mean(self.feats[idA,:], axis=0)
                f[1,:] = np.mean(self.feats[idB,:], axis=0)
        f = torch.Tensor(f)
        
        return f

    def __getitem__(self, index):
        x = int(index)

        feature = self.get_feat(x)

        return feature
