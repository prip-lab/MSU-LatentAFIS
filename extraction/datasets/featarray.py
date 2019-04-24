# featarray.py

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
import scipy.io
import pickle
from sklearn.utils import shuffle

def get_labels_from_txt(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        class_sorted = [x.split('/')[-2] for x in lines]
        classname = []
        classname[:] = class_sorted[:]
        class_sorted.sort()
    labels = [int(class_sorted.index(x)) for x in classname]
    return labels

class Featarray(data.Dataset):
    def __init__(self, feat_filename, if_norm=False):

        self.if_norm = if_norm

        if type(feat_filename) == np.ndarray:
            self.feats = feat_filename
        elif feat_filename.endswith('npy'):
            self.feats = np.load(feat_filename)
        elif feat_filename.endswith('mat'):
            self.feats = scipy.io.loadmat(feat_filename)
            self.feats = self.feats['feat']
        elif feat_filename.endswith('npz'):
            self.feats = np.load(feat_filename)
            self.feats = self.feats['feat']
        else:
            raise(RuntimeError('Format does not support!'))

        self.nimgs = self.feats.shape[0]

    def get_feat(self, f):
        f = torch.Tensor(f)
        return f

    def __len__(self):
        return self.nimgs

    def __getitem__(self, index):
        feature = self.get_feat(self.feats[index,:])
        # normalization
        if self.if_norm:
            norm = float(np.linalg.norm(feature))
            feature = feature/norm

        return feature
