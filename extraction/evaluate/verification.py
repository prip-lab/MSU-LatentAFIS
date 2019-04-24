import os
import numpy as np
import scipy.io
import math
import pickle
import time
import multiprocessing
from functools import partial
from sklearn.model_selection import KFold
from .eval_utils import *
import pdb

class FaceVerification:
    def __init__(self, label_filename,
        protocol='BLUFR', metric='cosine',
        nthreads=8,multiprocess=True,
        pair_index_filename=None,template_filename=None,
        pairs_filename=None,nfolds=10,
        nimgs=None, ndim=None):

        self.protocol = protocol
        self.metric = metric
        self.nthreads = nthreads
        self.multiprocess = multiprocess
        self.pair_index_filename = pair_index_filename
        self.template_filename = template_filename
        self.nfolds = nfolds

        if label_filename.endswith('npy'):
            self.label = np.load(label_filename)
        elif label_filename.endswith('txt'):
            self.label = get_labels_from_txt(label_filename)
        else:
            raise(RuntimeError('Format doest not support!'))

        if self.protocol == 'LFW':
            assert(pairs_filename is not None)
            pairfiles = read_pairfile(pairs_filename)
            index_dict = get_index_dict(label_filename)
            self.issame_label,self.pair_indices = get_pair_and_label(pairfiles, index_dict)

    def __call__(self, feat):
        print('Face Verification on {}'.format(self.protocol))
        
        if self.metric == 'cosine':
            feat = normalize(feat)

        if self.protocol == 'BLUFR':
            if self.metric == 'cosine':
                score_mat = np.dot(feat,feat.T)
            elif self.metric == 'Euclidean':
                score_mat = np.zeros((feat.shape[0],feat.shape[0]))
                for i in range(feat.shape[0]):
                    temp=feat[i,:]
                    temp=temp[None,:]
                    temp1=np.sum(np.square(feat-temp),axis=1)
                    score_mat[i,:] = -1*temp1[:]
            else:
                raise(RuntimeError('Metric doest not support!'))
            score_vec,label_vec = get_pairwise_score_label(score_mat,self.label)
            TARs,FARs,thresholds = ROC(score_vec,label_vec)

        elif self.protocol == 'LFW':
            feat1 = feat[self.pair_indices[:,0]]
            feat2 = feat[self.pair_indices[:,1]]
            if self.metric == 'cosine':
                score_vec = np.sum(feat1*feat2, axis=1)
            elif self.metric == 'Euclidean':
                score_vec = -1*np.sum(np.square(feat1 - feat2), axis=1)
            else:
                raise(RuntimeError('The disctnace metric does not support!'))
            avg,std,thd = cross_valid_accuracy(score_vec, self.issame_label,
                self.pair_indices, self.nfolds)
            print("Accuracy is {}".format(avg))
            return avg,std,thd

        elif self.protocol == 'IJBA':
            assert(self.pair_index_filename is not None)
            assert(type(self.pair_index_filename) == str)
            TARs = []
            FARs = []
            thresholds = []
            for i in range(10):
                sidx = str(i+1)
                splitfolder = os.path.join(self.pair_index_filename,'split'+sidx)

                with open(os.path.join(splitfolder,'gen_pairs.csv'), 'r') as f:
                    gen_pairs = f.readlines()
                    gen_pairs = [x.split('\n')[0] for x in gen_pairs]
                with open(os.path.join(splitfolder,'imp_pairs.csv'), 'r') as f:
                    imp_pairs = f.readlines()
                    imp_pairs = [x.split('\n')[0] for x in imp_pairs]
                with open(os.path.join(splitfolder,'temp_dict.pkl'), 'rb') as fp:
                    template = pickle.load(fp)
                pairs = [(0,x) for x in imp_pairs]
                pairs.extend([(1,x) for x in gen_pairs])

                if self.multiprocess:
                    begin = time.time()
                    pool = multiprocessing.Pool(self.nthreads)
                    score_parfunc = partial(score_per_pair, self.metric, feat, template)
                    results = pool.map(score_parfunc, pairs)
                    pool.close()
                    pool.join()
                    label_vec = [x[0] for x in results if x is not None]        
                    score_vec = [x[1] for x in results if x is not None]
                else:
                    label_vec = []
                    score_vec = []
                    begin = time.time()
                    for i,pair in enumerate(pairs):
                        r = score_per_pair(self.metric,feat,template,pair)
                        if r is not None:
                            label_vec.append(r[0])
                            score_vec.append(r[1])
                label_vec = np.array(label_vec).astype(bool)
                score_vec = np.array(score_vec).reshape(-1)
                TAR,FAR,threshold = ROC(score_vec,label_vec)
                TARs.append(TAR)
                FARs.append(FAR)
                thresholds.append(threshold)
            TARs = np.mean(np.array(TARs), axis=0).reshape(-1)
            FARs = np.mean(np.array(FARs), axis=0).reshape(-1)
            thresholds = np.mean(np.array(thresholds), axis=0).reshape(-1)

        elif self.protocol == 'IJBB':
            assert(type(self.pair_index_filename) == dict)
            assert(self.template_filename is not None)
            with open(self.pair_index_filename['genuine'], 'r') as f:
                gen_pairs = f.readlines()
                gen_pairs = [x.split('\n')[0] for x in gen_pairs]
            size = len(gen_pairs)
            with open(self.pair_index_filename['imposter'], 'r') as f:
                imp_pairs = f.readlines()
                imp_pairs = [x.split('\n')[0] for x in imp_pairs[:15*size]]
            with open(self.template_filename, 'rb') as fp:
                template = pickle.load(fp)
            pairs = [(0,x) for x in imp_pairs]
            pairs.extend([(1,x) for x in gen_pairs])
    
            if self.multiprocess:
                begin = time.time()
                pool = multiprocessing.Pool(self.nthreads)
                score_parfunc = partial(score_per_pair, self.metric, feat, template)
                results = pool.map(score_parfunc, pairs)
                pool.close()
                pool.join()
                label_vec = [x[0] for x in results if x is not None]        
                score_vec = [x[1] for x in results if x is not None]
                print('Time of multiple threads is {}'.format(time.time()-begin))
            else:
                label_vec = []
                score_vec = []
                begin = time.time()
                for i,pair in enumerate(pairs):
                    r = score_per_pair(self.metric,feat,template,pair)
                    if r is not None:
                        label_vec.append(r[0])
                        score_vec.append(r[1])
                print('Time of Single thread is {}'.format(time.time()-begin))
            label_vec = np.array(label_vec).astype(bool)
            score_vec = np.array(score_vec).reshape(-1)
            TARs,FARs,thresholds = ROC(score_vec,label_vec)

        else:
            raise(RuntimeError('Protocol doest not support!'))
        
        tar = find_tar(FARs, TARs, 0.001)
        print("TAR is {} at FAR 0.1%".format(tar))

        return tar,FARs,thresholds

def find_tar(FAR, TAR, far):
    i = 0
    while FAR[i] < far:
        i += 1
    tar = TAR[i]
    return tar

def ROC(score_vec, label_vec, thresholds=None, thred_FARs=False, get_false_indices=False):
    ''' Compute Receiver operating characteristic (ROC) with a score and label vector.'''
    assert score_vec.ndim == 1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    
    if thresholds is None:
        thresholds = find_thresholds_by_FAR(score_vec, label_vec, thred_FARs=thred_FARs)

    assert len(thresholds.shape)==1 
    if np.size(thresholds) > 10000:
        print('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))

    # FARs would be check again
    TARs = np.zeros(thresholds.shape[0])
    FARs = np.zeros(thresholds.shape[0])
    false_accept_indices = []
    false_reject_indices = []
    for i,threshold in enumerate(thresholds):
        accept = score_vec >= threshold
        TARs[i] = np.mean(accept[label_vec])
        FARs[i] = np.mean(accept[~label_vec])
        if get_false_indices:
            false_accept_indices.append(np.argwhere(accept & (~label_vec)).flatten())
            false_reject_indices.append(np.argwhere((~accept) & label_vec).flatten())

    if get_false_indices:
        return TARs, FARs, thresholds, false_accept_indices, false_reject_indices
    else:
        return TARs, FARs, thresholds

def cross_valid_accuracy(score_vec, label_vec, indices, nfolds):
    kfold = KFold(n_splits=nfolds, shuffle=False)

    accuracies = np.zeros(nfolds)
    thresholds = np.zeros(nfolds)

    for fold_idx, (train_set, test_set) in enumerate(kfold.split(indices)):
        _,threshold = accuracy(score_vec[train_set],label_vec[train_set])
        acc,_ = accuracy(score_vec[test_set], label_vec[test_set], threshold)
        accuracies[fold_idx] = acc
        thresholds[fold_idx] = threshold

    avg = np.mean(accuracies)
    std = np.std(accuracies)
    thd = np.mean(thresholds)

    return avg,std,thd

def score_per_pair(metric,feat,template,pair):
    label = pair[0]
    index = pair[1]
    if template is None:
        feat1 = feat[index[0]].view(1,-1)
        feat2 = feat[index[1]].view(1,-1)

    else:
        temp1 = index.split(',')[0]
        temp2 = index.split(',')[1]
        if temp1 in template.keys() and temp2 in template.keys():
            idA = template[temp1][1]
            feat1 = np.mean(feat[idA,:],axis=0,keepdims=True)
            idB = template[temp2][1]
            feat2 = np.mean(feat[idB,:],axis=0,keepdims=True)
        else:
            feat1 = None
            feat2 = None
    
    if feat1 is not None and feat2 is not None:
        if metric == 'cosine':
            score = np.dot(feat1,feat2.T)
        elif metric == 'Euclidean':
            score = -1*np.sum(np.square(feat1-feat2))
        else:
            raise(RuntimeError('The disctnace metric does not support!'))
        return label,score
    else:
        return label,None

def find_thresholds_by_FAR(score_vec, label_vec, thred_FARs=False, epsilon=1e-8):
    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    num_neg = len(score_neg)

    assert num_neg >= 1

    if thred_FARs:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = get_FARs()
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds

def accuracy(score_vec, label_vec, threshold=None):
    assert len(score_vec.shape)==1
    assert len(label_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype==np.bool
    
    # find thresholds by TAR
    if threshold is None:
        score_pos = score_vec[label_vec==True]
        thresholds = np.sort(score_pos)[::1]    
        if np.size(thresholds) > 10000:
            warning('number of thresholds (%d) very large, computation may take a long time!' % np.size(thresholds))    
        # Loop Computation
        accuracies = np.zeros(np.size(thresholds))
        for i, threshold in enumerate(thresholds):
            pred_vec = score_vec>=threshold
            accuracies[i] = np.mean(pred_vec==label_vec)
        argmax = np.argmax(accuracies)
        accuracy = accuracies[argmax]
        threshold = np.mean(thresholds[accuracies==accuracy])
    else:
        pred_vec = score_vec>=threshold
        accuracy = np.mean(pred_vec==label_vec)

    return accuracy, threshold
