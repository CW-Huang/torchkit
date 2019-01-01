# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:47:13 2017
@author: Chin-Wei
"""

import helpers
import os
import cPickle as pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
floatX = 'float32'

import scipy.io
from torch.utils.data import Dataset as Dataset



def load_bmnist_image(root='dataset'):
    helpers.create(root, 'bmnist')
    droot = root+'/'+'bmnist'
    
    if not os.path.exists('{}/binarized_mnist_train.amat'.format(droot)):
        from downloader import download_bmnist
        download_bmnist(droot)
    
    # Larochelle 2011
    path_tr = '{}/binarized_mnist_train.amat'.format(droot)
    path_va = '{}/binarized_mnist_valid.amat'.format(droot)
    path_te = '{}/binarized_mnist_test.amat'.format(droot)
    train_x = np.loadtxt(path_tr).astype(floatX).reshape(50000,784)
    valid_x = np.loadtxt(path_va).astype(floatX).reshape(10000,784)
    test_x = np.loadtxt(path_te).astype(floatX).reshape(10000,784)
    
    
    return train_x, valid_x, test_x


def load_mnist_image(root='dataset',n_validation=1345, state=123):
    helpers.create(root, 'bmnist')
    droot = root+'/'+'bmnist'
    
    if not os.path.exists('{}/train-images-idx3-ubyte'.format(droot)):
        from downloader import download_bmnist
        download_bmnist(droot)
    
    path_tr = '{}/train-images-idx3-ubyte'.format(droot)
    path_te = '{}/t10k-images-idx3-ubyte'.format(droot)
    train_x = np.loadtxt(path_tr).astype(floatX)
    test_x = np.loadtxt(path_te).astype(floatX)
    
    return train_x[:50000], train_x[50000:], test_x
    

def load_cifar10_image(root='dataset',labels=False):
    helpers.create(root, 'cifar10')
    droot = root+'/'+'cifar10'
    
    if not os.path.exists('{}/cifar10.pkl'.format(droot)):
        from downloader import download_cifar10
        download_cifar10(droot)
    
    f = lambda d:d.astype(floatX)
    filename = '{}/cifar10.pkl'.format(droot)
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    if tr_x.max() == 255:
        tr_x = tr_x / 256.
        te_x = te_x / 256.
        
    if labels:
        enc = OneHotEncoder(10)
        tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
        te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
        
        return (f(d) for d in [tr_x, tr_y, te_x, te_y])   
    else:
        return (f(d) for d in [tr_x, te_x])
    
    
def load_omniglot_image(root='dataset',n_validation=1345, state=123):
    helpers.create(root, 'omniglot')
    droot = root+'/'+'omniglot'
    
    if not os.path.exists('{}/omniglot.amat'.format(droot)):
        from downloader import download_omniglot
        download_omniglot(droot)
    
    
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    path = '{}/omniglot.amat'.format(droot)
    omni_raw = scipy.io.loadmat(path)

    train_data = reshape_data(omni_raw['data'].T.astype(floatX))
    test_data = reshape_data(omni_raw['testdata'].T.astype(floatX))

    n = train_data.shape[0]
    
    ind_va = np.random.RandomState(
        state).choice(n, n_validation, replace=False)
    
    ind_tr = np.delete(np.arange(n), ind_va)
    
    return train_data[ind_tr], train_data[ind_va], test_data



def load_caltech101_image(root='dataset'):
    # binary
    # tr: 4100 x 28 x 28
    # va: 2264 x 28 x 28
    # te: 2307 x 28 x 28
    helpers.create(root, 'caltech101')
    droot = root+'/'+'caltech101'
    fn = 'caltech101_silhouettes_28_split1.mat'
    
    if not os.path.exists('{}/{}'.format(droot, fn)):
        from downloader import download_caltech101
        download_caltech101(droot)
    
    ds = scipy.io.loadmat('{}/{}'.format(droot, fn))
    ds = [ds['train_data'], ds['val_data'], ds['test_data']]
    
    return [d.astype(floatX) for d in ds]



class DatasetWrapper(Dataset):

    def __init__(self, dataset, transform=None):

        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        
        sample = self.dataset[ind]
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class InputOnly(Dataset):

    def __init__(self, dataset):

        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind][0]




