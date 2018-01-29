#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""



import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module

import nn as nn_


# aliasing
N_ = None


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta 


tile = lambda x, r: np.tile(x,r).reshape(x.shape[0], x.shape[1]*r)


# %------------ MADE ------------% 

def get_rank(max_rank, num_out):
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = np.random.choice(max_rank,excess,False)
    rank_out = np.delete(rank_out,remove_ind)
    np.random.shuffle(rank_out)
    return rank_out.astype('float32')
    

def get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype('float32')

def get_masks_all(ds, fixed_order=False):
    # ds: list of dimensions dx, d1, d2, ... dh, dx, 
    #                       (2 in/output + h hidden layers)
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx-1, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - 1
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [np.zeros([ds[i+1],ds[i]]).astype('float32') for \
              i in range(len(ds)-1)]
    assert np.all(np.diag(reduce(np.dot,ms[::-1])) == 0), 'wrong masks'
    
    return ms, rx


def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False):
    ms, rx = get_masks_all([dim,]+[dh for i in range(num_layers-1)]+[dim,],
                           fixed_order)
    ml = ms[-1]
    ml_ = (ml.transpose(1,0)[:,:,None]*([np.cast['float32'](1),] *\
                           num_outlayers)).reshape(
                           dh, dim*num_outlayers).transpose(1,0)
    ms[-1]  = ml_
    return ms, rx


class MADE(Module):

    def __init__(self, dim, hid_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU(), fixed_order=False):
        super(MADE, self).__init__()
        
        oper = nn_.WNlinear
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_outlayers = num_outlayers
        self.activation = activation
        
        
        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers.
                           fixed_order)
        ms = map(torch.from_numpy, ms)
        self.rx = rx
        
        sequels = list()
        for i in range(num_layers-1):
            if i==0:
                sequels.append(oper(dim, hid_dim, True, ms[i], False))
                sequels.append(activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, True, ms[i], False))
                sequels.append(activation)
                
        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                hid_dim, dim*num_outlayers, True, ms[-1])

    def forward(self, input):
        hid = self.input_to_hidden(input)
        return self.hidden_to_output(hid).view(
                -1, self.dim, self.num_outlayers)

    def randomize(self):
        ms, rx = get_masks(self.dim, self.hid_dim,
                           self.num_layers, self.num_outlayers)
        for i in range(self.num_layers-1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i*2].mask.is_cuda:
                mask = mask.cuda()
            self.input_to_hidden[i*2].mask.data.zero_().add_(mask)
        self.rx = rx

class cMADE(Module):

    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU(), fixed_order=False):
        super(cMADE, self).__init__()
        
        oper = nn_.CWNlinear
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = nn_.Lambda(lambda x: (activation(x[0]), x[1]))
        
        
        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers,
                           fixed_order)
        ms = map(torch.from_numpy, ms)
        self.rx = rx
        
        sequels = list()
        for i in range(num_layers-1):
            if i==0:
                sequels.append(oper(dim, hid_dim, context_dim, 
                                    ms[i], False))
                sequels.append(self.activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, context_dim, 
                                    ms[i], False))
                sequels.append(self.activation)
                
        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                hid_dim, dim*num_outlayers, context_dim, ms[-1])
        

    def forward(self, inputs):
        input, context = inputs
        hid, _ = self.input_to_hidden((input, context))
        out, _ = self.hidden_to_output((hid, context))
        return out.view(-1, self.dim, self.num_outlayers), context

    def randomize(self):
        ms, rx = get_masks(self.dim, self.hid_dim,
                           self.num_layers, self.num_outlayers)
        for i in range(self.num_layers-1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i*2].mask.is_cuda:
                mask = mask.cuda()
            self.input_to_hidden[i*2].mask.zero_().add_(mask)
        self.rx = rx



# %------------ PixelCNN (stacks) ------------% 



def get_conv2d_mask(filter_shape, pp_mask=0):
    """
    pp_mask: per pixel mask (generated by the same generator as MADE)
    """
    mask = np.ones(filter_shape, dtype='float32')

    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            if i > filter_shape[2]//2:
                mask[:,:,i,j] = 0
            if i == filter_shape[2]//2 and j > filter_shape[3]//2:
                mask[:,:,i,j] = 0

    mask[:,:,filter_shape[2]//2,filter_shape[3]//2] = pp_mask
    
    return mask
    
    


class _PixelCNN_Block(Module):
    
    def __init__(self, dim_in, dim_out, filter_size=3,
                 activation=nn.ELU(), pp_mask=None):
        super(_PixelCNN_Block, self).__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.filter_size =filter_size
        self.activation = activation
        
        
        fs = filter_size
        mask = torch.from_numpy(
            get_conv2d_mask((dim_out, dim_in, 1, fs), pp_mask))
        
        # vertical stack
        self.v0 = nn_.WNconv2d(
            dim_in, dim_out, (fs//2+1, fs), padding=(fs//2+1, fs//2))
        self.v1 = nn_.WNconv2d(
            dim_out, dim_out, (1, 1), padding=(0, 0))
        
        # horizontal stack
        self.h01 = nn_.WNconv2d(
            dim_in, dim_out, (1, fs), padding=(0, fs//2), mask=mask)
        self.h02 = nn_.WNconv2d(
            dim_out, dim_out, (1, fs), padding=(0, fs//2), mask=None)
        
    
    
    def forward(self, inputs):
        vin, hin = inputs
        
        vout_ = self.v0(vin)
        vout = vout_[:,:,1:-(self.filter_size//2)-1,:]
        vh = self.v1(vout_)[:,:,:-(self.filter_size//2)-2,:]
        
        h1 = self.h01(hin)
        h2 = self.h02(vh)
        hout = h1 + h2
        
        return (vout, hout)
        
        
        
        
        

class PixelCNN(Module):
    
    def __init__(self, dim, hid_dim, num_layers,
                 filter_size=3, filter_size0=7,
                 num_outlayers=1, activation=nn.ELU()):
        super(PixelCNN, self).__init__()
        
        assert filter_size % 2 == 1, 'PixelCNN module only supports odd ' \
                                     'values of filter_size'
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.filter_size =filter_size
        self.num_outlayers = num_outlayers
        self.activation = nn_.Lambda(lambda x: (activation(x[0]), x[1]))
        oper = _PixelCNN_Block
        
        ms, rx = get_masks(dim, hid_dim, num_layers, num_outlayers)
        self.rx = rx
        
        sequels = list()
        for i in range(num_layers):
            if i==0:
                sequels.append(oper(dim, hid_dim, filter_size0, 
                                    activation=activation, 
                                    pp_mask=ms[i]))
                sequels.append(self.activation)
            elif i==num_layers-1:
                sequels.append(oper(hid_dim, dim*num_outlayers, filter_size,
                                    activation=activation, 
                                    pp_mask=ms[i]))
            else:
                sequels.append(oper(hid_dim, hid_dim, filter_size, 
                                    activation=activation, 
                                    pp_mask=ms[i]))
                sequels.append(self.activation)
        
        self.blocks = nn.Sequential(*sequels)
    
    def forward(self, input):
        _, out = self.blocks((input, input))
        f1 = out.size(2)
        f2 = out.size(3)
        return out.contiguous().view(-1, self.dim, self.num_outlayers, f1, f2)

       


if __name__ == '__main__':
    
    inp = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,784).astype('float32')))
    input = inp*2
    mdl = MADE(784, 1000, 3, 2)
    print mdl(input).size()
    
    mdl = cMADE(784, 1000, 200, 3, 2)
    con = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,200).astype('float32')))
    inputs = (input, con)
    print mdl(inputs)[0].size()

    mdl = PixelCNN(1,1,2,num_outlayers=1)
    
    
    
    