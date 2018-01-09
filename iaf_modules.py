#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""



import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import nn as nn_
import utils

# aliasing
N_ = None


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta 


tile = lambda x, r: np.tile(x,r).reshape(x.shape[0], x.shape[1]*r)

def get_mask(d0, d1, d2, diag=1):
    m1 = -(-d1/d0)
    m2 = -(-d2/d0)
    mask = np.zeros((d0, d0)).astype('float32')
    iu = np.triu_indices(d0)
    mask[iu] = 1
    if not diag:
        mask[range(d0), range(d0)] = 0
    mask = tile(tile(mask[:,:,N_], m2).T[:,:,N_], m1).T
    mask = np.delete(mask, np.arange(m1*d0-d1)*m1, 0)
    mask = np.delete(mask, np.arange(m2*d0-d2)*m2, 1)
    return mask


class MADE(Module):

    def __init__(self, dim, hid_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU()):
        super(MADE, self).__init__()
        
        oper = nn_.WNlinear
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.num_outlayers = num_outlayers
        self.activation = activation
        
        d0 = dim
        d1 = hid_dim
        d2 = dim
        
        m0 = utils.varify(get_mask(d0, d0, d1, 0))
        mh = utils.varify(get_mask(d0, d1, d1))
        ml = get_mask(d0, d1, d0)
        self.ms = [m0, mh, ml]
        ml_ = utils.varify(
                (ml[:,:,None]*([np.cast['float32'](1),] *\
                               num_outlayers)).reshape(
                               hid_dim, dim*num_outlayers))

        sequels = list()
        for i in range(num_layers-1):
            if i==0:
                sequels.append(oper(d0, d1, True, m0.permute(1,0), False))
                sequels.append(activation)
            else:
                sequels.append(oper(d1, d1, True, mh.permute(1,0), False))
                sequels.append(activation)
                
        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                d1, d2*num_outlayers, True, ml_.permute(1,0))

    def forward(self, input):
        hid = self.input_to_hidden(input)
        return self.hidden_to_output(hid).view(
                -1, self.dim, self.num_outlayers)


class cMADE(Module):

    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 num_outlayers=1, activation=nn.ELU()):
        super(cMADE, self).__init__()
        
        oper = nn_.CWNlinear
        
        self.dim = dim
        self.hid_dim = hid_dim
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = nn_.Lambda(lambda x: (activation(x[0]), x[1]))
        
        
        d0 = dim
        d1 = hid_dim
        d2 = dim
        dc = context_dim
        
        m0 = utils.varify(get_mask(d0, d0, d1, 0))
        mh = utils.varify(get_mask(d0, d1, d1))
        ml = get_mask(d0, d1, d0)
        self.ms = [m0, mh, ml]
        ml_ = utils.varify(
                (ml[:,:,None]*([np.cast['float32'](1),] *\
                               num_outlayers)).reshape(
                               hid_dim, dim*num_outlayers))

        sequels = list()
        for i in range(num_layers-1):
            if i==0:
                sequels.append(oper(d0, d1, dc, m0.permute(1,0), False))
                sequels.append(self.activation)
            else:
                sequels.append(oper(d1, d1, dc, mh.permute(1,0), False))
                sequels.append(self.activation)
                
        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
                d1, d2*num_outlayers, dc, ml_.permute(1,0))

    def forward(self, inputs):
        input, context = inputs
        hid, _ = self.input_to_hidden((input, context))
        out, _ = self.hidden_to_output((hid, context))
        return out.view(-1, self.dim, self.num_outlayers), context



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

