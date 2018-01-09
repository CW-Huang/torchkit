#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""


import numpy as np

import torch


delta = 1e-5
sigmoid = lambda x:torch.nn.functional.sigmoid(x) * (1-delta) + 0.5 * delta

c = - 0.5 * np.log(2*np.pi)
def log_normal(x, mean, log_var, eps=0.0001):
    return - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var/2. + c
    
def bceloss(pi, x):
    return - (x * torch.log(pi) + (1-x) * torch.log(1-pi))

def categorical_kl(q, p, logq=None, logp=None):
    """ 
        compute the kl divergence KL(q||p) for categorical distributions q, p
        q, p : (batch_size, num_classes)
    """
    if logq is None:
        logq = torch.log(q)
    if logp is None:
        logp = torch.log(p)
    
    return (q * (logq - logp)).sum(1)
    
    
def varify(x):
    return torch.autograd.Variable(torch.from_numpy(x))

def oper(array,oper,axis=-1,keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper

def log_sum_exp(A, axis=-1, sum_op=torch.sum):    
    maximum = lambda x: x.max(axis)[0]    
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max    
    return B

def log_mean_exp(A, axis=-1):
    return log_sum_exp(A, axis, sum_op=torch.mean)

