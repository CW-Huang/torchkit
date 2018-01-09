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
    return - log_var/2. - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) + c
    
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
    
    
    
    
    
    
    
