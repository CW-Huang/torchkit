#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""


import numpy as np

import torch


delta = 1e-7
sigmoid = lambda x:torch.nn.functional.sigmoid(x) * (1-delta) + 0.5 * delta

c = - 0.5 * np.log(2*np.pi)
def log_normal(x, mean, log_var, eps=0.00001):
    return - (x-mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var/2. + c

def log_laplace(x, mean, log_scale, eps=0.00001):
    return - torch.abs(x-mean) / (torch.exp(log_scale) + eps) - log_scale - np.log(2)


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


def factorial_gaussian_crossentropy(mean_q, log_var_q, mean_p, log_var_p, 
                                    eps=0.00001):
    """
    - E_q(log p)
    """
    return (
        log_var_p + (mean_q**2 +
                     mean_p**2 -
                     mean_q*mean_p*2 + 
                     torch.exp(log_var_q)) / (torch.exp(log_var_p) + eps) 
    )/2. - c



def factorial_gaussian_entropy(log_var_q):
    """
    - E_q(log q)
    """
    return (1+log_var_q)/2. - c
    

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

def log_sum_exp_np(A, axis=-1, sum_op=np.sum):    
    A_max = np.max(A, axis, keepdims=True)
    B = np.log(sum_op(np.exp(A-A_max),axis,keepdims=True)) + A_max    
    return B

def log_mean_exp_np(A, axis=-1):
    return log_sum_exp_np(A, axis, sum_op=np.mean)
    
    
def mul_grad_value(parameters, mul):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        p.grad.data.mul_(mul)
