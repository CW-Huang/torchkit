#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:02:58 2017

@author: Chin-Wei
"""

import torch
import torch.nn as nn
from torch.nn import Module
import nn as nn_
from torch.autograd import Variable
import iaf_modules 
import numpy as np

sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size())>2 else sum1(x)



class BaseFlow(Module):
    
    
    def sample(self, n=1, context=None):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim,]
        
        spl = Variable(torch.FloatTensor(n,*dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.random.rand(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
            np.zeros((n,self.context_dim)).astype('float32')))
            
        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()
        
        return self.forward((spl, lgd, context))
    
    
    
    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class LinearFlow(BaseFlow):
    
    def __init__(self, dim, context_dim, 
                 oper=nn_.ResLinear, realify=nn_.softplus):
        super(LinearFlow, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        self.mean = oper(context_dim, dim)
        self.lstd = oper(context_dim, dim)
        
    def forward(self,inputs):
        x, logdet, context = inputs
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)
        
        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context   


class IAF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.softplus):
        super(IAF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        self.made = iaf_modules.cMADE(
                dim, hid_dim, context_dim, num_layers, 2, activation)
       
        
    def forward(self,inputs):
        x, logdet, context = inputs
        out, _ = self.made((x, context))
        mean = out[:,:,0]
        lstd = out[:,:,1]
        std = self.realify(lstd)
        
        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context


class FlipFlow(BaseFlow):
    
    def __init__(self, dim):
        self.dim = dim
        super(FlipFlow, self).__init__()
        
    def forward(self, inputs):
        input, logdet, context = inputs
        
        dim = self.dim
        index = Variable(
                getattr(torch.arange(input.size(dim)-1, -1, -1), (
                        'cpu', 'cuda')[input.is_cuda])().long())
        
        output = torch.index_select(input, dim, index)
        
        return output, logdet, context
    

if __name__ == '__main__':
    
    
    
    
    inp = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,784).astype('float32')))
    con = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,200).astype('float32')))
    lgd = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2).astype('float32')))
    
    
    mdl = IAF(784, 1000, 200, 3)
    
    inputs = (inp, lgd, con)
    print mdl(inputs)[0].size()


    


