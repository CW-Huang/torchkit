#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:02:58 2017

@author: Chin-Wei
"""

import torch
from torch.nn import Module
import nn as nn_
from torch.autograd import Variable

sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size())>2 else sum1(x)



class LinearFlow(Module):
    
    def __init__(self, dim, context_dim, 
                 oper=nn_.ResLinear, realify=nn_.softplus):
        super(LinearFlow, self).__init__()
        self.realify = realify
        
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

 
class BlockAffineFlow(Module):
    # RealNVP

    def __init__(self, dim, context_dim, hid_dim, 
                 mask=0, realify=nn_.softplus):
        super(BlockAffineFlow, self).__init__()
        self.mask = mask
        self.dim = dim
        self.realify = realify
        self.gpu = True
        
        self.hid = nn_.WNBilinear(dim, context_dim, hid_dim)
        self.mean = nn_.ResLinear(hid_dim, dim)
        self.lstd = nn_.ResLinear(hid_dim, dim)

    def forward(self,inputs):
        x, logdet, context = inputs
        mask = Variable(torch.zeros(1, self.dim))
        if self.gpu:
            mask = mask.cuda()

        if self.mask:
            mask[:, self.dim/2:].data += 1
        else:
            mask[:, :self.dim/2].data += 1

        hid = self.hid(x*mask, context)
        mean = self.mean(hid)*(-mask+1) + self.mean.dot_h1.bias
        lstd = self.lstd(hid)*(-mask+1) + self.lstd.dot_h1.bias
        std = self.realify(lstd)

        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context

    def cuda(self):
        self.gpu = True
        return super(cuda, self).cuda()
