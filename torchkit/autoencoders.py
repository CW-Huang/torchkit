#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 18:50:15 2018

@author: chin-weihuang
"""


import nn as nn_
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch
from utils import bceloss

tanh = nn.Tanh()
print 'this is autoencoders'

class MNISTConvEnc(nn.Module):
    
    def __init__(self, dimc, act=nn.ELU()):
        super(MNISTConvEnc, self).__init__()
        self.enc = nn.Sequential(
            nn_.ResConv2d(1,16,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,32,3,2,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,2,padding=1,activation=act),
            act,
            nn_.Reshape((-1,32*4*4)),
            nn_.ResLinear(32*4*4,dimc),
            act
        )
        
    def forward(self, input):
        return self.enc(input)

class MNISTConvDec(nn.Module):
    
    def __init__(self, dimz, dimc, act=nn.ELU()):
        super(MNISTConvDec, self).__init__()
        self.dec = nn.Sequential(
            nn_.ResLinear(dimz,dimc),
            act,
            nn_.ResLinear(dimc,32*4*4),
            act,
            nn_.Reshape((-1,32,4,4)),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(32,32,3,1,padding=1,activation=act),
            act,
            nn_.slicer[:,:,:-1,:-1],                
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(32,16,3,1,padding=1,activation=act),
            act,
            nn_.ResConv2d(16,16,3,1,padding=1,activation=act),
            act,
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn_.ResConv2d(16,1,3,1,padding=1,activation=act),
        )

        
    def forward(self, input):
        return self.dec(input)

class MNISTGatedLinearEnc(nn.Module):
    
    def __init__(self, dimc, act=nn.ELU(), oper=nn_.WNlinear):
        super(MNISTGatedLinearEnc, self).__init__()
        self.enc = nn.Sequential(
            nn_.GatingLinear(784, dimc, oper=oper),
            nn_.GatingLinear(dimc, dimc, oper=oper),
        )
        
    def forward(self, input):
        return self.enc(input)

class MNISTGatedLinearDec(nn.Module):
    
    def __init__(self, dimz, dimc, act=nn.ELU(), oper=nn_.WNlinear):
        super(MNISTGatedLinearDec, self).__init__()
        self.dec = nn.Sequential(
            nn_.GatingLinear(dimz, dimc, oper=oper),
            nn_.GatingLinear(dimc, dimc, oper=oper),
            oper(dimc, 784),
        )
        
    def forward(self, input):
        return self.dec(input)


class BinaryLinear(nn.Module):
    
    def __init__(self, dim1, dim2, oper=nn_.WNlinear):
        super(BinaryLinear, self).__init__()
        self.func = oper(dim1, dim2)
        
    def forward(self, input, temperature=1.0):
        return tanh(self.func(input)/temperature) * (1-nn_.delta)
    
    def sample(self, input, temperature=1.0):
        prob = self.forward(input, temperature) * 0.5 + 0.5
        spl = (prob - torch.rand_like(prob) > 0.0).float() * 2.0 - 1.0
        return spl
    
    def evaluate(self, input, output):
        prob = self.forward(input) * 0.5 + 0.5
        output = output * 0.5 + 0.5
        return - nn_.sum_from_one(bceloss(prob, output))
        

class BinaryNonLinear(nn.Module):
    
    def __init__(self, dim1, dim2, oper=nn_.WNlinear):
        super(BinaryNonLinear, self).__init__()
        self.func = nn.Sequential(
            BinaryLinear(dim1, dim1),
            BinaryLinear(dim1, dim1),
            BinaryLinear(dim1, dim2),
        )
    
    def forward(self, input, temperature=1.0):
        return (self.func(input)/temperature) * (1-nn_.delta)
    
    def sample(self, input, temperature=1.0):
        prob = self.forward(input, temperature) * 0.5 + 0.5
        spl = (prob - torch.rand_like(prob) > 0.0).float() * 2.0 - 1.0
        return spl
    
    def evaluate(self, input, output):
        prob = self.forward(input) * 0.5 + 0.5
        output = output * 0.5 + 0.5
        return - nn_.sum_from_one(bceloss(prob, output))
    
    
        
class BinaryPrior(nn.Module):
    
    def __init__(self, dim):
        super(BinaryPrior, self).__init__()
        self.dim = dim if type(dim) is not int else [dim]
        self.logits = Parameter(torch.zeros(*self.dim))
        self.sigmoid = nn.Sigmoid()
        
    def sample(self, n):
        prob = self.sigmoid(self.logits)
        spl = Variable(
            (torch.rand(n,*self.dim).to(device=prob.device, dtype=prob.dtype) < prob).float()) * 2.0 - 1.0
        return spl
    
    def evaluate(self, z):
        prob = self.sigmoid(self.logits)
        z = z * 0.5 + 0.5
        return - nn_.sum_from_one(bceloss(prob, z))
    
    



