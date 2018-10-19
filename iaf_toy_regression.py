#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 23:59:31 2018

@author: chinwei
"""


import torch
import matplotlib.pyplot as plt
from optim import Adam
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from utils import log_normal, varify

import bnn
import nn as nn_
import flows


if __name__ == '__main__':
    
    np.random.seed(30)
    
    n = 20
    h = 100
    X = np.random.rand(n) * 8 - 4
    Y = X**3 + np.random.randn(n) * 3
    plt.scatter(X,Y)
    
    
    X = X.astype('float32').reshape(n,1)
    Y = Y.astype('float32').reshape(n,1)
    
    X_ = varify(X)
    Y_ = varify(Y)
    
    model = nn.Sequential(
        bnn.sWNLinear(1,h),
        nn.ELU(),
        bnn.sWNLinear(h,1))
    
    bnn.params.merged_sampler.add_common_flowlayer(
        lambda dim: nn_.SequentialFlow( 
                    flows.IAF_DSF(dim, 512, 1, 2, num_ds_dim=16), 
                    flows.FlipFlow(1), 
                    flows.IAF_DSF(dim, 512, 1, 2, num_ds_dim=16)) )
    

    
    std = 100.
    zero = Variable(torch.zeros(1))
    log_var = zero+2*np.log(std)
    
    def prior():
        ll = log_normal(bnn.params.merged_sampler.spls,zero,log_var).sum()
        return ll
    
    def likelihood(x,y):
        out = model(x)
        #return -1*((y - out)**2).sum(1)
        return log_normal(y,out,zero+np.log(9)).sum(1)
    
    def lossf(x,y):
        ll = likelihood(x,y).sum() + prior() + bnn.params.merged_sampler.logdet 
        return -ll/float(n)
    
    

    
    L = 32
    adam = Adam(bnn.params.parameters(), 0.001)
    


    
    T = 2500
    x1, x2 = -6, 6
    y1, y2 = -100, 100
    for i in range(T):
        
        adam.zero_grad()
        bnn.params.sample()
        loss = lossf(X_,Y_)
        loss.backward()
        adam.step()

        if i % 100 == 0:
            print i, loss.data.numpy()[0]
    
    N = 500
    xx = varify(np.linspace(x1,x2,N).astype('float32').reshape(N,1))
    yys = list()
    for i in range(32):
        bnn.params.sample()
        yys.append(model(xx).data.numpy())
    
    xx = xx.data.numpy()[:,0]
    yys = np.concatenate(yys,axis=1)
    yy = yys.mean(1)
    ss = yys.std(1)
    
    plt.fill_between(xx, yy+3*ss, yy-3*ss, facecolor='blue', alpha=0.1)
    plt.fill_between(xx, yy+2*ss, yy-2*ss, facecolor='blue', alpha=0.1)
    plt.fill_between(xx, yy+ss, yy-ss, facecolor='blue', alpha=0.1)
    
    
    plt.plot(xx,yy)
    plt.ylim(y1,y2)
    plt.xlim(x1,x2)
    
    plt.grid('on')
    
#    plt.plot(xx.data.numpy()[:,0], 
#             model(xx).data.numpy()[:,0])
    
    
#       ff= lambda dim: nn_.SequentialFlow( 
#                    flows.IAF(dim, 256, 1, 2, realify=nn_.softplus), 
#                    flows.FlipFlow(1), 
#                    flows.IAF(dim, 256, 1, 2, realify=nn_.softplus),
#                    flows.FlipFlow(1), 
#                    flows.IAF(dim, 256, 1, 2, realify=nn_.softplus))
        
        