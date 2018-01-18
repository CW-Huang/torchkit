#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 01:00:42 2017

@author: chinwei

a simple MADE example
"""



import numpy as np
import utils

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nn as nn_
import iaf_modules
import matplotlib.pyplot as plt


rescale = lambda x: x*0.95 + 0.025
logit = lambda x: torch.log(rescale(x)) - torch.log(1-rescale(x))
logistic = lambda x: 1 / (torch.exp(-x)+1)

class model(object):
    
    def __init__(self):
        self.mdl = iaf_modules.MADE2(784, 512, 4, 1)

        self.optim = optim.Adam(self.mdl.parameters(), lr=0.001, 
                                betas=(0.9, 0.999))
        
        trs = transforms.Compose([transforms.ToTensor()])
        self.data_loader = DataLoader(datasets.MNIST('data/mnist',
                                                     train=True,
                                                     download=True,
                                                     transform=trs),
                                      batch_size = 32,
                                      shuffle = True)
            
    def train(self):
        
        
        for epoch in range(10):
            for it, (x, y) in enumerate(self.data_loader):
                self.optim.zero_grad()
                
                x = torch.bernoulli(x)
                x = Variable(x.view(-1, 784))
                out = nn_.sigmoid(self.mdl(x)[:,:,0])
                loss = utils.bceloss(out, x).sum(1).mean()
                
                loss.backward()
                self.optim.step()
                
                if ((it + 1) % 10) == 0:
                    print 'Epoch: [%2d] [%4d/%4d] loss: %.8f' % \
                        (epoch+1, it+1, 
                         self.data_loader.dataset.__len__() // 32,
                         loss.data[0])
                 
                self.mdl.randomize()

mdl = model()
mdl.train()


spl = utils.varify(np.random.randn(64,784).astype('float32'))

ranks = (mdl.mdl.rx)
ind = np.argsort(ranks)
for i in range(784):
    
    out = mdl.mdl(spl)
    spl[:,ind[i]] = torch.bernoulli(nn_.sigmoid(out[:,ind[i]]))

plt.imshow(nn_.sigmoid(out[56]).data.numpy().reshape(28,28), cmap='gray')




