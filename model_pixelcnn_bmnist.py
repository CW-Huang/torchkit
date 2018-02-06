#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 02:04:29 2018

@author: chinwei
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
#import matplotlib.pyplot as plt
import os
import scipy.misc


rescale = lambda x: x*0.95 + 0.025
logit = lambda x: torch.log(rescale(x)) - torch.log(1-rescale(x))
logistic = lambda x: 1 / (torch.exp(-x)+1)

cuda = False

class model(object):
    def __init__(self):
        self.mdl = iaf_modules.PixelCNN(1,16,4,5,num_outlayers=1)
        if cuda:
            self.mdl = self.mdl.cuda()
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
                if cuda:
                    x = x.cuda()
                x = Variable(x.view(-1, 1, 28, 28))
                out = nn_.sigmoid(self.mdl((x,0))[0]).permute(0,3,1,2)
                loss = utils.bceloss(out, x).sum(1).sum(1).sum(1).mean()
                loss.backward()
                self.optim.step()
                if ((it + 1) % 100) == 0:
                    print 'Epoch: [%2d] [%4d/%4d] loss: %.8f' % \
                        (epoch+1, it+1, 
                         self.data_loader.dataset.__len__() // 32,
                         loss.data[0])
                 


mdl = model()
mdl.train()

n = 16
spl = utils.varify(np.random.randn(n,1,28,28).astype('float32'))
spl.volative = True
mdl.mdl = mdl.mdl.eval()

for i in range(0,28):
    for j in range(28):
        
        out, _ = mdl.mdl((spl, 0))
        out = out.permute(0,3,1,2)
        proba = nn_.sigmoid(out[:,0,i,j])
        spl.data[:,0,i,j] = torch.bernoulli(proba).data
        #unif = torch.zeros_like(proba)
        #unif.data.uniform_(0,1)
        #spl[:,0,i,j] = torch.ge(proba,unif).float()

#plt.imshow(nn_.sigmoid(out[3,0]).data.numpy().reshape(28,28), cmap='gray')


path = './temp_results'
if not os.path.exists(path):
    os.makedirs(path)

for i in range(n):
    scipy.misc.imsave(path+'/{}.png'.format(i), 
                      nn_.sigmoid(out[i,0]).data.numpy().reshape(28,28))



