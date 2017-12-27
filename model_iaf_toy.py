#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 01:00:42 2017

@author: chinwei

a simple MADE example
"""



import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import nn as nn_
import flows
import matplotlib.pyplot as plt

from toy_energy import U2 as ef



class model(object):
    
    def __init__(self, target_energy):
        self.mdl = nn_.SequentialFlow( 
                flows.IAF(2, 128, 1, 3), 
                flows.FlipFlow(1), 
                flows.IAF(2, 128, 1, 3),
                flows.FlipFlow(1), 
                flows.IAF(2, 128, 1, 3))

        self.optim = optim.Adam(self.mdl.parameters(), lr=0.0005, 
                                betas=(0.9, 0.999))
        
        self.target_energy = target_energy
        
    def train(self):
        
        total = 2000
        
        for it in range(total):

            self.optim.zero_grad()
            
            spl, logdet, _ = self.mdl.sample(64)
            losses = - self.target_energy(spl) - logdet
            loss = losses.mean()
            
            loss.backward()
            self.optim.step()
            
            if ((it + 1) % 100) == 0:
                print 'Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data[0])
             


# build and train
mdl = model(ef)
mdl.train()


# plot figure
fig = plt.figure()
        
ax = fig.add_subplot(1,2,1)
x = np.linspace(-10,10,1000)
y = np.linspace(-10,10,1000)
xx,yy = np.meshgrid(x,y)
X = np.concatenate((xx.reshape(1000000,1),yy.reshape(1000000,1)),1)
X = X.astype('float32')
X = Variable(torch.from_numpy(X))
Z = ef(X).data.numpy().reshape(1000,1000)
ax.pcolormesh(xx,yy,np.exp(Z))
ax.axis('off')
plt.xlim((-10,10))
plt.ylim((-10,10))

ax = fig.add_subplot(1,2,2)
data = mdl.mdl.sample(1000000)[0].data.numpy()
XX = data[:,0]
YY = data[:,1]
plot = ax.hist2d(XX,YY,200,range=np.array([(-10, 10), (-10, 10)]))
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.axis('off')



