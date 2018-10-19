#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:43:15 2017

@author: chinwei


note that thought it's called energy function here, they are actually 
log-density (negative energy)

"""

import torch as T
from utils import log_normal, varify

import torch
cuda = torch.cuda.is_available()
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


delta = 1e-6


def logsumexp(inputs, dim=None, keepdim=False):
    return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)
    

def U1(Z):
    z1 = Z[:,0]
    E1 = 0.5 * (((Z**2).sum(1)**0.5 - 2.) / 0.4) ** 2
    E2 = - T.log(T.exp(-0.5 * ((z1-2)/0.6)**2) + 
                 T.exp(-0.5 * ((z1+2)/0.6)**2))   
    return - (E1 + E2)
 
def U2(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    E1 = 0.5 * (((0.5*Z**2).sum(1)**0.5 - 2.) / 0.5) ** 2
    E2 = - T.log(T.exp(-0.5 * ((z1-2)/0.6)**2) + 
                 T.exp(-0.5 * ((T.sin(z1)/0.5))**2) + 
                 T.exp(-0.5 * ((z1+z2+2.5)/0.6)**2))   
    return - (E1 + E2)
  
def U3(Z):
    z1 = Z[:, 0]
    z2 = Z[:, 1]
    R = 2.0
    return - 1.*(R-(z1**2+.5*z2**2)**0.5)**2
    

def U4(Z, small=False):
    
    Z = Z*2
    if small:
        mean = varify(np.array([[-2., 0.],
                                [2., 0.],
                                [0., 2.],
                                [0., -2.]],
                                dtype='float32'))
        lv = Variable(np.log(torch.ones(1)*0.2))
    else:
        mean = varify(np.array([[-5., 0.],
                                [5., 0.],
                                [0., 5.],
                                [0., -5.]],
                                dtype='float32'))
        lv = Variable(np.log(torch.ones(1)*1.5))
    
    if cuda:
        mean = mean.cuda()
        lv = lv.cuda()
    
    d1 = log_normal(Z, mean[None,0,:], lv).sum(1)+np.log(0.1)
    d2 = log_normal(Z[:,:], mean[None,1,:], lv).sum(1)+np.log(0.3)
    d3 = log_normal(Z[:,:], mean[None,2,:], lv).sum(1)+np.log(0.4)
    d4 = log_normal(Z[:,:], mean[None,3,:], lv).sum(1)+np.log(0.2)
    
    return logsumexp(torch.cat(
        [d1[:,None],d2[:,None],d3[:,None],d4[:,None]],1),1) + 2.5
    

def U5(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    w1 = torch.sin(z1*0.5*math.pi)
    return - 0.5*((z2-w1)/0.4)**2  - 0.1 * (z1**2) 


def U6(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    w1 = torch.sin(z1*0.5*math.pi)
    w2 = 3*torch.exp(-0.5*(z1-2)**2)
    return logsumexp(torch.cat(
            [(-0.5*((z2-w1)/0.35)**2)[:,None],
             (-0.5*((z2-w1+w2)/0.35)**2)[:,None]], 1), 1) - 0.05 * (z1**2)



def U7(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    w1 = torch.cos(z1*0.35*math.pi)
    w2 = 2*torch.exp(-0.2*((z1-2)*2)**2)
    return logsumexp(torch.cat(
            [(-0.5*((z2-w1)/0.35)**2)[:,None],
             (-0.5*((z2-w1+w2)/0.35)**2)[:,None]], 1), 1) - 0.1 * ((z1-2)**2) 


def U8(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    w1 = torch.sin(z1*0.5*math.pi)
    w3 = 2.5*F.sigmoid((z1-2)/0.3)

    return logsumexp(torch.cat(
          [(logsumexp(torch.cat(
                  [(-0.5*((z2-w1)/0.4)**2)[:,None],
                   (-0.5*((z2-w1+w3)/0.35)**2)[:,None]], 1), 1) - 
                    0.05 * ((z1-2)**2+(z2+3)**2))[:,None] ,
         -2.5*(Z[:,0:1]-2)**2 - 2.5*(Z[:,1:2]-2)**2], 1), 1)


def U9(Z):
    z1 = Z[:,0]
    z2 = Z[:,1]
    w1 = torch.sin(z1*0.5*math.pi)
    w3 = 2.5*F.sigmoid((z1-2)/0.3)
    return logsumexp(torch.cat(
        [(logsumexp(torch.cat(
                  [(-0.5*((z2-w1)/0.4)**2)[:,None],
                   (-0.5*((z2-w1+w3)/0.35)**2)[:,None]], 1), 1) - 
                    0.05 * ((z1)**2+(z2)**2))[:,None] ,
         2*U3(Z*1.5-2)[:,None]], 1), 1) -1
   


if __name__ == '__main__':
    # visualizing the energy function
    
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ef = U9
    
    mm, MM = -5, 5
    
    n = 300
    # plot figure
    fig = plt.figure(figsize=(10,10))
    
    for j in range(1,9+1):
        ax = fig.add_subplot(3,3,j)
        x = np.linspace(mm,MM,n)
        y = np.linspace(mm,MM,n)
        xx,yy = np.meshgrid(x,y)
        X = np.concatenate((xx.reshape(n**2,1),yy.reshape(n**2,1)),1)
        X = X.astype('float32')
        X = Variable(torch.from_numpy(X))
        Z = eval('U{}(X)'.format(j)).data.numpy().reshape(n,n)
        #plt.pcolormesh(xx,yy,np.exp(Z), cmap='RdBu_r')#), norm=colors.NoNorm())
        sns.heatmap(np.exp(Z)[::-1], ax=ax, cmap="YlGnBu", cbar=False) #YlGnBu
        plt.axis('off')
        #plt.xlim((mm,MM))
        #plt.ylim((mm,MM))
    
    plt.tight_layout()
    #plt.savefig('targets')





