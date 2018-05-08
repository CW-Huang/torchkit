#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:56:47 2018

@author: chinwei
"""




import torch
import torch.nn as nn
from torch.autograd import Variable


from nn import WNlinear, CWNlinear, softplus, sigmoid
from utils import log_normal
import numpy as np

zero = Variable(torch.zeros(1,1))
one = Variable(torch.ones(1,1))
inv = np.log(np.exp(1)-1)


class Transition(nn.Module):
    
    def __init__(self, dim, h, oper=WNlinear, 
                 gate=True, actv=nn.ReLU(), cdim=0):
        super(Transition, self).__init__()
        
        if cdim:
            self.hidn = CWNlinear(dim, h, cdim)
        else:
            self.hidn = oper(dim,h)
        self.actv = actv
        self.mean = oper(h,dim)
        self.lstd = oper(h,dim)
        if gate:
            self.gate = oper(h,dim)
        self.ifgate = gate
        self.reset_params()
        
    def reset_params(self):
        self.lstd.bias.data.zero_().add_(inv)
        if isinstance(self.lstd, WNlinear):
            self.lstd.scale.data.uniform_(-0.001,0.001)
        elif isinstance(self.lstd, nn.Linear):
            self.lstd.weight.data.uniform_(-0.001,0.001)
        if self.ifgate:
            self.gate.bias.data.zero_().add_(-1.0)
        
    def forward(self, z, z_targ=None, context=None):
        
        ep = Variable(torch.zeros(z.size()).normal_()) 
        if context is not None:
            h = self.actv(self.hidn(z, context))
        else:
            h = self.actv(self.hidn(z))
        if self.ifgate:
            gate = sigmoid(self.gate(h)) 
            mean = gate*(self.mean(h)) + (1-gate)*z
        else:
            mean = self.mean(h)
        lstd = self.lstd(h)
        std = softplus(lstd)
        z_ = mean + ep * std
        if z_targ is None:
            return z_, log_normal(z_, mean, torch.log(std) * 2).sum(1)
        else:
            return z_, log_normal(z_targ, mean, torch.log(std) * 2).sum(1)
    




class HVI(nn.Module):
    
    def __init__(self, p, h, T, fT, f0=None):
        super(HVI, self).__init__()
        
        self.p = p
        self.T = T
        self.fT = fT
        if f0 is None:
            self.f0 = lambda z: log_normal(z, zero, zero).sum(1)
        else:
            self.f0 = f0
    
    def forward(self, n=64, sigma=1):
        raise NotImplementedError
    
    def samples(self, n=64, sigma=1):
        raise NotImplementedError
    
    def sample(self, n=64, T=None, sigma=1, context=None):
        T = self.T if T is None else T
        z = 0
        for t,z_ in enumerate(self.samples(n, sigma, context)):
            if t <= T:
                z = z_
        return z




class AHVI_MultipleTransition(HVI):
    """
    annealed hierarchical variational inference with multiple transitions
    """
    
    def __init__(self, p, h, T, fT, f0=None, 
                 oper=WNlinear,gate=True,actv=nn.ReLU(),cdim=0):
        super(AHVI_MultipleTransition, self).__init__(p, h, T, fT, f0)
        
        self.fchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(T)])
        self.bchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(T)])
        
    
    def forward(self, n=64, sigma=1, context=None):
        
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        ll = Variable(torch.zeros(n))
        for j in range(self.T):
            ftrans = self.fchain[j]
            btrans = self.bchain[j]
            z_t, qj = ftrans(z_, None, context)
            z_r, rj = btrans(z_t, z_, context)
            beta = (j+1)/float(self.T)
            ll = ll + self.fT(z_t) * beta + self.f0(z_t) * (1-beta) + (rj-qj)
            z_ = Variable(z_t.data)

        return ll
    
    def samples(self, n=64, sigma=1, context=None):
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        for trans in self.fchain:
            z_, _ = trans(z_, None, context)
            yield z_.data.cpu().numpy()
    


class AHVI_SingleTransition(HVI):
    """
    annealed hierarchical variational inference with single transition
    """
    
    def __init__(self, p, h, T, fT, f0=None,
                 oper=WNlinear,gate=True,actv=nn.ReLU(),cdim=0):
        super(AHVI_SingleTransition, self).__init__(p, h, T, fT, f0)
        
        self.fchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(1)])
        self.bchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(1)])
        
    
    def forward(self, n=64, sigma=1, sample=False, context=None):
        
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        ll = Variable(torch.zeros(n))
        if sample:
            t_ = np.random.randint(1,self.T+1)
        else:
            t_ = 0
        for j in range(self.T):
            ftrans = self.fchain[0]
            btrans = self.bchain[0]
            z_t, qj = ftrans(z_, None, context)
            z_r, rj = btrans(z_t, z_, context)
            z_ = Variable(z_t.data)
            if j+1 != t_ and sample:
                continue
            
            beta = (j+1)/float(self.T)
            ll = ll + self.fT(z_t) * beta + self.f0(z_t) * (1-beta) + (rj-qj)
            
        return ll
    
    def samples(self, n=64, sigma=1, context=None):
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        for j in range(self.T):
            trans = self.fchain[0]
            z_, _ = trans(z_, None, context)
            yield z_.data.cpu().numpy()
    
    
class HVI_MultipleTransition(HVI):
    """
    hierarchical variational inference with multiple transitions
    """
    
    def __init__(self, p, h, T, fT, f0=None, 
                 oper=WNlinear,gate=True,actv=nn.ReLU(),cdim=0):
        super(HVI_MultipleTransition, self).__init__(p, h, T, fT, f0)
        
        self.fchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(T)])
        self.bchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(T)])
        
    
    def forward(self, n=64, sigma=1, context=None):
        
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        ll = Variable(torch.zeros(n))
        for j in range(self.T):
            ftrans = self.fchain[j]
            btrans = self.bchain[j]
            z_t, qj = ftrans(z_, None, context)
            z_r, rj = btrans(z_t, z_, context)
            ll = ll + (rj-qj)
            z_ = z_t
        
        ll = ll + self.fT(z_t)
        
        return ll
    
    def samples(self, n=64, sigma=1, context=None):
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        for trans in self.fchain:
            z_, _ = trans(z_, None, context)
            yield z_.data.cpu().numpy()
    


class HVI_SingleTransition(HVI):
    """
    annealed hierarchical variational inference with single transition
    """
    
    def __init__(self, p, h, T, fT, f0=None,
                 oper=WNlinear,gate=True,actv=nn.ReLU(),cdim=0):
        super(HVI_SingleTransition, self).__init__(p, h, T, fT, f0)
        
        self.fchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(1)])
        self.bchain = nn.ModuleList(
            [Transition(p,h,oper,gate,actv,cdim) for i in range(1)])
        
    
    def forward(self, n=64, sigma=1, context=None):
        
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        ll = Variable(torch.zeros(n))
        for j in range(self.T):
            ftrans = self.fchain[0]
            btrans = self.bchain[0]
            z_t, qj = ftrans(z_, None, context)
            z_r, rj = btrans(z_t, z_, context)
            ll = ll + (rj-qj)
            z_ = z_t
            
        ll = ll + self.fT(z_t)
            
        return ll
    
    def samples(self, n=64, sigma=1, context=None):
        ep = Variable(torch.zeros((n, self.p)).normal_()) * sigma
        z_ = ep
        for j in range(self.T):
            trans = self.fchain[0]
            z_, _ = trans(z_, None, context)
            yield z_.data.cpu().numpy()
    



if __name__ == '__main__':
    from toy_energy import U4 as ef
    import matplotlib.pyplot as plt
    from optim import Adam

    
    
    bs = 64
    ss = 0.001
    T = 10
    
    chain = HVI_MultipleTransition(2, 64, T, ef)
    optim = Adam(chain.parameters(), ss)
    
    for i in range(2000):
        optim.zero_grad()
        ll = chain.forward(bs)
        loss = -ll.mean()
        loss.backward()
        optim.step()
        if i%1000==0:
            print i, loss
    

    # plotting    
    mm, MM = -5, 5
    n = 400
    fig = plt.figure(figsize=(1.6*T,2))
    for t in range(1,T+1):
        beta = (t)/float(chain.T)
                
        ax = fig.add_subplot(2,T,t)
        x = np.linspace(mm,MM,n)
        y = np.linspace(mm,MM,n)
        xx,yy = np.meshgrid(x,y)
        X = np.concatenate((xx.reshape(n**2,1),yy.reshape(n**2,1)),1)
        X = X.astype('float32')
        X = Variable(torch.from_numpy(X))
        Z = (ef(X) * beta + chain.f0(X) * (1-beta)).data.numpy().reshape(n,n)
        ax.pcolormesh(xx,yy,np.exp(Z), cmap='RdBu_r')
        ax.axis('off')
        plt.xlim((mm,MM))
        plt.ylim((mm,MM))
        
        ax = fig.add_subplot(2,T,t+T)
        data = chain.sample(n**2, t)
        XX = data[:,0]
        YY = data[:,1]
        plot = ax.hist2d(
            XX,YY,200,range=np.array([(-10, 10), (-10, 10)]), cmap='RdBu_r')
        plt.xlim((mm,MM))
        plt.ylim((mm,MM))
        plt.axis('off')
        
    #plt.tight_layout()
