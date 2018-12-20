#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:41:42 2018

@author: chin-weihuang
"""


import torch
from torch.autograd import backward




def reinforce(reward, log_prob, retain_graph=False, iib=None, idb=None):
    reward = reward.detach()
    if idb is not None:
        b = idb.detach()
    else:
        b = torch.zeros_like(reward)
    if iib is not None:
        c = iib.detach()
    else:
        c = torch.zeros(1)
    if idb is not None and iib is not None:
        ((reward - idb - iib)**2).mean().backward()
    if idb is not None and iib is None:
        ((reward - idb)**2).mean().backward()
    if idb is None and iib is not None:
        ((reward - iib)**2).mean().backward()
    
    r = reward-b-c
    backward(-(log_prob*r).mean(), retain_graph=retain_graph)
    





if __name__ == '__main__':
    
    from torchvision import datasets, transforms
    import transforms as transforms_
    import helpers
    import autoencoders as aes 
    from torch import optim, nn
    from itertools import chain 
    import utils
    import numpy as np
    
    nmc = 3
    lr1 = 0.0015
    lr2 = 0.0003
    batch_size = 20
    zdim = 200
    epoch = 10
    print_every = 50
    
    droot, sroot, spath = helpers.getpaths()
    helpers.create(droot, 'mnist')
    
    
    ds_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms_.binarize()])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(droot+'/mnist', download=True, train=True, 
        transform=ds_transforms), batch_size=batch_size, 
        shuffle=True)
    
    enc = aes.BinaryLinear(784, zdim)
    dec = aes.BinaryLinear(zdim, 784)
    prior = aes.BinaryPrior(zdim)
    iib = nn.parameter.Parameter(torch.zeros(1)-200)
    
    optim1 = optim.Adam(
        chain(dec.parameters(), 
              prior.parameters(), 
              [iib]),
        lr=lr1/float(nmc))
    optim2 = optim.Adam(
        chain(enc.parameters()),
        lr=lr2/float(nmc))
    zero = utils.varify(np.zeros(1).astype('float32'))
    
    def ELBO(x):
        z = enc.sample(x)
        px_z = dec.evaluate(z,x)
        qz_x = enc.evaluate(x,z)
        pz = prior.evaluate(z)
        elbo = px_z + pz - qz_x.detach()
        return elbo, qz_x
    
    def get_grad(x, multiply=1):
        n = x.size(0)
        x = x.repeat([multiply, 1])
        elbo, q = ELBO(x)
        reinforce(elbo, q, idb=None, iib=iib)
        iwlb = utils.log_mean_exp(elbo.view(multiply,n).permute(1,0),1)
        loss = (-iwlb).mean()
        loss.backward()
        return loss.data.cpu().numpy()
    
    # begin training
    count = 0
    for e in range(epoch):
        for x, _ in train_loader:
            optim1.zero_grad()
            optim2.zero_grad()
            x = x.view(-1,784) * 2.0 - 1.0
            loss = get_grad(x, nmc)
            optim1.step()
            optim2.step()
            count += 1
            if count % print_every == 0:
                print('[{}] {}'.format(e, loss))
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    