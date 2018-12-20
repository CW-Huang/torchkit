#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:00:37 2018

@author: chinwei
"""

import torch
from nn import logit


class binarize(object):
    """ Dynamically binarize the image """


    def __call__(self, x):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        threshold = torch.zeros_like(x)
        threshold.uniform_()
        
        
        return torch.ge(x, threshold).float()


class realify(object):
    """ 
    - rescale [0,1] to [0,255]
    - add uniform(0,1) noise
    - rescale to [0+delta,1-delta]
    - pass through logit
    """

    def __init__(self, delta=0.01, noise=1.0):
        self.delta = delta
        self.noise = noise

    def __call__(self, x):

        x_ = x * 255.
        noise = torch.zeros_like(x).uniform_(0,self.noise)
        x_ += noise
        a, b = x_.min(), x_.max()
        x_ -= a
        x_ /= (b-a)
        x_ *= 1 - self.delta * 2
        x_ += self.delta
        
        return logit(x_)


class noisify(object):
    """ 
    - rescale [0,1] to [0,255]
    - add uniform(0,1) noise
    - rescale to [0,1]
    """

    def __init__(self, noise=1.0):
        self.noise = noise

    def __call__(self, x):

        x_ = x * 255.
        noise = torch.zeros_like(x).uniform_(0,self.noise)
        x_ += noise
        
        return x_/256.


class scaleshift(object):
    """ y = ax + b
    """

    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return x * self.a + self.b




class from_numpy(object):
    
    def __call__(self, x):
        return torch.from_numpy(x)





    
    
    
