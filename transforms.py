#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 00:00:37 2018

@author: chinwei
"""

import torch


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

class from_numpy(object):
    
    def __call__(self, x):
        return torch.from_numpy(x)