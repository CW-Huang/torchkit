#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:07:44 2018

@author: chin-weihuang
"""


import os

def getpaths(save_folder='default'):
    droot = os.environ['DATASETS']
    sroot = os.environ['SAVEPATH']
    spath = sroot+'/'+save_folder
    create(spath)
    return droot, sroot, spath

def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


