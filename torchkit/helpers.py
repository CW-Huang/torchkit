#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:07:44 2018

@author: chin-weihuang
"""


import os
import torch
import json


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


class Model(object):
    
    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, states):
        raise NotImplementedError

    
class Trainer(object):
    
        
    def train(self, *input):
        raise NotImplementedError

    def evaluate(self, *input):
        raise NotImplementedError
    
    def impatient(self):
        self.check()
        current_epoch = self.train_params['e']
        bestv_epoch = self.train_params['best_val_epoch']
        return current_epoch - bestv_epoch > self.patience
        
    def save(self, fn):
        self.check()
        torch.save(self.model.state_dict(), fn+'_model.pt')
        torch.save(self.model.optim.state_dict(), fn+'_optim.pt')
        with open(fn+'_args.txt','w') as out:
            out.write(json.dumps(self.args.__dict__,indent=4))
        with open(fn + '_train_params.txt', 'w') as out:
            out.write(json.dumps(self.train_params, indent=4))

    def load(self, fn):
        self.check()
        self.model.load_state_dict(torch.load(fn+'_model.pt'))
        self.model.optim.load_state_dict(torch.load(fn+'_optim.pt'))
        self.train_params = json.load(open(fn + '_train_params.txt', 'r'))
    
    def check(self):
        if not hasattr(self, 'model'):
            raise Exception('`Trainer` must have `Model`'\
                            ' as instance variable `self.model`')
        if not hasattr(self, 'args'):
            raise Exception('`Trainer` must have'\
                            ' instance variable `args`')
        if not hasattr(self, 'train_params'):
            raise Exception('`Trainer` must have'\
                            ' instance variable `train_params`')
    
class MultiOptim(object):
    
    def __init__(self, *optims):
        self.optims = optims
    
    def state_dict(self):
        return [o.state_dict() for o in self.optims]
    
    def load_state_dict(self, state_dicts):
        for s, o in zip(state_dicts, self.optims):
            o.load_state_dict(s)
    
    def step(self):
        for o in self.optims:
            o.step()
            

def logging(s, path=False, filename='log.txt', print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        assert path, 'path is not define. path: {}'.format(path)
        with open(os.path.join(path, filename), 'a+') as f_log:
            f_log.write(s + '\n')