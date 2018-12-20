#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:56:33 2018

@author: chinwei
"""
import torch
from torch.optim.optimizer import Optimizer
import math


class Adam(Optimizer):
    """
    Adam with polyak averaging
    
    to access the expentially decayed average of parameters, do optim.swap().
    after accessing the parameters, do optim.swap() again to recover the
    current value
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, polyak=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        polyak=polyak)
        assert polyak >= 0.0 and polyak <= 1.0, 'invalid polyak decay term'
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        polyak = self.defaults['polyak']
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse ' \
                                       'gradients, please consider ' \
                                       'SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Exponential moving average of param
                    state['exp_avg_param'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. 
                        # of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment 
                    # running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / \
                                          bias_correction1
                
                p.data.addcdiv_(-step_size, exp_avg, denom)
                state['exp_avg_param'] = \
                    ( polyak * state['exp_avg_param'] + \
                      (1-polyak) * p.data )
                        
        return loss
    
    def swap(self):
        """
        swapping the running average of params and the current params
        
        for saving parameters using polyak averaging
        
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                new = p.data
                p.data = state['exp_avg_param']
                state['exp_avg_param'] = new
                
                
                