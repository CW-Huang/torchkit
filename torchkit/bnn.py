#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:45:45 2018

@author: chinwei
"""

import numpy as np
import torch
import torch.nn as nn
import nn as nn_
import flows
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class FalseType(object):
    """ dummy variable """
    


class MergedSampler(nn.Module):
    
    def __init__(self):
        super(MergedSampler, self).__init__()
        self.init_flows = nn.ModuleList()#list()
        self.init_flow_type = None
        self.registerable = True 
        self.merged = False
        self.tot_dim = 0
        self.common_flowlayers = nn.Sequential()
        self.num_common_flowlayers = 0
        self.logdet = None
        self.spls = None
    
    def register_sampler(self, init_flow):
        assert self.registerable, Exception('no longer registerable')
        
        self.init_flows.append(init_flow)
        if self.init_flow_type is None:
            self.init_flow_type = type(init_flow)
        else:
            if not isinstance(init_flow, self.init_flow_type):
                self.init_flow_type = FalseType
        
        self.tot_dim += int(init_flow.dim)
        ind1 = self.tot_dim-init_flow.dim
        ind2 = self.tot_dim
        s = lambda: self.slicedsample(ind1, ind2)
        return s
    
    def merge_linear_flows(self):
        assert not self.merged, Exception('flows already merged')
        assert self.init_flow_type is flows.LinearFlow, Exception('not linear')
        
        if all(
            [type(initf.mean) is nn_.ResLinear and \
             type(initf.lstd) is nn_.ResLinear \
             for initf in self.init_flows]):
            cat_oper = None
            # TODO: implement
            pass
        elif all(
            [type(initf.mean) is nn_.WNlinear and \
             type(initf.lstd) is nn_.WNlinear \
             for initf in self.init_flows]):
            cat_oper = None
            # TODO: implement
            pass
        elif all(
            [type(initf.mean) is nn.Linear and \
             type(initf.lstd) is nn.Linear \
             for initf in self.init_flows]):

            mw = torch.cat([initf.mean.weight for initf in self.init_flows])
            sw = torch.cat([initf.lstd.weight for initf in self.init_flows])
            
            # assume all have bias
            mb = torch.cat([initf.mean.bias for initf in self.init_flows])
            sb = torch.cat([initf.lstd.bias for initf in self.init_flows])
            
            def cat_oper(inputs):
                x, logdet, context = inputs
                
                mean = F.linear(context, mw, mb)
                lstd = F.linear(context, sw, sb)
                
                # assume the same
                realify = self.init_flows[-1].realify
                std = realify(lstd)
                
                x_ = mean + std * x
                logdet_ = nn_.sum_from_one(torch.log(std)) + logdet

                return x_, logdet_, context
                
        self.merged = True
        
        return cat_oper
        
        
        
        
    def add_common_flowlayer(self, flow):
        """ arg of flow is dimension of random variable """
        self.common_flowlayers.add_module(
            str(self.num_common_flowlayers), flow(self.tot_dim))
        self.num_common_flowlayers += 1
        
    def sample0(self, n=1, context=None):
        #self.registerable = False
        if self.init_flow_type is flows.LinearFlow:
            if not self.merged:
                self.cat_oper = self.merge_linear_flows()
            
            
            spl = torch.autograd.Variable(
                torch.FloatTensor(n,self.tot_dim).normal_())
            lgd = torch.autograd.Variable(
                torch.from_numpy(np.random.rand(n).astype('float32')))
            if context is None:
                context = torch.autograd.Variable(torch.from_numpy(
                np.ones((n,1)).astype('float32')))
                
            if hasattr(self, 'gpu'):
                if self.gpu:
                    spl = spl.cuda()
                    lgd = lgd.cuda()
                    context = context.gpu()
        

            return self.cat_oper((spl, lgd, context))
                
        else:
            # TODO: sequential sampling
            pass
        
    def sample(self, n=1, context=None):
        self.spls, self.logdet, _ = \
            self.common_flowlayers(self.sample0(n, context))
    
    def slicedsample(self, ind1, ind2):
        #print ind1, ind2
        return self.spls[:,ind1:ind2]
        


class Param(nn.Module):
    
    def __init__(self, param):
        super(Param, self).__init__()
        self.param = Parameter(torch.from_numpy(param))
        

class SharedParams(nn.Module):
    
    def __init__(self):
        super(SharedParams, self).__init__()
        self.sharedparams = nn.ModuleList()#list()
        self.counter = 0
    
    def register_param(self, param):
        self.sharedparams.append(Param(param))
        self.counter += 1
        ind = self.counter - 1
        return lambda: self.sharedparams[ind].param
        

class Parameters(nn.Module):
    
    def __init__(self):
        super(Parameters, self).__init__()
        
        self.merged_sampler = MergedSampler()
        self.shared_params = SharedParams()
    
    def add_params_0(self, size, dimc=1):
        # stochastic
        dim = np.prod(size)
        spler = self.merged_sampler.register_sampler(
            flows.LinearFlow(dim, dimc, oper=nn.Linear))
        print 'registering stochastic parameter of size {}'.format(size)
        return lambda: spler().contiguous().view(-1,*size) 
    
    def add_params_1(self, param):
        # deterministic
        return self.shared_params.register_param(param)
        
        
    def sample(self, n=1, context=None):
        self.merged_sampler.sample(n, context)
        
        
    def __call__(self, params, dimc=1, type=0):
        if type == 0:
            return self.add_params_0(params, dimc)
        elif type == 1:
            return self.add_params_1(params)

###############################################################################
params = Parameters() #########################################################
###############################################################################

class sLinear(nn.Module):
    
    def __init__(self, in_features, out_features, bias=0, dimc=1):
        super(sLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = params((out_features,in_features), dimc)
        if bias == 0:
            # stochastic
            self.bias = params((out_features,), dimc, 0)
        elif bias == 1:
            # deterministic
            self.bias = params(
                np.zeros((1, out_features,), dtype='float32'), None, 1)

    def forward(self, input, conditional=False):
        if conditional:
            pass # TODO: implement
        else:
            weight = self.weight()[0]
            bias = self.bias()[0]
            return F.linear(input, weight, bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'



class sWNLinear(nn.Module):
    
    def __init__(self, in_features, out_features, dimc=1):
        super(sWNLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.scale = params((1, out_features,), dimc)

        self.direction = params(
            np.random.randn(out_features, in_features).astype('float32'), None, 1)
        self.bias = params(
            np.zeros((1, out_features,), dtype='float32'), None, 1)

    def forward(self, input, conditional=False):
        if conditional:
            pass # TODO: implement
        else:
            dir_ = self.direction()
            weight = dir_.div(dir_.pow(2).sum(1).sqrt()[:,None])
            bias = self.bias()[0]
            scale = self.scale()[0]
            return scale * F.linear(input, weight, bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'



