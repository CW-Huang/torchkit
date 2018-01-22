#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:02:58 2017

@author: Chin-Wei
"""

import torch
import torch.nn as nn
from torch.nn import Module
import nn as nn_
from torch.autograd import Variable
import iaf_modules 
import utils
import numpy as np

sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size())>2 else sum1(x)


log = lambda x: torch.log(x*1e2)-np.log(1e2)


class BaseFlow(Module):
    
    
    def sample(self, n=1, context=None):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim,]
        
        spl = Variable(torch.FloatTensor(n,*dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
            np.ones((n,self.context_dim)).astype('float32')))
            
        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()
        
        return self.forward((spl, lgd, context))
    
    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class LinearFlow(BaseFlow):
    
    def __init__(self, dim, context_dim, 
                 oper=nn_.ResLinear, realify=nn_.softplus):
        super(LinearFlow, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        self.mean = oper(context_dim, dim)
        self.lstd = oper(context_dim, dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.mean.dot_01.scale.data.uniform_(-0.001, 0.001)
        self.mean.dot_h1.scale.data.uniform_(-0.001, 0.001)
        self.mean.dot_01.bias.data.uniform_(-0.001, 0.001)
        self.mean.dot_h1.bias.data.uniform_(-0.001, 0.001)
        self.lstd.dot_01.scale.data.uniform_(-0.001, 0.001)
        self.lstd.dot_h1.scale.data.uniform_(-0.001, 0.001)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1-nn_.delta)-1) * 0.5
            self.lstd.dot_01.bias.data.uniform_(inv-0.001, inv+0.001)
            self.lstd.dot_h1.bias.data.uniform_(inv-0.001, inv+0.001)
        else:
            self.lstd.dot_01.bias.data.uniform_(-0.001, 0.001)
            self.lstd.dot_h1.bias.data.uniform_(-0.001, 0.001)


    def forward(self, inputs):
        x, logdet, context = inputs
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)
        
        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context   


class IAF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.softplus):
        super(IAF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        self.made = iaf_modules.cMADE(
                dim, hid_dim, context_dim, num_layers, 2, activation)
       
        self.reset_parameters()
        
    def reset_parameters(self):
        self.made.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.made.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.made.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.made.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1-nn_.delta)-1) 
            self.made.hidden_to_output.cbias.bias.data[1::2].uniform_(inv,inv)
        
        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.made((x, context))
        mean = out[:,:,0]
        lstd = out[:,:,1]
        std = self.realify(lstd)
        
        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context

 
class BlockAffineFlow(Module):
    # RealNVP

    def __init__(self, dim, context_dim, hid_dim, 
                 mask=0, realify=nn_.softplus):
        super(BlockAffineFlow, self).__init__()
        self.mask = mask
        self.dim = dim
        self.realify = realify
        self.gpu = True
        
        self.hid = nn_.WNBilinear(dim, context_dim, hid_dim)
        self.mean = nn_.ResLinear(hid_dim, dim)
        self.lstd = nn_.ResLinear(hid_dim, dim)

    def forward(self,inputs):
        x, logdet, context = inputs
        mask = Variable(torch.zeros(1, self.dim))
        if self.gpu:
            mask = mask.cuda()

        if self.mask:
            mask[:, self.dim/2:].data += 1
        else:
            mask[:, :self.dim/2].data += 1

        hid = self.hid(x*mask, context)
        mean = self.mean(hid)*(-mask+1) + self.mean.dot_h1.bias
        lstd = self.lstd(hid)*(-mask+1) + self.lstd.dot_h1.bias
        std = self.realify(lstd)

        x_ = mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context

    def cuda(self):
        self.gpu = True
        return super(cuda, self).cuda()


class IAF_DSF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.softplus,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DSF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        
        self.made = iaf_modules.cMADE(
                dim, hid_dim, context_dim, num_layers, 
                num_ds_multiplier*(hid_dim/dim)*num_ds_layers, activation)
        
        self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim/dim)*num_ds_layers, 
                3*num_ds_layers*num_ds_dim, 1)
        self.sf = SigmoidFlow()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)
        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.made((x, context))
        out = out.permute(0,2,1)
        dsparams = self.out_to_dsparams(out).permute(0,2,1)
        nparams = self.num_ds_dim*3
        
        h = x
        for i in range(self.num_ds_layers):
            params = dsparams[:,:,i*nparams:(i+1)*nparams]
            h, logdet = self.sf(h, logdet, params)
        
        return h, logdet, context




class SigmoidFlow(BaseFlow):
    
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        
        #self.act_a = lambda x: torch.exp(x) + nn_.delta
        self.act_a = lambda x: nn_.softplus(x)#torch.exp(x) + nn_.delta
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,dim=2)
        
    def forward(self, x, logdet, dsparams):
        
        ndim = self.num_ds_dim
        a = self.act_a(dsparams[:,:,0*ndim:1*ndim])
        b = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(dsparams[:,:,2*ndim:3*ndim])
        
        
        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj = log(w) + nn_.logsigmoid(pre_sigm) + \
            nn_.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj,2).sum(2)
        logdet_ = logj + np.log(1-nn_.delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet
            
            
        return xnew, logdet
        
        

class IAF_DDSF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.softplus,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DDSF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        
        self.made = iaf_modules.cMADE(
                dim, hid_dim, context_dim, num_layers, 
                num_ds_multiplier*(hid_dim/dim)*num_ds_layers, activation)
        
        self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim/dim)*num_ds_layers, 
                3*num_ds_layers*num_ds_dim, 1)
        self.sf = SigmoidFlow()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)
        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.made((x, context))
        out = out.permute(0,2,1)
        dsparams = self.out_to_dsparams(out).permute(0,2,1)
        nparams = self.num_ds_dim*3
        
        h = x
        for i in range(self.num_ds_layers):
            params = dsparams[:,:,i*nparams:(i+1)*nparams]
            h, logdet = self.sf(h, logdet, params)
        
        return h, logdet, context


class DenseSigmoidFlow(BaseFlow):
    
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        
        self.act_a = lambda x: torch.exp(x) + nn_.delta
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,dim=2)
        
    def forward(self, x, logdet, dsparams):
        
        ndim = self.num_ds_dim
        a = self.act_a(dsparams[:,:,0*ndim:1*ndim])
        b = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(dsparams[:,:,2*ndim:3*ndim])
        
        
        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj = log(w) + nn_.logsigmoid(pre_sigm) + \
            nn_.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj,2).sum(2)
        logdet_ = logj + np.log(1-nn_.delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet
            
            
        return xnew, logdet


class FlipFlow(BaseFlow):
    
    def __init__(self, dim):
        self.dim = dim
        super(FlipFlow, self).__init__()
        
    def forward(self, inputs):
        input, logdet, context = inputs
        
        dim = self.dim
        index = Variable(
                getattr(torch.arange(input.size(dim)-1, -1, -1), (
                        'cpu', 'cuda')[input.is_cuda])().long())
        
        output = torch.index_select(input, dim, index)
        
        return output, logdet, context
    

if __name__ == '__main__':
    
    
    
    
    inp = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,784).astype('float32')))
    con = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,200).astype('float32')))
    lgd = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2).astype('float32')))
    
    
    mdl = IAF(784, 1000, 200, 3)
    
    inputs = (inp, lgd, con)
    print mdl(inputs)[0].size()
    
    
    mdl = IAF_DSF(784, 1000, 200, 3)
    mdl(inputs)
    
    
    


