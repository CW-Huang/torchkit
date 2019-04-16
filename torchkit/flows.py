#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:02:58 2017

@author: Chin-Wei
"""

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import nn as nn_
from nn import log 
from torch.autograd import Variable
import iaf_modules 
import utils
import numpy as np


sum_from_one = nn_.sum_from_one


class BaseFlow(Module):
    
    
    def sample(self, n=1, context=None, **kwargs):
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

        
        if type(dim) is int:
            dim_ = dim
        else:
            dim_ = np.prod(dim)
        
        self.mean = oper(context_dim, dim_)
        self.lstd = oper(context_dim, dim_)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        if isinstance(self.mean, nn_.ResLinear):
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
        elif isinstance(self.mean, nn.Linear):
            self.mean.weight.data.uniform_(-0.001, 0.001)
            self.mean.bias.data.uniform_(-0.001, 0.001)
            self.lstd.weight.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1-nn_.delta)-1) * 0.5
                self.lstd.bias.data.uniform_(inv-0.001, inv+0.001)
            else:
                self.lstd.bias.data.uniform_(-0.001, 0.001)


    def forward(self, inputs):
        x, logdet, context = inputs
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)
        
        if type(self.dim) is int:
            x_ = mean + std * x
        else:
            size = x.size()
            x_ = mean.view(size) + std.view(size) * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context   


class BlockAffineFlow(Module):
    # NICE, volume preserving
    # x2' = x2 + nonLinfunc(x1)
    
    def __init__(self, dim1, dim2, context_dim, hid_dim, activation=nn.ELU()):
        super(BlockAffineFlow, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.actv = activation

        
        self.hid = nn_.WNBilinear(dim1, context_dim, hid_dim)
        self.shift = nn_.WNBilinear(hid_dim, context_dim, dim2)
        
    def forward(self,inputs):
        x, logdet, context = inputs
        x1, x2 = x
        
        hid = self.actv(self.hid(x1, context))
        shift = self.shift(hid, context)
        
        x2_ = x2 + shift

        return (x1, x2_), 0, context
    
    
class IAF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.sigmoid, fixed_order=False):
        super(IAF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 2, 
                    activation, fixed_order)
            self.reset_parameters()
        
        
    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1-nn_.delta)-1) 
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(inv,inv)
        elif self.realify == nn_.sigmoid:
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(2.0,2.0)
        
        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, iaf_modules.cMADE):
            mean = out[:,:,0]
            lstd = out[:,:,1]
            
        std = self.realify(lstd)
        
        if self.realify == nn_.softplus:
            x_ = mean + std * x
        elif self.realify == nn_.sigmoid:
            x_ = (-std+1.0) * mean + std * x
        elif self.realify == nn_.sigmoid2:
            x_ = (-std+2.0) * mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context


class IAF_VP(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), fixed_order=True):
        super(IAF_VP, self).__init__()
        
        self.dim = dim
        self.context_dim = context_dim
        
        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 1, 
                    activation, fixed_order)
            self.reset_parameters()
        
        
    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)

        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        mean = out[:,:,0]
        x_ = mean + x
        return x_, logdet, context



class IAF_DSF(BaseFlow):
    
    mollify=0.0
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DSF, self).__init__()
        
        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers
        
        

        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 
                    num_ds_multiplier*(hid_dim//dim)*num_ds_layers, 
                    activation, fixed_order)
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim//dim)*num_ds_layers, 
                3*num_ds_layers*num_ds_dim, 1)
            self.reset_parameters()
        
        
        self.sf = SigmoidFlow(num_ds_dim)
        
        
        
    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)
        
        inv = np.log(np.exp(1-nn_.delta)-1) 
        for l in range(self.num_ds_layers):
            nc = self.num_ds_dim
            nparams = nc * 3
            s = l*nparams
            self.out_to_dsparams.bias.data[s:s+nc].uniform_(inv,inv)
        
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, iaf_modules.cMADE):
            out = out.permute(0,2,1)
            dsparams = self.out_to_dsparams(out).permute(0,2,1)
            nparams = self.num_ds_dim*3
      
        mollify = self.mollify
        h = x.view(x.size(0), -1)
        for i in range(self.num_ds_layers):
            params = dsparams[:,:,i*nparams:(i+1)*nparams]
            h, logdet = self.sf(h, logdet, params, mollify)
       
        return h, logdet, context




class SigmoidFlow(BaseFlow):
    
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        
        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,dim=2)
        
    def forward(self, x, logdet, dsparams, mollify=0.0, delta=nn_.delta):
        
        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:,:,0*ndim:1*ndim])
        b_ = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(dsparams[:,:,2*ndim:3*ndim])
        
        a = a_ * (1-mollify) + 1.0 * mollify
        b = b_ * (1-mollify) + 0.0 * mollify
        
        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1-delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj = F.log_softmax(dsparams[:,:,2*ndim:3*ndim], dim=2) + \
            nn_.logsigmoid(pre_sigm) + \
            nn_.logsigmoid(-pre_sigm) + log(a)

        logj = utils.log_sum_exp(logj,2).sum(2)
        logdet_ = logj + np.log(1-delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet
        
        
        return xnew, logdet
        
        

class IAF_DDSF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), fixed_order=False,
                 num_ds_dim=4, num_ds_layers=1, num_ds_multiplier=3):
        super(IAF_DDSF, self).__init__()
        
        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

        if type(dim) is int:
            self.mdl = iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 
                    num_ds_multiplier*(hid_dim/dim)*num_ds_layers, 
                    activation, fixed_order)
        
        num_dsparams = 0
        for i in range(num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = num_ds_dim
            if i == num_ds_layers-1:
                out_dim = 1
            else:
                out_dim = num_ds_dim
          
            u_dim = in_dim
            w_dim = num_ds_dim
            a_dim = b_dim = num_ds_dim
            num_dsparams += u_dim + w_dim + a_dim + b_dim
            
            self.add_module('sf{}'.format(i),
                            DenseSigmoidFlow(in_dim,
                                             num_ds_dim,
                                             out_dim))
        if type(dim) is int:
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim/dim)*num_ds_layers, 
                num_dsparams, 1)
        else:
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier*(hid_dim/dim[0])*num_ds_layers, 
                num_dsparams, 1)
        
        
        self.reset_parameters()
        

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)
        
            
    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        out = out.permute(0,2,1)
        dsparams = self.out_to_dsparams(out).permute(0,2,1)
        
        
        start = 0

        h = x.view(x.size(0),-1)[:,:,None]
        n = x.size(0)
        dim = self.dim if type(self.dim) is int else self.dim[0]
        lgd = Variable(torch.from_numpy(
            np.zeros((n, dim, 1, 1)).astype('float32')))
        if self.out_to_dsparams.weight.is_cuda:
            lgd = lgd.cuda()
        for i in range(self.num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = self.num_ds_dim
            if i == self.num_ds_layers-1:
                out_dim = 1
            else:
                out_dim = self.num_ds_dim
            
            u_dim = in_dim
            w_dim = self.num_ds_dim
            a_dim = b_dim = self.num_ds_dim
            end = start + u_dim + w_dim + a_dim + b_dim
            
            params = dsparams[:,:,start:end]
            h, lgd = getattr(self,'sf{}'.format(i))(h, lgd, params)
            start = end
        
        assert out_dim == 1, 'last dsf out dim should be 1'
        return h[:,:,0], lgd[:,:,0,0].sum(1) + logdet, context
            
            

class DenseSigmoidFlow(BaseFlow):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x,dim=3)
        self.act_u = lambda x: nn_.softmax(x,dim=3)
        
        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)
        
        
    def forward(self, x, logdet, dsparams):
        inv = np.log(np.exp(1-nn_.delta)-1) 
        ndim = self.hidden_dim
        pre_u = self.u_[None,None,:,:]+dsparams[:,:,-self.in_dim:][:,:,None,:]
        pre_w = self.w_[None,None,:,:]+dsparams[:,:,2*ndim:3*ndim][:,:,None,:]
        a = self.act_a(dsparams[:,:,0*ndim:1*ndim]+inv)
        b = self.act_b(dsparams[:,:,1*ndim:2*ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)
        
        
        pre_sigm = torch.sum(u * a[:,:,:,None] * x[:,:,None,:], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm[:,:,None,:], dim=3)
        x_pre_clipped = x_pre * (1-nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1-x_pre_clipped)
        xnew = x_
        
        logj = F.log_softmax(pre_w, dim=3) + \
            nn_.logsigmoid(pre_sigm[:,:,None,:]) + \
            nn_.logsigmoid(-pre_sigm[:,:,None,:]) + log(a[:,:,None,:])
        # n, d, d2, dh
        
        logj = logj[:,:,:,:,None] + F.log_softmax(pre_u, dim=3)[:,:,None,:,:]
        # n, d, d2, dh, d1
        
        logj = utils.log_sum_exp(logj,3).sum(3)
        # n, d, d2, d1
        
        logdet_ = logj + np.log(1-nn_.delta) - \
            (log(x_pre_clipped) + log(-x_pre_clipped+1))[:,:,:,None]
        
        
        logdet = utils.log_sum_exp(
            logdet_[:,:,:,:,None] + logdet[:,:,None,:,:], 3).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0
            
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
    


class Sigmoid(BaseFlow):
    
    def __init__(self):
        super(Sigmoid, self).__init__()
        
    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise(Exception('inputs length not correct'))
        
        output = F.sigmoid(input)
        logdet += sum_from_one(- F.softplus(input) - F.softplus(-input))
        
        
        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise(Exception('inputs length not correct'))


class Logit(BaseFlow):

    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise(Exception('inputs length not correct'))

        output = log(input) - log(1-input)
        logdet -= sum_from_one(log(input) + log(-input+1))        
      
        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise(Exception('inputs length not correct'))



class Shift(BaseFlow):
    
    def __init__(self, b):
        self.b = b
        super(Shift, self).__init__()
        
    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise(Exception('inputs length not correct'))
        
        output = input + self.b
        
        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise(Exception('inputs length not correct'))
        
        

class Scale(BaseFlow):
    
    def __init__(self, g):
        self.g = g
        super(Scale, self).__init__()
        
    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise(Exception('inputs length not correct'))
        
        output = input * self.g
        logdet += np.log(np.abs(self.g)) * np.prod(input.size()[1:])
        
        
        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise(Exception('inputs length not correct'))
        
        
    
    

if __name__ == '__main__':
    
    
    
    
    inp = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,784).astype('float32')))
    con = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2,200).astype('float32')))
    lgd = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(2).astype('float32')))
    
    
    mdl = IAF(784, 1000, 200, 3)
    
    inputs = (inp, lgd, con)
    print(mdl(inputs)[0].size())
    
    
    mdl = IAF_DSF(784, 1000, 200, 3)
    print(mdl(inputs)[0].size())
    
    
    n = 2
    dim = 2
    num_ds_dim = 4
    num_in_dim = 1
    dsf = DenseSigmoidFlow(num_in_dim,num_ds_dim,num_ds_dim)
    
    mdl = IAF_DSF(784, 1000, 200, 3, num_ds_layers=2)
    print(mdl(inputs)[0].size())
    
    
