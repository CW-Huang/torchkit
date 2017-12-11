#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""



import math

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

# aliasing
N_ = None

class WNlinear(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(WNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.direction = Parameter(torch.Tensor(out_features, in_features))
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', N_)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        dir_ = self.direction
        direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:,N_])
        weight = self.scale[:,N_].mul(direction)
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


      
class WNBilinear(Module):
    
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(WNBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.direction = Parameter(torch.Tensor(
                out_features, in1_features, in2_features))
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', N_)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        dir_ = self.direction
        direction = dir_.div(dir_.pow(2).sum(1).sum(1).sqrt()[:,N_,N_])
        weight = self.scale[:,N_,N_].mul(direction)
        return F.bilinear(input1, input2, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in1_features=' + str(self.in1_features) \
            + ', in2_features=' + str(self.in2_features) \
            + ', out_features=' + str(self.out_features) + ')'


                                           
class _WNconvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_WNconvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        # weight – filters tensor (out_channels x in_channels/groups x kH x kW)
        if transposed:
            self.direction = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.scale = Parameter(torch.Tensor(in_channels))
        else:
            self.direction = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.scale = Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', N_)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is N_:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class WNconv2d(_WNconvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(WNconv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        dir_ = self.direction
        direction = dir_.div(
            dir_.pow(2).sum(1).sum(1).sum(1).sqrt()[:,N_,N_,N_])
        weight = self.scale[:,N_,N_,N_].mul(direction)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResConv2d(nn.Module):
    
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, 
            padding=0, dilation=1, groups=1, bias=True, activation=nn.ReLU(),
            oper=WNconv2d):
        super(ResConv2d, self).__init__()
        
        self.conv_0h = oper(
                in_channels, out_channels, kernel_size, stride, 
                padding, dilation, groups, bias)
        self.conv_h1 = oper(
                out_channels, out_channels, 3, 1, 1, 1, 1, True)
        self.conv_01 = oper(
                in_channels, out_channels, kernel_size, stride, 
                padding, dilation, groups, bias)
        
        self.activation = activation

    def forward(self, input):
        h = self.activation(self.conv_0h(input))
        out_nonlinear = self.conv_h1(h)
        out_skip = self.conv_01(input)
        return out_nonlinear + out_skip

class ResLinear(nn.Module):
    
    def __init__(
            self, in_features, out_features, bias=True, same_dim=False,
            activation=nn.ReLU(), oper=WNlinear):
        super(ResLinear, self).__init__()
        
        self.same_dim = same_dim
        
        self.dot_0h = oper(in_features, out_features, bias)
        self.dot_h1 = oper(out_features, out_features, bias)
        if not same_dim:
            self.dot_01 = oper(in_features, out_features, bias)
        
        self.activation = activation
        
    def forward(self, input):
        h = self.activation(self.dot_0h(input))
        out_nonlinear = self.dot_h1(h)
        out_skip = input if self.same_dim else self.dot_01(input)
        return out_nonlinear + out_skip
