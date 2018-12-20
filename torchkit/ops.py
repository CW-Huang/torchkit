#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:29:04 2018

@author: chin-weihuang
"""




from flows import BaseFlow
from nn import SequentialFlow

def mollify(flows, mm=0.0):
    if isinstance(flows, BaseFlow):
        if hasattr(flows, 'mollify'):
            flows.mollify = mm
    
    elif isinstance(flows, SequentialFlow):
        for flow in flows:
            mollify(flow, mm)
                
    
    