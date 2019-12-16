# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:46:04 2019

@author: zhanghonglu_i
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
__all__ = ['smallModel']

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        #nn.init.uniform(m.weight.data)
        #nn.init.normal(m.weight.data)
        #nn.init.xavier_normal(m.weight.data)
        nn.init.constant(m.bias, 0.01)


def conv_bn(inp, oup, stride,pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, pad, bias=True),
        nn.BatchNorm2d(oup,momentum=0.01),
        nn.ReLU(inplace=True)
    )


def conv(inp, oup, stride,pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, pad, bias=True),
        #nn.BatchNorm2d(oup,momentum=0.01),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
        nn.BatchNorm2d(inp,momentum=0.01),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        nn.BatchNorm2d(oup,momentum=0.01),
        nn.ReLU(inplace=True),
    )

def conv_dw_no_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
        nn.BatchNorm2d(inp,momentum=0.5),
        nn.ReLU(inplace=True),
        #nn.PReLU(),
	#nn.Tanh(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
        nn.BatchNorm2d(oup,momentum=0.5),
        nn.ReLU(inplace=True),
        #nn.PReLU(),
	#nn.Tanh(),
    )
    

class Net_no_bn_attrs_64x64(nn.Module):

    def __init__(self, embeding_size=256, use_fp16=False):
        super(Net_no_bn_attrs_64x64, self).__init__()
        self.use_fp16 = use_fp16

        self.model = nn.Sequential(
	    conv_dw_no_bn(3,32,2),
            conv_dw_no_bn(32,32,1),
            conv_dw_no_bn(32,64,2),#
            conv_dw_no_bn(64,64,1),#
            conv_dw_no_bn(64,96,2),#
            conv_dw_no_bn(96,96,1),#
            conv_dw_no_bn(96,128, 2),#4
            conv_dw_no_bn(128, 128, 2),#2
            nn.AvgPool2d(2, ceil_mode=True),
            #nn.MaxPool2d(2,ceil_mode=True),
            nn.Dropout(p=0.5),
        )
        
        self.fc1 = nn.Linear(128, embeding_size)
        self.apply(weights_init)

    def forward(self,x):
        x = self.model(x)
        x = x.view(-1,128)
        x = self.fc1(x)
        return x
    
def smallModel(embeding_size=256):
    model = Net_no_bn_attrs_64x64(embeding_size=embeding_size)
    return model

