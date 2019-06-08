#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/4
# Revised By Hongyu Lin on 2018/10/11

import torch
import torch.nn as nn
from torch.autograd import Variable


class BilinearAttention2DLayer(nn.Module):
    def __init__(self, x_hidden_size, y_hidden_size, bias = False):

        super(BilinearAttention2DLayer, self).__init__()
        
        self.x_hidden_size = x_hidden_size
        self.y_hidden_size = y_hidden_size
        self.linear = torch.nn.Linear(self.x_hidden_size,y_hidden_size,bias= bias)

    def forward(self, x,y):
        '''
        Parameter:
        x               : [B,T_x,d1]
        y               : [B,T_y,d2]
        
        Return:
        xWy     :[B,T_x,T_y], the attention score of each [B,T_x] to [B,T_y]
        '''
        xW = self.linear(x) #[B,T_x,y_hidden_dim]
        xWy = torch.matmul(xW,y.transpose(1,2))    #[B,T_x,T_y]

        return xWy

if __name__ =="__main__":
    torch.cuda.set_device(4)
    device = torch.device("cuda:4")
    
    a = [[[1.0,2,3],[4,5,6],[7,8,9]],
         [[7,8,9],[10,11,12],[7,8,9]]]
    b = [[[1.0,2],[3,4]],
         [[5,6],[7,8]]]
    a = torch.tensor(a)
    b = torch.tensor(b)
    layer = BilinearAttention2DLayer(3,2)
    print layer(a,b)
