#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/4
# Revised By Hongyu Lin on 2018/10/11

import torch
import torch.nn as nn
from torch.autograd import Variable
from BilinearAttention2DLayer import BilinearAttention2DLayer 


class AttentionLayer(nn.Module):
    def __init__(self, query_hidden_size, ctx_hidden_size):

        super(AttentionLayer, self).__init__()
        
        self.query_hidden_size = query_hidden_size
        self.ctx_hidden_size = ctx_hidden_size
        self.attention_func = BilinearAttention2DLayer(query_hidden_size,ctx_hidden_size)
        self.MAX_NEG = - 999999

    def forward(self,query,ctx,mask = None):
        '''
        Parameter:
        query           : [B,query_hidden_size]
        ctx             : [B,T,ctx_hidden_size]
        mask            : [B,T]
        
        Return:
        features     :[B,ctx_hidden_size], the attention averaged feature vector of context
        '''
        B, query_hidden_size = query.shape
        query = query.view(B,1,query_hidden_size)
        atten_val = self.attention_func(query,ctx).squeeze(dim=1)  #[B,T]
        if mask is not None:
            atten_val = atten_val + (1-mask) * self.MAX_NEG
        probs = torch.nn.functional.softmax(atten_val,dim=1)    #[B,T]
        features = torch.einsum("btd,bt->bd",(ctx,probs))   #[B,ctx_hidden_size]
        
        return features

if __name__ =="__main__":
    torch.cuda.set_device(4)
    device = torch.device("cuda:4")
    
    '''
    a = [[[1.0,2,3],[4,5,6],[7,8,9]],
         [[7,8,9],[10,11,12],[7,8,9]]]
    b = [[[1.0,2],[3,4]],
         [[5,6],[7,8]]]
    a = torch.tensor(a)
    b = torch.tensor(b)
    layer = BilinearAttention2DLayer(3,2)
    print layer(a,b)
    '''