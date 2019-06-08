#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Roger on 2017/9/4
# Revised By Hongyu Lin on 2018/10/11

import torch
import torch.nn as nn
from torch.autograd import Variable


class TimeConvLayer(nn.Module):
    def __init__(self, input_size, hidden_size, win_size, bias=True):
        '''
        Parameter:
        window_size     : The window size to consider as context, note that the kernel size is [win_size*2+1,input_size]
        '''

        super(TimeConvLayer, self).__init__()
        # Define Parameter
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.win_size = win_size
        self.bias = bias

        # Define Layer
        # (N, Cin, Hin, Win)
        # In NLP, Hin is max sentence length, Win is Word Embedding Size
        #print padding
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=hidden_size,
                                    kernel_size=(self.win_size *2 +1, self.input_size),
                                    # window_size padding for length
                                    # zero padding for word dim
                                    padding= (self.win_size, 0),
                                    bias=bias)

    def forward(self, inputs):
        """
        :param inputs: batch x len x input_size
        :return:
                if padding is False:
                    batch x len - window_size + 1 x hidden_size
                if padding is True
                    batch x len + window_size - 1 x hidden_size
        """
        # (batch x len x input_size) -> (batch x 1 x len x input_size)
        inputs = torch.unsqueeze(inputs, 1)
        # (batch x 1 x len x input_size) -> (batch x hidden_size x new_len x 1)
        _temp = self.conv_layer(inputs)
        # (batch x hidden_size x new_len x 1)
        # -> (batch x hidden_size x new_len)
        # -> (batch x new_len x hidden_size)
        _temp.squeeze_(3)
        return torch.transpose(_temp, 1, 2)

if __name__ =="__main__":
    torch.cuda.set_device(4)
    device = torch.device("cuda:4")
    #device = torch.device("cpu")
    batch_size = 3
    max_length = 4
    hidden_size = 5
    n_layers =1

    # container
    batch_in = torch.zeros((batch_size, max_length, 1)).to(device = device)
    #print batch_in

    vec_1 = torch.FloatTensor([[1, 2, 3,4]]).t().to(device = device)
    vec_2 = torch.FloatTensor([[1, 2, 0,0]]).t().to(device = device)
    vec_3 = torch.FloatTensor([[1, 0, 0,0]]).t().to(device = device)

    batch_in[0] = vec_1
    batch_in[1] = vec_2
    batch_in[2] = vec_3

    seq_lengths = torch.tensor([3,1,2]).to(device = device)

    cnn = TimeConvLayer(1,10,1).to(device = device)
    
    batch_in = Variable(batch_in).to(device = device)
    print batch_in.shape
    output = cnn(batch_in)
    print output
