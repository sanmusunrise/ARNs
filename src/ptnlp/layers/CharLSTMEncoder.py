import sys 
sys.path.append("../../") 
from ptnlp.utils import logger
from ptnlp.layers.EmbeddingLayer import EmbeddingLayer
from ptnlp.layers.LSTMLayer import LSTMLayer

import torch
import torch.nn as nn

class CharLSTMEncoder(nn.Module):
    
    def __init__(self,char_embedding_dim,word_encoding_dim,num_vocab):
        
        super(CharLSTMEncoder, self).__init__()
        
        self.char_embedding_dim = char_embedding_dim
        self.word_encoding_dim = word_encoding_dim
        self.num_vocab = num_vocab
        
        self.char_embedding = EmbeddingLayer(dim = self.char_embedding_dim, trainable = True)
        self.char_embedding.initialize_with_random(self.num_vocab)
        
        self.rnn_layer = LSTMLayer(D_in = self.char_embedding_dim, 
                                   D_out = self.word_encoding_dim,
                                   n_layers = 1,
                                   dropout = 0,
                                   bidirectional = True)
                                   
    
    def forward(self,chars,char_seq_len):
        '''
        Parameters:
        chars           : An int tensor with shape [B,max_sent_len,max_word_len] indicates the char ids.
        char_seq_len    : An int tensor with shape [B,max_sent_len] indicates the length of each word,
                          Please note that the length of padding words should be set to **at least 1** as pytorch rnn requires.
        '''
        B,max_sent_len,max_word_len = chars.shape
        
        chars = chars.view(-1,max_word_len)
        char_seq_len = char_seq_len.view(-1)
        
        embed_chars = self.char_embedding(chars)
        
        hidden,(state,cell) = self.rnn_layer(embed_chars,char_seq_len,total_length = max_word_len)
        ret = state[:,-1,:,:]
        ret = torch.cat((ret[:,0,:],ret[:,1,:]),dim = 1)
        ret = ret.view(B,max_sent_len,self.word_encoding_dim)
        
        return ret
        
if __name__ =="__main__":
    encoder = CharLSTMEncoder(5,10,8)
    sents = [[[2, 1, 1, 1], [5, 6, 4, 3], [5, 0, 3, 1], [1, 1, 1, 1]], 
             [[0, 3, 1, 1], [5, 0, 4, 3], [1, 1, 1, 1], [1, 1, 1, 1]]]
    seq_len = [[1, 4, 3, 1], 
               [2, 4, 1, 1]]
    sents = torch.tensor(sents)
    seq_len = torch.tensor(seq_len)
    print encoder(sents,seq_len)
