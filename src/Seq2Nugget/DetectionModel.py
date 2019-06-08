import torch
import torch.nn as nn
from torch.autograd import Variable

import sys 
sys.path.append("..")
from ptnlp.layers.EmbeddingLayer import EmbeddingLayer
from ptnlp.layers.LSTMLayer import LSTMLayer
from ptnlp.layers.CharLSTMEncoder import CharLSTMEncoder
from ptnlp.layers.TimeConvLayer import TimeConvLayer


class DetectionModel(torch.nn.Module):
    
    def __init__(self,**kwargs):
        """
        Parameters in kwargs:
        word2id             :Word2id mapping from dictionary used to initialize WordEmbedding from pretrain
        pretrain_file       :File contains pretrained word embedding
        embedding_trainable :Whether the pretrained embedding is trainable, [default True]
        pos2id              :POS2id mapping from dictionary used to randomly initialize POS Embedding
        word_embedding_dim  :Dimension of WordEmbedding
        pos_embedding_dim   :Dimension of POSEmbedding

        char_embedding_dim  :Dimension of character embedding vector
        word_encoding_dim   :Dimension of output word representation from chars
        char2id             :char2id mapping from dictionary used to randomly initialize char embedding
        
        max_seq_len         :The maximum sequence length
        hidden_dim          :Dimension of LSTM output, must be an Even
        output_dim          :Dimension of the output dense layer, equals to the label size
        dropout_rate        :The dropout rate after LSTM.
        """

        super(DetectionModel, self).__init__()

        for key in kwargs:
            self.__dict__[key] = kwargs[key]
        
        self.word_embedding = EmbeddingLayer(dim = self.word_embedding_dim, trainable = self.embedding_trainable)
        self.word_embedding.load_from_pretrain(self.pretrain_file,self.word2id)
        
        self.char_encoder = CharLSTMEncoder(char_embedding_dim = self.char_embedding_dim,
                                            word_encoding_dim = self.word_encoding_dim,
                                            num_vocab = len(self.char2id))

        self.pos_embedding = EmbeddingLayer(dim = self.pos_embedding_dim, trainable = True)
        self.pos_embedding.initialize_with_random(len(self.pos2id))
        
        self.rnn_layer = LSTMLayer(D_in = self.word_embedding_dim + self.pos_embedding_dim + self.word_encoding_dim, 
                                   D_out = self.hidden_dim,
                                   n_layers = 1,
                                   dropout = 0,
                                   bidirectional = True)
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate)    #0.3
        
        self.dense_layer = torch.nn.Linear(self.hidden_dim,self.hidden_dim)
        self.conv_layer = TimeConvLayer(self.hidden_dim,self.hidden_dim,3)
        self.output_layer = torch.nn.Linear(self.hidden_dim,self.output_dim)    

    def forward(self,do_softmax = False,**kwargs):
        """
        Parameters in kwargs:
        words                   :Input tensor contains word_ids with shape [B,T]
        poss                    :Input tensor contains POS_ids with shape[B,T]
        seq_len                 :Input tensor contains length of each sequence with shape [B]
        chars                   :Input tensor contains char_ids with shape [B,max_seq_len,max_word_len]
        char_len                :Input tensor contains length of each words with shape [B,max_seq_len],
                                 Please note that the length of padding words should be set to **at least 1** as pytorch rnn requires.
        """
    
        words = kwargs["words"]
        poss = kwargs["poss"]
        seq_len = kwargs["seq_len"]
        chars = kwargs["chars"]
        char_len = kwargs["char_len"]

        embed_words = self.word_embedding(words)                            #[B,T,word_embedding_dim]
        embed_poss = self.pos_embedding(poss)                               #[B,T,pos_embedding_dim]
        embed_chars = self.char_encoder(chars,char_len)                     #[B,T,word_encoding_dim]
        concat_embedding = torch.cat((embed_words,embed_poss,embed_chars),dim = 2)      #[B,T,word_embedding_dim + pos_embedding_dim + word_encoding_dim]

        hidden,_ = self.rnn_layer(concat_embedding,seq_len,total_length = self.max_seq_len)   #[B,T,hidden_dim]

        hidden = torch.nn.functional.relu(self.dense_layer(hidden))
        #dropout2 = torch.nn.Dropout(p=0.3)
        hidden = self.dropout_layer(hidden)
        output = self.output_layer(hidden)   #[B,T,output_dim]
        
        if do_softmax:
            output = torch.nn.functional.softmax(output,dim=2)      #[B,T,output_dim]

        return output
