import torch
import torch.nn as nn
from torch.autograd import Variable

import sys 
sys.path.append("..")
from ptnlp.layers.EmbeddingLayer import EmbeddingLayer
from ptnlp.layers.LSTMLayer import LSTMLayer
from ptnlp.layers.CharLSTMEncoder import CharLSTMEncoder
from ptnlp.layers.TimeConvLayer import TimeConvLayer
from ptnlp.layers.BilinearAttention2DLayer import BilinearAttention2DLayer


class BoundaryModel(torch.nn.Module):
    
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
        dropout_rate        :The dropout rate after LSTM.
        
        win_size            :The size of windows to detect
        """

        super(BoundaryModel, self).__init__()

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

        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate)
        self.conv_layer_left = TimeConvLayer(self.hidden_dim,self.hidden_dim,2)
        self.conv_layer_right = TimeConvLayer(self.hidden_dim,self.hidden_dim,2)
        
        self.nugget_bilinear_layer_left = BilinearAttention2DLayer(self.hidden_dim, self.hidden_dim)
        self.nugget_bilinear_layer_right = BilinearAttention2DLayer(self.hidden_dim, self.hidden_dim)
        
        self.nugget_linear_layer_left = torch.nn.Linear(self.hidden_dim,1)
        self.nugget_linear_layer_right = torch.nn.Linear(self.hidden_dim,1)

        left_boundary_mask = torch.zeros(self.max_seq_len,self.max_seq_len)
        right_boundary_mask = torch.zeros(self.max_seq_len,self.max_seq_len)
        
        
        for i in xrange(self.max_seq_len):
            left_boundary_mask[i,max(i-self.win_size,0):i+1] = 1
            right_boundary_mask[i,i:min(i+self.win_size+1,self.max_seq_len)] = 1
        left_boundary_mask = (1- left_boundary_mask) * -9999999
        right_boundary_mask = (1- right_boundary_mask)  * -9999999
        
        Parameter = torch.nn.Parameter
        left_boundary_mask = Parameter(left_boundary_mask, requires_grad =False)
        right_boundary_mask = Parameter(right_boundary_mask, requires_grad =False)
        
        self.register_parameter("left_boundary_mask",left_boundary_mask)
        self.register_parameter("right_boundary_mask",right_boundary_mask)
        

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
        hidden = self.dropout_layer(hidden)

        left_features = torch.nn.functional.relu(self.conv_layer_left(hidden)) #[B,T,hidden_dim]
        right_features = torch.nn.functional.relu(self.conv_layer_right(hidden))   #[B,T,hidden_dim]

        bilinaer_nugget_left = self.nugget_bilinear_layer_left(hidden,left_features)  #[B,T,T] stores T scores for the left boundary of [B,T] tokens
        bilinear_nugget_right = self.nugget_bilinear_layer_right(hidden,right_features)  #[B,T,T] stores T scores for the right boundary of [B,T] tokens
        
        #print "left_features shape",left_features.shape
        linear_nugget_left = self.nugget_linear_layer_left(left_features).expand(-1,-1,self.max_seq_len).transpose_(1,2)    #[B,T,T], slices along dim=1 are the same the score of each token as boundary
        linear_nugget_right = self.nugget_linear_layer_right(right_features).expand(-1,-1,self.max_seq_len).transpose_(1,2) #[B,T,T], slices along dim=1 are the same the score of each token as boundary
        
        batch_size = linear_nugget_left.shape[0]
        
        left_scores = linear_nugget_left + bilinaer_nugget_left + self.left_boundary_mask.expand(batch_size,-1,-1)
        right_scores = bilinear_nugget_right + linear_nugget_right + self.right_boundary_mask.expand(batch_size,-1,-1)
        
        
        if do_softmax:
            left_scores = torch.nn.functional.softmax(left_scores,dim=2)      #[B,T,T]
            right_scores = torch.nn.functional.softmax(right_scores,dim=2)      #[B,T,T]
        
        return left_scores, right_scores
