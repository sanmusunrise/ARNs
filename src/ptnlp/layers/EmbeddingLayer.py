import sys 
sys.path.append("../../") 
from ptnlp.utils import logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class EmbeddingLayer(nn.Module):

    def __init__(self,dim,trainable = True,dtype = None):
        super(EmbeddingLayer, self).__init__()
        self.embeddings = None
        self.trainable = trainable
        #self.id2word = {}
        #self.word2id = {}
        self.dim = dim
        self.dtype = dtype if dtype else torch.float32

    def initialize_with_random(self,num_vocab):
        self.embeddings = nn.Embedding(num_vocab, self.dim)


    def load_from_pretrain(self,pretrain_file,
                           word2id = None,
                           trainable = None,
                           data_format = "word2vec_text"):
        id2vec = {}
        for line in open(pretrain_file):
            line = line.decode("utf-8").strip().split()
            word = line[0]
            vec = [float(i) for i in line[1:]]
            if len(vec) != self.dim:
                logger.warning( "Dimension of vector for word '%s' is not coherent with the default demension, skipped." % word )
                continue
            if word2id and word not in word2id:
                #print word.encode("utf-8"), "in word2vec file but not in word2id"
                continue
            if word2id:
                idx = word2id[word]
                id2vec[idx] = vec
            else:
                id2vec[len(id2vec)] = vec
        
        embeddings = []
        if word2id:
            total_words = len(word2id)
        else:
            total_words = len(id2vec)

        for i in xrange(total_words):
            if i in id2vec:
                embeddings.append(id2vec[i])
            else:
                #print "word in word2id but not in word2vec, randomly initialized"
                embeddings.append(self.create_random_embeddings())

        assert len(embeddings) == len(word2id)
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(embeddings,dtype = self.dtype),freeze = not self.trainable)

    def create_random_embeddings(self):
        scale = 0.1
        vec = np.random.uniform(low=-scale, high=scale, size=[self.dim]) 
        #vec = vec / np.sqrt(np.sum(vec*vec))
        
        return list(vec)

    def forward(self,x):
        if not self.embeddings:
            logger.error("Using an word embedding without initialization, exit.")
            exit(-1)
        return self.embeddings(x)
    
    def save(self,file_name):
        pass

    def restore(self,file_name):
        pass
    

if __name__ =="__main__":
    word2id = {"I":0,
               "you":2,
               "love":1,
               ".":3}
    embeddings = EmbeddingLayer(3,trainable = False)
    embeddings.load_from_pretrain(sys.argv[1],word2id)
    #embeddings.initialize_with_random(4)
    x = [0,2]
    y = embeddings(torch.tensor(x,dtype = torch.long))
    print y
