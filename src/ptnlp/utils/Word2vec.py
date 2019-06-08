from numpy_utils import cosine_similarity
import gensim
class Word2vec():
    def __init__(self,vector_file,binary=True):
        self.word_embeddings = None
        self.load_word2vec(vector_file,binary)

    def load_word2vec(self,file_name,binary):
        model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=binary)
        self.word_embeddings = model
    def get_similarity(self,w1,w2):
        if not ((w1 in self.word_embeddings) and (w2 in self.word_embeddings)):
            return None
        return cosine_similarity(self.word_embeddings[w1],self.word_embeddings[w2])
    def get_vector(self,word):
        return self.word_embeddings[word].tolist()

    def is_oov(self,w):
        if w in self.word_embeddings:
            return False
        return True


if __name__ =="__main__":
    import sys 
    vector_file = sys.argv[1]
    pair_file = sys.argv[2]
    model = Word2vec(vector_file) 
    #print model.word_embeddings
    #for key in model.word_embeddings:
        #print [key]
    
    
    for line in open(pair_file):
        line = line.strip().split()

        word1 = line[0]
        word2 = line[1]

        print word1,word2,model.get_similarity(word1,word2)
    
