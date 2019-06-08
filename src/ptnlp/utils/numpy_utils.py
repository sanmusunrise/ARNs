import numpy as np
from scipy import linalg, mat, dot

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def cosine_similarity(a,b):
    a = mat(a)
    b = mat(b)

    c = dot(a,b.T)/linalg.norm(a)/linalg.norm(b)
    return c[0,0]

def matrix_argkmax(m,k):
    row_num,col_num = m.shape
    arr = m.reshape((1,-1))[0]
    order = arr.argsort()[-k:]
    result = []
    for idx in order:
        row = idx / col_num
        col = idx % col_num
        result.append((row,col))
    result.reverse()
    return result


class BowTransformer():

    def __init__(self,total_word):
        self.total_word = total_word
        #self.iden_mat = np.identity(total_word)

    def transform_vec(self,word_ids):
        bow = np.zeros(self.total_word)
        for w in word_ids:
            bow[w] += 1
        return bow
    def transform_matrix(self,doc):
        bow = np.zeros((len(doc),self.total_word))
        for i in xrange(len(doc)):
            for w in doc[i]:
                bow[i][w] += 1        
        return bow



if __name__ == "__main__":
    a = [5,6,3]
    b = [2,4,8]
    #iden_mat = np.identity(5)
    #total_word = 5
    #print BowTransformer(total_word).transform_matrix(a)
    #a = np.array(a)
    #print matrix_argkmax(a,2)
    c = cosine_similarity(a,b)
    print c[0,0]
    print cosine_similarity(a,b)
