# Filename: WordDictionary.py
# Author: Hongyu Lin
# Last_Update: 2018/8/20


"""Char Dictionary to create and map between words and char_ids


"""

class CharDictionary():
    
    def __init__(self):
        self.word2cnt = {}
        self.id2word = []
        self.word2id = {}
        self.min_cnt = None
        self.add_unk = None
        self.sort_by_cnt = None

    def load_from_word2cnt(self,word2cnt,min_cnt = 0,sort_by_cnt = True):
        self.word2cnt = word2cnt
        self.id2word = []
        self.word2id = {}
        self.min_cnt = min_cnt
        #self.add_unk = add_unk
        self.sort_by_cnt = sort_by_cnt
        
        self.id2word.append("<UNK>")
        self.word2id["<UNK>"] = 0
        self.id2word.append("<PADDING>")
        self.word2id["<PADDING>"] = 1

        w_cnt_pairs = word2cnt.items()
        if sort_by_cnt:
            w_cnt_pairs = sorted(w_cnt_pairs,key = lambda x:x[1],reverse = True)
        
        self.word2cnt["<UNK>"] = 9999999
        self.word2cnt["<PADDING>"] = 9999999

        for w,cnt in w_cnt_pairs:
            if cnt < min_cnt:
                continue
            if w in self.word2id:
                continue
            idx = len(self.word2id)
            self.word2id[w] = idx
            self.id2word.append(w)

    def singleid2word_translate(self,sents):
        return "".join([self.id2word[i] if i < len(self.id2word) else "<ERR_TOKEN_ID>" for i in sents])


    def word2id_translate(self,words,max_word_len = -1,max_sent_len = -1,padding = True):
        sent_len = len(words)

        ret = [self.single_word2id_translate(w,max_word_len,padding = True) for w in words]
        if max_sent_len >0:
            ret = ret[:max_sent_len]
        if padding and len(ret) <max_sent_len:
            ret = ret + [([self.word2id["<PADDING>"]] * max_word_len,1)] * (max_sent_len- len(ret))

        ret_word = []
        ret_len = []
        
        for cs,l in ret:
            ret_word.append(cs)
            ret_len.append(l)
        return ret_word,ret_len,min(sent_len,len(ret_word))

    def padding(self,words,char_lens,max_word_len,max_sent_len):
        sent_len = len(words)
        assert sent_len == len(char_lens)
        
        words = words + [[self.word2id["<PADDING>"]] * max_word_len] * (max_sent_len- sent_len)
        char_lens = char_lens + [1] * (max_sent_len- sent_len)
        words = words[:max_sent_len]
        char_lens = char_lens[:max_sent_len]

        return words,char_lens,min(sent_len,len(words))

    def single_word2id_translate(self,word,max_len = -1,padding = True):
        word_len = len(word)

        ret = [self.word2id[i] if i in self.word2id else self.word2id["<UNK>"] for i in word]
        if max_len >0:
            ret = ret[:max_len]
        if padding and len(ret) < max_len:
            ret = ret + [self.word2id["<PADDING>"]] * (max_len - len(ret))
        return ret,min(word_len,len(ret))

    def save(self,file_path):
        out = open(file_path,"w")
        out.write("%d\t%s\n"% (self.min_cnt,self.sort_by_cnt))
        for i,word in enumerate(self.id2word):
            data_line = "%d\t%s\t%d\n" % (i,word,self.word2cnt[word])
            out.write(data_line.encode("utf-8"))


    def restore(self,file_path):
        
        f = open(file_path)
        h = f.next().strip().split("\t")
        self.min_cnt = int(h[0])
        self.sort_by_cnt = eval(h[1])
        
        for line in f:
            idx, word,cnt = line.strip().split("\t")
            self.word2cnt[word] = int(cnt)
            self.id2word.append(word)
            self.word2id[word] = int(idx)

        assert int(idx) == len(self.id2word)-1


if __name__ =="__main__":
    
    word2cnt = {"I":10,
                "l":3,
                "o":1,
                "v":5,
                "e":9}
        
    dictionary = CharDictionary()
    dictionary.load_from_word2cnt(word2cnt,min_cnt=0,sort_by_cnt = True)
    print dictionary.id2word
    sent1 = "I love lae".split(" ")
    sent2 = "he lave".split(" ")
    words1,cl1,sl1 = dictionary.word2id_translate(sent1,4,4)
    words2,cl2,sl2 =  dictionary.word2id_translate(sent2,4,4)
    print [cl1,cl2]
    print [words1,words2]
    #for i,j in zip(char_lengs,words):
    #    print i,j
    #print sent_len

    exit(0)
    
    words = dictionary.id2word_translate(ids)
    print ids
    print words
    dictionary.save("test.dat")
    
    d1 = WordDictionary()
    d1.restore("test.dat")
    ids = d1.word2id_translate(sent)
    words = d1.id2word_translate(ids)
    print ids
    print words
