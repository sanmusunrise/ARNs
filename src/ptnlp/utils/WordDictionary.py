# Filename: WordDictionary.py
# Author: Hongyu Lin
# Last_Update: 2018/11/4

# Update:
# 2018/11/04 : Adding case-sensitive


"""Word Dictionary to create and map between words and ids


"""

class WordDictionary():
    
    def __init__(self,case_sensitive = False):
        self.word2cnt = {}
        self.id2word = []
        self.word2id = {}
        self.min_cnt = None
        self.add_unk = None
        self.sort_by_cnt = None
        self.case_sensitive = case_sensitive

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

    def id2word_translate(self,sents):
        return [self.id2word[i] if i < len(self.id2word) else "<ERR_WORD_ID>" for i in sents]


    def word2id_translate(self,sents,max_len = -1,padding = True):
        sent_len = len(sents)
        
        if self.case_sensitive:
            ret = [self.word2id[i] if i in self.word2id else self.word2id["<UNK>"] for i in sents]
        else:
            ret = [self.word2id[i.lower()] if i.lower() in self.word2id else self.word2id["<UNK>"] for i in sents]
        if max_len >0:
            ret = ret[:max_len]
        if padding and len(ret) < max_len:
            ret = ret + [self.word2id["<PADDING>"]] * (max_len - len(ret))
        return ret,min(sent_len,len(ret))

    
    def padding(self,sents,max_len):    
        sent_len = len(sents)
        if len(sents) < max_len:
            ret = sents + [self.word2id["<PADDING>"]] * (max_len - sent_len)
        else:
            ret = sents
        ret = ret[:max_len]
        return ret,min(sent_len,len(ret))
    
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
            idx, word,cnt = line.decode("utf-8").strip().split("\t")
            self.word2cnt[word] = int(cnt)
            self.id2word.append(word)
            self.word2id[word] = int(idx)

        assert int(idx) == len(self.id2word)-1


if __name__ =="__main__":
    
    word2cnt = {"I":10,
                "love":3,
                "you":1,
                ".":5}
    
    dictionary = WordDictionary()
    dictionary.load_from_word2cnt(word2cnt,min_cnt=0,sort_by_cnt = True)
    
    sent = "I love you .".split(" ")
    ids = dictionary.word2id_translate(sent)
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
