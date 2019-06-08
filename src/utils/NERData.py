import sys 
sys.path.append("..")

from ptnlp.utils.WordDictionary import WordDictionary as Dict


class NERData(object):

    def __init__(self,**kwargs):
        self.sentences = []     #[([w1,w2,..,],[pos1,pos2,...],len),....]
        
        #The span is recorded as [begin,end).
        self.annotations = {}   #sent_id ->[(begin,end,type),...,]
        
        self.word_dict = None
        self.pos_dict = None
        
        self.label2id = None
        self.id2label = None

        self.data_file_name = None
        self.max_seq_len = 50 if "max_seq_len" not in kwargs else int(kwargs["max_seq_len"])
        self.is_train = True if "is_train" not in kwargs else kwargs["is_train"]

        self.sent_feats = []

        if "word_dict" in kwargs:
            self.set_word_dict(kwargs["word_dict"])
        if "pos_dict"  in kwargs:
            self.set_pos_dict(kwargs["pos_dict"])
        
        if "data_file" in kwargs:
            self.load_data_file(kwargs["data_file"]) 
        if "label2id_file" in kwargs:
            self.load_label2id(kwargs["label2id_file"])

    def load_label2id(self,file_name):
        
        self.label2id = {}
        self.id2label = {}

        for line in open(file_name):
            line = line.strip().split()
            label = line[1]
            i = int(line[0])
            self.label2id[label] = i
            self.id2label[i] = label

    def load_data_file(self,data_file):
        data = []
        for line in open(data_file):
            line = line.decode("utf-8").strip()
            data.append(line)
        
        for idx in xrange(len(data)/4):
            words = data[idx*4]
            words = words.split()
            #words = [w.lower() for w in words]
            poss = data[idx*4+1]
            poss = poss.split()
            annos = data[idx*4+2].strip()
            if len(annos):
                annos = annos.split(" | ")
                #annos = [i.strip() for i in annos]
            else:
                annos = []

            sent_id = len(self.sentences)
            self.annotations[sent_id] = []
            #word_ids,word_len = self.word_dict.word2id_translate(words,max_len = self.max_seq_len,padding =True)
            #pos_ids,pos_len = self.pos_dict.word2id_translate(poss,max_len = self.max_seq_len,padding =True)
            
            #assert word_len == pos_len
            self.sentences.append( (words,poss) )
            
            #print annos
            for an in annos:
                g1 = an.find(",")
                g2 = an.find(" ")
                start = an[:g1]
                end = an[g1+1:g2]
                tp = an[g2+1:]
                self.annotations[sent_id].append( (int(start),int(end),tp) )
        self.data_file_name = data_file[data_file.rfind("/")+1:]
    
    def set_word_dict(self,word_dict):
        self.word_dict = word_dict
    def set_pos_dict(self,pos_dict):
        self.pos_dict = pos_dict
    
    '''
    def load_word_dict(self,dict_file):
        self.word_dict = Dict()
        self.word_dict.restore(dict_file)

    def load_pos_dict(self,dict_file):
        self.pos_dict = Dict()
        self.pos_dict.restore(dict_file)
    '''


    def create_sent_feats(self):
        pass

if __name__ =="__main__":
    import sys
    
    kwargs = {"word_dict_file":sys.argv[2],
              "pos_dict_file" :sys.argv[3],
              "data_file"     :sys.argv[1],
              "label2id_file" :sys.argv[4]}
    corpus = NERData(**kwargs)
    for idx,sent in enumerate(corpus.sentences):
        print sent[0]
        print sent[1]
        print corpus.annotations[idx]
        print "--------------------------"

