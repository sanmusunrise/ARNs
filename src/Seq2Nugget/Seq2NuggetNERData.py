import sys 
sys.path.append("..")

from utils.NERData import *
import random
from ptnlp.utils.WordDictionary import WordDictionary
from ptnlp.utils.CharDictionary import CharDictionary

class Seq2NuggetNERData(NERData):

    def __init__(self,**kwargs):
        super(Seq2NuggetNERData, self).__init__(**kwargs)
        self.char_dict = kwargs["char_dict"]
        self.max_word_len = kwargs["max_word_len"]
        self.create_sent_feats()
        self.win_size = kwargs["win_size"] 
        if self.is_train:
            self.create_training_data()
    
    def create_sent_feats(self):
        for words,poss in self.sentences:
            word_ids,word_len = self.word_dict.word2id_translate(words,max_len = self.max_seq_len,padding =True)
            pos_ids,pos_len = self.pos_dict.word2id_translate(poss,max_len = self.max_seq_len,padding =True)
            char_ids,char_len,word_len_from_char = self.char_dict.word2id_translate(words, max_word_len = self.max_word_len, max_sent_len = self.max_seq_len) 

            assert word_len ==pos_len
            assert word_len == word_len_from_char
            self.sent_feats.append((word_ids,pos_ids,word_len,char_ids,char_len))
        
        assert len(self.sent_feats) == len(self.sentences)

    def create_training_data(self):
        training_data = []  #(sent_id,[label_id])
        for sent_id,sent_annos in self.annotations.items():
            anns = [self.label2id['NIL']] * self.max_seq_len
            left_label = range(self.max_seq_len)
            right_label = range(self.max_seq_len)
            anns_span =[]
            for i in xrange(len(anns)):
                anns_span.append((i,i+1))
            
            anno_len = [999] * self.max_seq_len

            for b,e,l in sent_annos:
                length = e-b
                if e >=self.max_seq_len:
                    continue
                for i in xrange(b,e):
                    #if i-b > self.win_size:
                    #    continue
                    #if e-1-i > self.win_size:
                    #    continue
                    if i < self.max_seq_len and anno_len[i] > length:
                        anns[i] = self.label2id[l]
                        left_label[i] = b
                        right_label[i] = e-1
                        anns_span[i] = (b,e)
                        anno_len[i] = length
            packages = []
            for b1,e1 in anns_span:
                pack = [0.0] * len(anns_span)
                for i,(b2,e2) in enumerate(anns_span):
                    if b1 ==b2 and e1 ==e2:
                        pack[i] = 1.0
                packages.append(pack)

            training_data.append((sent_id,anns,left_label,right_label,packages))
        self.training_data = training_data

    
    def mini_batches_for_train(self,batch_size = 30):        
        random.shuffle(self.training_data)
        current_batch = self.create_empty_batch()
        for sent_id,annos,left_label,right_label,pack in self.training_data:
            words,poss,seq_len,char_ids,char_len = self.sent_feats[sent_id]

            current_batch['sent_ids'].append(sent_id)
            current_batch['words'].append(words)
            current_batch['chars'].append(char_ids)
            current_batch['char_len'].append(char_len)
            current_batch['seq_len'].append(seq_len)
            current_batch['poss'].append(poss)
            
            current_batch['cls_labels'].append(annos)
            current_batch['left_labels'].append(left_label)
            current_batch['right_labels'].append(right_label)

            seq_mask = [0.0] * self.max_seq_len
            for i in xrange(seq_len):
                seq_mask[i] = 1.0
            current_batch['seq_mask'].append(seq_mask)
            current_batch["packages"].append(pack)

            current_batch['batch_size'] +=1

            if current_batch['batch_size'] ==batch_size:
                yield current_batch
                current_batch = self.create_empty_batch()

        if current_batch['batch_size']:
            yield current_batch

    def mini_batches_for_test(self,batch_size = 100):
        
        current_batch = self.create_empty_batch()

        for sent_id,(words,poss,seq_len,chars,char_len) in enumerate(self.sent_feats):
            current_batch['sent_ids'].append(sent_id)
            current_batch['words'].append(words)
            current_batch['poss'].append(poss)
            current_batch['seq_len'].append(seq_len)
            current_batch['chars'].append(chars)
            current_batch['char_len'].append(char_len)
            
            seq_mask = [0.0] * self.max_seq_len
            for i in xrange(seq_len):
                seq_mask[i] = 1.0
            current_batch['seq_mask'].append(seq_mask)

            current_batch['batch_size'] +=1

            if current_batch['batch_size'] ==batch_size:
                yield current_batch
                current_batch = self.create_empty_batch()

        if current_batch['batch_size']:
            yield current_batch
        


    
    def create_empty_batch(self):
        data = {}
        data['batch_size'] = 0
        data["is_train"] = self.is_train
        
        data["sent_ids"] = []   #(B)
        data["words"] = []  #(B*max_seq_len)
        data["poss"] = []   #(B*max_seq_len)
        data["chars"] = []  #[B,max_seq_len,max_word_len]
        data["char_len"] = []   #[B,max_seq_len]
        data["seq_len"] = [] #(B)
        data["seq_mask"] = []   #(B*max_seq_len)
        data["packages"] = []   #[B,T,T]
        
        data["cls_labels"] = [] #(B*max_seq_len)
        data['left_labels'] = []   #(B*max_seq_len)
        data['right_labels'] = []    #(B*max_seq_len)
        return data




if __name__ =="__main__":
    import sys
    word_dict = WordDictionary()
    word_dict.restore(sys.argv[2])
    pos_dict = WordDictionary()
    pos_dict.restore(sys.argv[3])
    kwargs = {"data_file"     :sys.argv[1],
              "word_dict"     :word_dict,
              "pos_dict"      :pos_dict,
              "label2id_file" :sys.argv[4],
              "is_train"      :True,
              "win_size"      :4,
              "detection_result_file":sys.argv[5]}
    corpus = DetectionNERData(**kwargs)
    '''
    for idx,sent in enumerate(corpus.sentences):
        print sent[0]
        print sent[1]
        print corpus.annotations[idx]
        print "--------------------------"
    '''
    
    for batch_size in [1]:
        total_sents = 0
        for batch in corpus.mini_batches_for_train(batch_size):
            for k,v in batch.items():
                if k =="packages":
                    print k
                    for t in v[0]:
                        print t[:10]
                else:
                    print k,":",v
            print "-----------------------------"
            total_sents += batch['batch_size']
        assert total_sents == len(corpus.sentences)
