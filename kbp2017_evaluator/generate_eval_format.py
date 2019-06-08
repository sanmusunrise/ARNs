import sys
import os

def load_sys_output(file_name):
    ret = {}
    for line in open(file_name):
        line = line.strip().split("\t")
        sent_id = int(line[0])
        ret[sent_id] = []
        for ss in line[1:]:
            ss = ss.split("|")
            ret[sent_id].append((int(ss[0]),int(ss[1]) -1,ss[2]))
    return ret

def load_sents_to_doc(file_name):
    data = []
    f = open(file_name)
    content = f.readlines()
    data += content
    assert len(data) %4 ==0
    
    sent2doc = {}
    sent2off = {}
    for idx in xrange(len(data)/4):
        annos = data[idx*4+3].strip()
        annos = annos.split()
        tokens = data[idx*4].strip().split()
        #print len(tokens),len(annos) -1,idx
        assert len(tokens) == len(annos) -1
        doc_id = annos[0]
        sent2doc[idx] = doc_id
        sent2off[idx] = []
        for t in annos[1:]:
            t = t.split(",")
            sent2off[idx].append( (int(t[0]),int(t[0]) + int(t[1]) -1) )
    return sent2doc, sent2off

def generate_sys_output(sys_output,sent2doc,sent2off):
    ret = []
    for sent_id in sys_output:
        doc_id = sent2doc[sent_id]
        for b,e,t in sys_output[sent_id]:
            #print sent2off[sent_id][b]
            b_off = sent2off[sent_id][b][0]
            #print len(sent2off[sent_id]),b,e,sent_id
            if e >=len(sent2off[sent_id]):
                #print "cross boundary"
                e = len(sent2off[sent_id]) -1
            e_off = sent2off[sent_id][e][1]
            ret.append( (doc_id,str(b_off),str(e_off),"NIL000",str(1.0),t) )
    return ret
    
def generate_golden(file_name):
    ret = []
    for line in open(file_name):
        line = line.strip().split()
        ret.append( (line[0],line[1],line[2],"NIL000",str(1.0),line[4]) )
    return ret

def output_file(data,file_name):
    output = open(file_name,"w")
    for tup in data:
        output.write("\t".join(tup) + "\n")
    output.close()
    
if __name__ =="__main__":
    sys_file = sys.argv[1]
    sys_output = load_sys_output(sys_file)
    data_file = "../data/KBP2017/test.dat"
    sent2doc, sent2off = load_sents_to_doc(data_file)
    sys_tup = generate_sys_output(sys_output,sent2doc, sent2off)
    #golden_file = "/home1/hongyu/NER/model_data/KBP2017/all_annotation/test.dat"
    #gol_tup = generate_golden(golden_file)
    output_file(sys_tup,"sys_tmp.dat")
    #output_file(gol_tup,"golden.dat")
    os.system("cat sys_tmp.dat kbp2017_test_eval_files/authors.dat > " + sys.argv[2])
    os.system("rm sys_tmp.dat")

