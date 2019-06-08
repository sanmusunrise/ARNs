import sys 
sys.path.append("../")

from Seq2Nugget.Seq2Nugget import Seq2Nugget
from Seq2Nugget.Seq2NuggetNERData import Seq2NuggetNERData

import torch

if __name__ =="__main__":
    device_id = 0
    torch.cuda.set_device(device_id)
    
    data_dir = "../../data/KBP2017/"
    pretrain_embedding = data_dir + "glove_6b_dim100.dat"
    max_seq_len = 200
    max_word_len = 20
    win_size = 200
    
    train_config = {"detection_train_epoch":0,
                    "boundary_train_epoch":0,
                    "joint_train_epoch":100,
                    "batch_size":30,
                    "detection_coldstart":True,
                    "boundary_coldstart":True,
                    "weight_decay_round":5,
                    "detection_learning_rate":1.5,
                    "boundary_learning_rate":1.0,
                    "negative_weight":2.0,
                    "max_margin":0.75,
                    "train_detection_in_joint":True,
                    "max_seq_len":max_seq_len,
                    "device":torch.device('cuda:' + str(device_id)),
                    "word_dict_file" : data_dir + "word_dict.dat",
                    "pos_dict_file"  : data_dir + "pos_dict.dat",
                    "char_dict_file" : data_dir + "char_dict.dat",
                    "save_dir" : "KBP_Model/",
                    "is_save_config":True}
    
    detection_config = {"pretrain_file" : pretrain_embedding,
                        "embedding_trainable": True,
                        "word_embedding_dim": 100,
                        "pos_embedding_dim": 20,
                        "char_embedding_dim": 50,
                        "word_encoding_dim": 100,
                        "max_seq_len": max_seq_len,
                        "hidden_dim" : 500, #100
                        "output_dim": 8,
                        "dropout_rate":0.3}
                        
    boundary_config =  {"pretrain_file" : pretrain_embedding,
                        "embedding_trainable": True,
                        "word_embedding_dim": 100,
                        "pos_embedding_dim": 20,
                        "char_embedding_dim": 50,
                        "word_encoding_dim": 100,
                        "max_seq_len": max_seq_len,
                        "hidden_dim" : 500, #100
                        "output_dim": 8,
                        "dropout_rate":0.3,
                        "win_size":win_size}
                        
    label2id_file = data_dir + "label2id.dat"
    
    model = Seq2Nugget(train_config,detection_config,boundary_config)
    word_dict = model.word_dict
    pos_dict = model.pos_dict
    char_dict = model.char_dict
    
    train_set = Seq2NuggetNERData(word_dict = word_dict,pos_dict = pos_dict,char_dict = char_dict, 
                                 data_file = data_dir + "train.dat",
                                 label2id_file = label2id_file,
                                 max_seq_len = max_seq_len,
                                 max_word_len = max_word_len,
                                 win_size = win_size,
                                 is_train = True)
    dev_set = Seq2NuggetNERData(word_dict = word_dict,pos_dict = pos_dict,char_dict = char_dict,
                               data_file = data_dir + "dev.dat",
                               label2id_file = label2id_file,
                               max_seq_len = max_seq_len,
                               max_word_len = max_word_len,
                               win_size = win_size,
                               is_train = False)
    test_set = Seq2NuggetNERData(word_dict = word_dict,pos_dict = pos_dict, char_dict = char_dict,
                                data_file = data_dir + "test.dat",
                                win_size = win_size,
                                label2id_file = label2id_file,
                                max_seq_len = max_seq_len,
                                max_word_len = max_word_len,
                                is_train = False)
    model.train(train_set,dev_set,test_set)
    #detector.train(train_set)

