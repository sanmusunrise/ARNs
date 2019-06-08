import sys 
sys.path.append("..")

from Seq2NuggetNERData import Seq2NuggetNERData
from BoundaryModel import BoundaryModel
from DetectionModel import DetectionModel
from ptnlp.utils.WordDictionary import WordDictionary
from ptnlp.utils.CharDictionary import CharDictionary
from ptnlp.loss.MaxMarginLossLayer import MaxMarginLossLayer
from ptnlp.loss.BagLossWithMarginLayer import BagLossWithMarginLayer
from ptnlp.loss.BagLossLayer import BagLossLayer
from ptnlp.utils import logger
from utils.ExactMatchEvaluator import ExactMatchEvaluator
from utils.DetectionEvaluatorForE2E import DetectionEvaluator
from Seq2NuggetLogger import Seq2NuggetLogger
import torch
import sys
import shelve
import os

class Seq2Nugget(object):
    
    def __init__(self,train_config,detection_config,boundary_config):
        self.initialize(train_config,detection_config,boundary_config)


    def initialize(self,train_config,detection_config,boundary_config):
        """
        Parameters:

        """

        for key in train_config:
            self.__dict__[key] = train_config[key]

        self.word_dict = WordDictionary()
        self.word_dict.restore(self.word_dict_file)
        self.pos_dict = WordDictionary()
        self.pos_dict.restore(self.pos_dict_file)

        self.char_dict = CharDictionary()
        self.char_dict.restore(self.char_dict_file)
        
        self.detection_model_config = detection_config
        self.boundary_model_config = boundary_config
        
        self.detection_model_config["word2id"] = self.word_dict.word2id
        self.detection_model_config["pos2id"] = self.pos_dict.word2id
        self.detection_model_config["char2id"] = self.char_dict.word2id
        
        self.boundary_model_config["word2id"] = self.word_dict.word2id
        self.boundary_model_config["pos2id"] = self.pos_dict.word2id
        self.boundary_model_config["char2id"] = self.char_dict.word2id
        
        
        self.detection_model = DetectionModel(**self.detection_model_config)
        self.boundary_model = BoundaryModel(**self.boundary_model_config)

        self.detection_model.to(device = self.device)
        self.boundary_model.to(device = self.device)
        
        self.joint_loss_fn = BagLossWithMarginLayer()
        self.detection_loss_fn = BagLossLayer()
        self.boundary_loss_fn = MaxMarginLossLayer(self.max_seq_len, self.max_margin).to(device = self.device)
        
        self.detection_optimizer = torch.optim.Adadelta(self.detection_model.parameters(), lr=self.detection_learning_rate)
        self.boundary_optimizer = torch.optim.Adadelta(self.boundary_model.parameters(), lr=self.boundary_learning_rate)
        
        self.result_logger = Seq2NuggetLogger()
        self.match_eval = ExactMatchEvaluator()
        self.det_eval = DetectionEvaluator()
        
        self.current_epoch = -1
        self.detection_lr_decay_before = 0
        self.boundary_lr_decay_before = 0

        if self.is_save_config:
            self.save_config(train_config,detection_config,boundary_config)

    def update_detection_lr(self):
        if self.detection_lr_decay_before >=self.weight_decay_round and self.current_epoch - self.result_logger.best_detection_epoch() >= self.weight_decay_round:
            self.detection_lr_decay_before = 0
            for g in self.detection_optimizer.param_groups:
                g['lr'] /=2
            return True
        self.detection_lr_decay_before +=1
        return False
    
    def update_boundary_lr(self):
        if self.boundary_lr_decay_before >= self.weight_decay_round and self.current_epoch - self.result_logger.best_boundary_epoch() >= self.weight_decay_round:
            self.boundary_lr_decay_before = 0
            for g in self.boundary_optimizer.param_groups:
                g['lr'] /=2
            return True
        self.boundary_lr_decay_before +=1
        return False
    
    def update_all_lr(self):
        if self.current_epoch - self.result_logger.best_all_epoch()  >= self.weight_decay_round:
            for g in self.boundary_optimizer.param_groups:
                g['lr'] /=2
            for g in self.detection_optimizer.param_groups:
                g['lr'] /=2
            return True
        return False
    
    def restore_best_detection(self,dev_data,test_data):
        logger.info("Restore previous best detection model at epoch %d." % (self.current_epoch))
        self.restore_using_best(True,False)
        self.decode_and_log(dev_data,test_data)
        logger.info("\n" + str(self.result_logger.result_at_epoch()))
    
    def restore_best_boundary(self,dev_data,test_data):
        logger.info("Restore previous best Boundary model at epoch %d." % (self.current_epoch))
        self.restore_using_best(False,True)
        self.decode_and_log(dev_data,test_data)
        logger.info("\n" + str(self.result_logger.result_at_epoch()))
    
    def train(self,train_data,dev_data = None,test_data =None):
         
        if not self.detection_coldstart:
            self.current_epoch +=1
            logger.info("Trying to restore previous detection model at epoch %d." % (self.current_epoch))
            self.restore_best_detection(dev_data,test_data)

        if not self.boundary_coldstart:
            self.current_epoch +=1
            logger.info("Trying to restore previous boundary model at epoch %d." % (self.current_epoch))
            self.restore_best_boundary(dev_data,test_data)
        
        for epoch in xrange(self.detection_train_epoch):
            self.current_epoch +=1
            epoch_loss = self.train_detection_epoch(train_data)
            logger.info( "Training detection model at epoch %d, the loss is %f" %(self.current_epoch,epoch_loss))
            self.decode_and_log(dev_data,test_data)
            logger.info("\n" + str(self.result_logger.result_at_epoch()))
            
            if self.result_logger.is_best_detection():
                logger.info("Better detection performance achieved at epoch %d, saving the detection model." %(self.current_epoch))
                self.save_best(True,False)
            
            if self.update_detection_lr():
                self.current_epoch +=1
                logger.info("Detection Learning rate decay at epoch %d" % (self.current_epoch))
                self.restore_best_detection(dev_data,test_data)
            
        #restore best detection model currently.
        if self.detection_train_epoch:
            self.current_epoch +=1
            logger.info("Finish training detection model at epoch %d." % (self.current_epoch))
            self.restore_best_detection(dev_data,test_data)
        
        for epoch in xrange(self.boundary_train_epoch):
            self.current_epoch +=1
            epoch_loss = self.train_boundary_epoch(train_data)
            logger.info( "Training boundary model at epoch %d, the loss is %f" %(self.current_epoch,epoch_loss))
            self.decode_and_log(dev_data,test_data)
            logger.info("\n" + str(self.result_logger.result_at_epoch()))
            
            if self.result_logger.is_best_boundary():
                logger.info("Better Boundary performance achieved at epoch %d, saving the boundary model." %(self.current_epoch))
                self.save_best(False,True)
                
            
            if self.update_boundary_lr():
                self.current_epoch +=1
                logger.info("Boundary Learning rate decay at epoch %d" % (self.current_epoch))
                self.restore_best_boundary(dev_data,test_data)
            
        #restore best boundary model currently.
        if self.boundary_train_epoch:
            self.current_epoch +=1
            logger.info("Finish training boundary model at epoch %d." % (self.current_epoch))
            self.restore_best_boundary(dev_data,test_data)
        
        
        for epoch in xrange(self.joint_train_epoch):
            self.current_epoch +=1
            epoch_loss = self.train_joint_epoch(train_data)
            logger.info( "Training joint model at epoch %d, the loss is %f" %(self.current_epoch,epoch_loss))
            self.decode_and_log(dev_data,test_data)
            logger.info("\n" + str(self.result_logger.result_at_epoch()))
            
            if self.train_detection_in_joint and self.result_logger.is_best_detection():
                logger.info("Better detection performance achieved at epoch %d, saving the detection model." %(self.current_epoch))
                self.save_best(True,False)
            
            if self.result_logger.is_best_boundary():
                logger.info("Better Boundary performance achieved at epoch %d, saving the boundary model." %(self.current_epoch))
                self.save_best(False,True)
            
            if self.result_logger.is_best_overall():
                logger.info("Better Overall performance achieved at epoch %d, saving the boundary model." %(self.current_epoch))
                self.save_best(False,True)
                logger.info("Save two overall best models at epoch %d." %(self.current_epoch))
                self.save_all_model(self.save_dir +"snapshot_best_overall" + ".dat")
                
            
            if self.update_detection_lr() or self.update_boundary_lr():
                self.current_epoch +=1
                logger.info("Learning rate decay at epoch %d" % (self.current_epoch))
                self.restore_best_detection(dev_data,test_data)
                self.restore_best_boundary(dev_data,test_data)
            
        #restore all models
        self.current_epoch +=1
        logger.info("Finish the entire training at epoch %d, now restore previous best models." % (self.current_epoch))
        self.restore_using_best(True,True)
        self.decode_and_log(dev_data,test_data,is_save = True)
        logger.info("\n" + str(self.result_logger.result_at_epoch()))
        
    def decode_and_log(self,dev_data = None,test_data = None,is_save = False):
        if dev_data:
            dev_result = self.test(dev_data)
            dev_p,dev_r,dev_f1 = self.match_eval.eval(dev_result,dev_data.annotations)
            dev_dp,dev_dr,dev_df1 = self.det_eval.eval(dev_result,dev_data.annotations)
            
            self.result_logger.update_detection_result("dev",dev_dp,dev_dr,dev_df1)
            self.result_logger.update_boundary_result("dev",dev_p,dev_r,dev_f1)
            
            if is_save:
                self.save_decode_result(dev_result,"dev_epoch_%d" %(self.current_epoch))
            
        if test_data:
            test_result = self.test(test_data)
            test_p,test_r,test_f1 = self.match_eval.eval(test_result,test_data.annotations)
            test_dp,test_dr,test_df1 = self.det_eval.eval(test_result,test_data.annotations)
            
            self.result_logger.update_detection_result("test",test_dp,test_dr,test_df1)
            self.result_logger.update_boundary_result("test",test_p,test_r,test_f1)
            
            if is_save:
                self.save_decode_result(test_result,"test_epoch_%d" %(self.current_epoch))

    def train_detection_epoch(self,data):
        epoch_loss = 0.0
        self.detection_model.train()
        self.boundary_model.train()

        for batch in data.mini_batches_for_train(self.batch_size):
            
            words = torch.tensor(batch["words"], device =self.device)
            poss = torch.tensor(batch["poss"], device = self.device)
            seq_len = torch.tensor(batch["seq_len"], device = self.device)

            chars = torch.tensor(batch["chars"], device = self.device)
            char_len = torch.tensor(batch["char_len"], device = self.device)
            
            cls_labels = torch.tensor(batch["cls_labels"], device = self.device)
            left_labels = torch.tensor(batch["left_labels"], device = self.device)
            right_labels = torch.tensor(batch["right_labels"], device = self.device)
            
            seq_mask = torch.tensor(batch["seq_mask"], device = self.device)
            packages = torch.tensor(batch["packages"],device = self.device)
            
            cls_pred = self.detection_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)
                                                       
            weight = [self.negative_weight] + [1.0] * (self.detection_model.output_dim -1)
            weight = torch.tensor(weight).to(device = self.device)
            
            detection_loss = self.detection_loss_fn(cls_pred,cls_labels,packages,seq_mask,weight = weight)
            
            self.detection_optimizer.zero_grad()
            detection_loss.backward()
            self.detection_optimizer.step()
            
            epoch_loss += detection_loss.item()

        return epoch_loss

        
    def train_boundary_epoch(self,data):
        epoch_loss = 0.0
        self.detection_model.train()
        self.boundary_model.train()

        for batch in data.mini_batches_for_train(self.batch_size):
            
            words = torch.tensor(batch["words"], device =self.device)
            poss = torch.tensor(batch["poss"], device = self.device)
            seq_len = torch.tensor(batch["seq_len"], device = self.device)

            chars = torch.tensor(batch["chars"], device = self.device)
            char_len = torch.tensor(batch["char_len"], device = self.device)
            
            cls_labels = torch.tensor(batch["cls_labels"], device = self.device)
            left_labels = torch.tensor(batch["left_labels"], device = self.device)
            right_labels = torch.tensor(batch["right_labels"], device = self.device)
            
            seq_mask = torch.tensor(batch["seq_mask"], device = self.device)
            packages = torch.tensor(batch["packages"],device = self.device)

            left_pred,right_pred = self.boundary_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)                                  
            left_pred = torch.tanh(left_pred)
            right_pred = torch.tanh(right_pred)
            
            B,T1,T2 = left_pred.shape
            assert T1 == T2
            left_pred = left_pred.view(B*T1,T2)
            right_pred = right_pred.view(B*T1,T2)
            left_labels = left_labels.view(B*T1)
            right_labels = right_labels.view(B*T1)
            
            NIL_ID = data.label2id['NIL']
            anchor_mask = (cls_labels !=NIL_ID).float()

            boundary_loss = self.boundary_loss_fn(left_pred,left_labels,False) + self.boundary_loss_fn(right_pred,right_labels,False)
            boundary_loss = boundary_loss.view(B,T1) * anchor_mask

            boundary_loss = torch.sum(boundary_loss) / torch.sum(anchor_mask)
            
            self.boundary_optimizer.zero_grad()
            boundary_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.boundary_model.parameters(),max_norm = 3)
            self.boundary_optimizer.step()
            
            epoch_loss += boundary_loss.item()

        return epoch_loss
    
    def train_joint_epoch(self,data):
        
        epoch_loss = 0.0
        self.detection_model.train()
        self.boundary_model.train()

        for batch in data.mini_batches_for_train(self.batch_size):
            
            words = torch.tensor(batch["words"], device =self.device)
            poss = torch.tensor(batch["poss"], device = self.device)
            seq_len = torch.tensor(batch["seq_len"], device = self.device)

            chars = torch.tensor(batch["chars"], device = self.device)
            char_len = torch.tensor(batch["char_len"], device = self.device)
            
            cls_labels = torch.tensor(batch["cls_labels"], device = self.device)
            left_labels = torch.tensor(batch["left_labels"], device = self.device)
            right_labels = torch.tensor(batch["right_labels"], device = self.device)
            
            seq_mask = torch.tensor(batch["seq_mask"], device = self.device)
            packages = torch.tensor(batch["packages"],device = self.device)

            cls_pred = self.detection_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)
            left_pred,right_pred = self.boundary_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)                                  
            left_pred = torch.tanh(left_pred)
            right_pred = torch.tanh(right_pred)
            
            weight = [self.negative_weight] + [1.0] * (self.detection_model.output_dim -1)
            weight = torch.tensor(weight).to(device = self.device)
            
            B,T1,T2 = left_pred.shape
            assert T1 == T2
            left_pred = left_pred.view(B*T1,T2)
            right_pred = right_pred.view(B*T1,T2)
            left_labels = left_labels.view(B*T1)
            right_labels = right_labels.view(B*T1)
            
            NIL_ID = data.label2id['NIL']
            anchor_mask = (cls_labels !=NIL_ID).float()

            boundary_loss = self.boundary_loss_fn(left_pred,left_labels,False) + self.boundary_loss_fn(right_pred,right_labels,False)
            boundary_loss = boundary_loss.view(B,T1) * anchor_mask

            loss = self.joint_loss_fn(cls_pred,cls_labels,packages,boundary_loss,seq_mask,weight = weight)
            
            self.detection_optimizer.zero_grad()
            self.boundary_optimizer.zero_grad()
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.boundary_model.parameters(),max_norm = 3)
            if self.train_detection_in_joint:
                self.detection_optimizer.step()
            self.boundary_optimizer.step()
            
            epoch_loss += loss.item()

        return epoch_loss

    def test(self,data):
        self.detection_model.eval()
        self.boundary_model.eval()

        test_result = {}
        for batch in data.mini_batches_for_test(batch_size=100):
            words = torch.tensor(batch["words"], device =self.device)
            poss = torch.tensor(batch["poss"], device = self.device)
            seq_len = torch.tensor(batch["seq_len"], device = self.device)
            chars = torch.tensor(batch["chars"], device = self.device)
            char_len = torch.tensor(batch["char_len"], device = self.device)


            cls_pred = self.detection_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)
            left_pred,right_pred = self.boundary_model(do_softmax = False, 
                                                       words = words,
                                                       poss = poss, 
                                                       chars = chars,
                                                       char_len = char_len,
                                                       seq_len = seq_len)    
                                              
            cls_outputs = torch.argmax(cls_pred,dim = 2).tolist()                                 
            left_outputs = torch.argmax(left_pred,dim = 2).tolist()   #[B,T] which stores the left label_id
            right_outputs = torch.argmax(right_pred,dim = 2).tolist()   #[B,T] which stores the left label_id
            
            assert len(batch["sent_ids"]) ==len(left_outputs)

            for sent_idx,(sent_id,sent_cls_pred,sent_len) in enumerate(zip(batch["sent_ids"],cls_outputs,seq_len.tolist())):
                for token_idx,label in enumerate(sent_cls_pred):
                    if token_idx >= sent_len:
                        continue
                    if label == data.label2id['NIL']:
                        continue
                    if not sent_id in test_result:
                        test_result[sent_id] =[]
                    b = left_outputs[sent_idx][token_idx]
                    e = right_outputs[sent_idx][token_idx] +1
                    test_result[sent_id].append((b,e,data.id2label[label],token_idx))

        return test_result
    
    def save_config(self,train_config,detection_config,boundary_config):
        file_name = self.save_dir + "model_config.dat"
        #exceptions = ['detection_model','boundary_model','detection_optimizer','boundary_optimizer','best_f1']
        
        f = shelve.open(file_name)
        f["train_config"] = train_config
        f["detection_config"] = detection_config
        f["boundary_config"] = boundary_config
        f.close()

    def save_detection_model(self,file_name):
        torch.save(self.detection_model.state_dict(),file_name+ ".detection_model")
        
    def save_boundary_model(self,file_name):
        torch.save(self.boundary_model.state_dict(),file_name+".boundary_model")
        
    def restore_detection_model(self,file_name):
        if os.path.exists(file_name+ ".detection_model"):
            self.detection_model.load_state_dict(torch.load(file_name+ ".detection_model",map_location=lambda storage, loc: storage.cuda(self.device) ))
            logger.info("Restore detection model from %s" %(file_name+ ".detection_model" ))
        else:
            logger.info("Restore from %s failed, no file exists." %(file_name+ ".detection_model"))
    def restore_boundary_model(self,file_name):
        if os.path.exists(file_name+ ".boundary_model"):
            self.boundary_model.load_state_dict(torch.load(file_name+ ".boundary_model", map_location=lambda storage, loc: storage.cuda(self.device)))
            logger.info("Restore boundary model from %s" %(file_name+ ".boundary_model"))
        else:
            logger.info("Restore from %s failed, no file exists." %(file_name+ ".boundary_model"))
            
    def resotre_all_model(self,file_name,dev_data,test_data):
        self.restore_detection_model(file_name,dev_data,test_data)
        self.restore_boundary_model(file_name,dev_data,test_data)
    
    def save_all_model(self,file_name):
        self.save_detection_model(file_name)
        self.save_boundary_model(file_name)
        

    def restore_config(self,file_name):
        f = shelve.open(file_name)
        for key in f:
            self.__dict[key] = f[key]

    def restore_model(self,file_name):
        self.model = torch.load(file_name + ".model")
        self.optimizer = torch.load(file_name + ".optimizer")

        f = shelve.open(file_name + ".info")
        self.best_f1 = f['best_f1']
        self.current_epoch = f['current_epoch']
        f.close()

    def restore_using_best(self,restore_detection = True, restore_boundary = True):
        file_name = self.save_dir + "snapshot_best" + ".dat"
        if restore_detection:
            self.restore_detection_model(file_name)
        if restore_boundary:
            self.restore_boundary_model(file_name)

    def save_epoch(self):
        file_name = self.save_dir +"snapshot_" + str(self.current_epoch) + ".dat"
        self.save_model(file_name)

    def save_best(self,save_detection = True, save_boundary = True):
        file_name = self.save_dir +"snapshot_best" + ".dat"
        if save_detection:
            self.save_detection_model(file_name)
        if save_boundary:
            self.save_boundary_model(file_name)
            
    def save_decode_result(self,result,prefix):
        file_name = self.save_dir + prefix +"_result" + ".dat"
        output = open(file_name,"w")
        for sent_id in result:
            output.write(str(sent_id) + "\t")
            for b,e,tp,c in result[sent_id]:
                s = "|".join(map(str,[b,e,tp,c]))
                output.write(s + "\t")
            output.write("\n")

