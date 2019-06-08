import sys 
sys.path.append("../../") 
from ptnlp.utils import logger
from ptnlp.functions.batch_index_select import *
from ptnlp.functions.SeqMaskGenerator import *
import torch
import torch.nn as nn

class MiddleAverageSpanEncoder(nn.Module):
    
    def __init__(self,max_seq_len):
        
        super(MiddleAverageSpanEncoder, self).__init__()
        self.seq_mask_gen = SeqMaskGenerator(max_seq_len)

    
    def forward(self,x,seq_len):
        '''
        Parameters:
        x               : [B,max_span_len,d]
        seq_len         : An int tensor with shape [B]
        
        Return:
        ret             :[B,3*d]
        '''
        B, max_span_len, d = x.shape
        left_idx = torch.zeros(B,1).to(device = x.device,dtype = torch.long)
        right_idx = (seq_len -1).view(B,1)
        
        left_features = batch_index_select(x,left_idx).squeeze(dim=1)
        right_features = batch_index_select(x,right_idx).squeeze(dim=1)
        
        seq_mask = self.seq_mask_gen(seq_len)
        seq_mask_right = self.seq_mask_gen(seq_len-1)
        seq_mask[:,0] -= 1
        seq_mask = seq_mask * seq_mask_right
        seq_mask = seq_mask.expand(d,-1,-1).transpose(0,1).transpose(1,2)
        middle_features = torch.sum(x * seq_mask,dim = 1) / (torch.sum(seq_mask,dim=1) + 1e-8)

        ret = torch.cat([left_features,middle_features,right_features],dim = 1)
        return ret
        
        
        
        
if __name__ =="__main__":
    device_id = 6
    torch.cuda.set_device(device_id)
    device = torch.device('cuda:6')

    encoder = MiddleAverageSpanEncoder(4).to(device = device)
    x = [[[1.0,2,3,4],[4,5,6,7],[7,8,9,10],[10,11,12,13]],
         [[7,8,9,10],[10,11,12,13],[7,8,9,10],[10,11,12,13]],
         [[7,8,9,10],[10,11,12,13],[7,8,9,10],[10,11,12,13]]]
    x = torch.tensor(x).to(device = device)
    
    seq_len = [1,2,4]
    seq_len = torch.tensor(seq_len).to(device = device)
    print encoder(x,seq_len)
