import torch

import sys 
sys.path.append("../../")
from ptnlp.functions.SeqMaskGenerator import SeqMaskGenerator
from ptnlp.functions.SelectMaskGenerator import SelectMaskGenerator
from torch.nn import Parameter as Parameter

class WeightedMarginLossLayer(torch.nn.Module):

    def __init__(self,target_size,margin,left_wt=1,right_wt=1):
        super(WeightedMarginLossLayer,self).__init__()
        
        self.margin  = margin
        self.target_size = target_size
        self.left_wt = left_wt
        self.right_wt = right_wt
        
        MAX_VAL = Parameter(torch.tensor(99999.0),requires_grad =False)
        ZERO = Parameter(torch.tensor(0,dtype = torch.float),requires_grad =False)
        self.register_parameter("MAX_VAL",MAX_VAL)
        self.register_parameter("ZERO",ZERO)
        
        self.seq_mask = SeqMaskGenerator(target_size)
        self.select_mask = SelectMaskGenerator(target_size)
    
    def golden_score(self,y_pred,targets):
        return torch.gather(y_pred,dim=1,index = targets.unsqueeze(dim=1)).expand(-1,self.target_size)
    
    def forward(self,y_pred,targets):
        '''
        Parameters:
        y_pred      : [B,C] tensor saves the output score for C choices of B instances*.
        targets     : [B] tensor, the target choice of each instance
        '''
        
        left_mask = self.seq_mask(targets)
        golden_mask = self.select_mask(targets)
        right_mask = 1-left_mask - golden_mask

        weight_mtx = left_mask * self.left_wt + right_mask * self.right_wt

        gol_pred = self.golden_score(y_pred,targets)
        pred_margin = gol_pred - y_pred

        element_loss = torch.max(self.ZERO,self.margin - pred_margin) * weight_mtx
        
        instance_loss = torch.sum(element_loss,dim = 1)
        instance_wt = torch.sum(weight_mtx,dim=1) * self.target_size
        loss = torch.sum(instance_loss / instance_wt)
        
        return loss


if __name__ =="__main__":
    
    y_pred = [[1,8,4,5],
              [2,9,0,3],
              [-2,0,2,1.0]]
    targets = [0,2,3]
    y_pred = torch.tensor(y_pred)
    targets = torch.tensor(targets)
    margin = 2
    
    
    layer = WeightedMarginLossLayer(4,margin)
    print layer(y_pred,targets)
