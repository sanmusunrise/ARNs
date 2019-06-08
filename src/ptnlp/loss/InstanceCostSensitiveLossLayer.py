from torch.nn import Parameter
import torch.nn as nn
import torch

class InstanceCostSensitiveLossLayer(nn.Module):
    def __init__(self, label_size, lambda_weight =1.0, reduce=False):
        super(InstanceCostSensitiveLossLayer, self).__init__()
        self.lambda_weight = lambda_weight
        self.reduce = reduce
        self.label_size = label_size
        
        mask_except_golden = Parameter(1.0 - torch.eye(n=self.label_size,dtype = torch.float),requires_grad =False)
        self.register_parameter("mask_except_golden",mask_except_golden)
        

    def forward(self, x, targets, mask=None):
        
        prob = nn.functional.softmax(x, 1)
        log_prob = torch.log(prob +1e-8)
        negative_entropy = prob * log_prob
        
        golden_log_prob = torch.gather(log_prob,dim=1,index = targets.unsqueeze(dim=1)).squeeze_()
        
        neg_mask = torch.index_select(self.mask_except_golden, 0, targets)
        regularizer = torch.sum(neg_mask * negative_entropy, dim=1)
        
        loss =  - (golden_log_prob - self.lambda_weight * regularizer)

        if self.reduce:
            return torch.mean(loss)
        else:
            return loss.view(-1,1)
            
if __name__ =="__main__":
    
    y_pred = [[1,8,4,5],
              [2,9,0,3],
              [-1,0,2,1.0]]
    targets = [1,2,2]
    
    y_pred = torch.tensor(y_pred)
    targets = torch.tensor(targets)
    
    loss = InstanceCostSensitiveLossLayer(4,1,False)
    print loss(y_pred,targets)
