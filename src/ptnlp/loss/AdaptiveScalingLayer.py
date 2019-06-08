import torch
import torch.nn as nn

class AdaptiveScalingLayer(nn.Module):

    def __init__(self, instance_loss, neg_indexes, label_size, beta=1.0, size_average=True):
        super(AdaptiveScalingLayer, self).__init__()
        self.loss = instance_loss
        self.neg_indexes = neg_indexes
        self.beta = beta
        self.label_size = label_size
        self.size_average = size_average

    def forward(self, x, target):

        batch_size = target.size(0)
        prob = nn.functional.softmax(x, 1).data
        target_data = target.data.unsqueeze(1)

        pos_label_mask = torch.ones(batch_size, self.label_size)
        neg_label_mask = torch.zeros(batch_size, self.label_size)
        for index in self.neg_indexes:
            pos_label_mask[:, index] = 0.
            neg_label_mask[:, index] = 1.

        pos_label_mask = pos_label_mask.to(x.device)
        neg_label_mask = neg_label_mask.to(x.device)
        
        tp = torch.sum(torch.gather(prob * pos_label_mask, 1, target_data))
        tn = torch.sum(torch.gather(prob * neg_label_mask, 1, target_data))
        p = torch.gather(pos_label_mask, 1, target_data)
        n = torch.gather(neg_label_mask, 1, target_data)
        weight_beta = tp / (self.beta * self.beta * torch.sum(p).item() + torch.sum(n).item() - tn)
        #print map(float,[weight_beta,tp,tn,torch.sum(p),torch.sum(n)])
        weight_beta = (n.float() * weight_beta + p.float())#.detach()
         
        instance_loss = self.loss.forward(x, target)
        weight_loss = instance_loss * weight_beta

        if self.size_average:
            #loss = torch.mean(weight_loss)
            #print loss
            return torch.mean(weight_loss)
        else:
            return weight_loss
            
if __name__ =="__main__":
    from InstanceCostSensitiveLossLayer import *
    
    y_pred = [[1,8,4,5],
              [2,9,0,3],
              [-1,0,2,1.0]]
    targets = [1,2,0]
    
    y_pred = torch.tensor(y_pred)
    targets = torch.tensor(targets)
    loss = InstanceCostSensitiveLossLayer(4,1,False)
    loss = AdaptiveScalingLayer(loss,[0],4)
    print loss(y_pred,targets)
