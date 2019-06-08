import sys 
import torch
import torch.nn as nn

class AggregateSumLayer(nn.Module):
    
    def __init__(self,max_time):
        super(AggregateSumLayer, self).__init__()
        self.max_time = max_time   
    
    def forward(self,x,reverse = False):
        start = self.max_time-1 if reverse else 0
        end = -1 if reverse else self.max_time
        step = -1 if reverse else 1
        
        cnt = 0
        rst = []
        agg_sum = torch.zeros(x.shape[0],x.shape[2],device = x.device)
        for idx in xrange(start,end,step):
            cnt +=1
            agg_sum += x[:,idx,:]
            rst.append(agg_sum.clone() / cnt)
        
        if reverse:
            rst.reverse()
        return torch.stack(tuple(rst),dim=1)
            
            
        
        
if __name__ =="__main__":
    a = [[[1.0,2,3],[4,5,6]],
         [[7,8,9],[10,11,12]]]
    a = torch.tensor(a)
    print a
    layer = AggregateSumLayer(max_time =2)
    print layer(a,reverse = True)
