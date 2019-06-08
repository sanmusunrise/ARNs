import torch
from torch.nn import Parameter as Parameter

class SelectMaskGenerator(torch.nn.Module):
    def __init__(self,max_len):
        super(SelectMaskGenerator,self).__init__()
        self.max_len = max_len
        tri_mtx = torch.eye(self.max_len)
        tri_mtx = Parameter(tri_mtx,requires_grad =False)
        
        self.register_parameter("tri_mtx",tri_mtx)
    
    def forward(self,seq_len):
        input_shape = list(seq_len.shape)
        output_shape = input_shape + [self.max_len]
        
        seq_len = seq_len.view(-1)
        mask = torch.index_select(self.tri_mtx, 0, seq_len).view(*output_shape)
        
        return mask
        
        
if __name__ =="__main__":
    mg = SelectMaskGenerator(5)

    seq_len = torch.tensor([3,4,2,0,4,0,3])
    print mg(seq_len)
    
