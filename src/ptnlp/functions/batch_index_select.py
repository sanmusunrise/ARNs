import torch

def batch_index_select(x,index,dim=1):
    '''
    Parameters:
    x       : The input data, the first dimension is the batch, Commonly a [B,T,d] tensor
    index   : The index to select, with shape [B,k]
    dim     : The dimension along with the selection happens, must >=1 and < x.dim
    '''
    
    trans_x = torch.transpose(x,dim,1)
    shape = list(trans_x.shape[2:]) + list(index.shape)
    mask = index.expand(shape)
    mask = torch.transpose(mask,0,len(shape)-2)
    mask = torch.transpose(mask,1,len(shape)-1)
    ret = torch.gather(trans_x,dim=1,index = mask)
    
    return torch.transpose(ret,1,dim)
    

if __name__ =="__main__":
    a = [[[1,2,3],[4,5,6]],
         [[7,8,9],[10,11,12]],
         [[2,2,3],[4,5,6]],
         [[8,8,9],[10,11,12]]]
    a = torch.tensor([a,a])
    print a
   
    index = [[1],
             [0]]#,
             #[0],
             #[1]]
    index = torch.tensor(index)
    print batch_index_select(a,index,1)
