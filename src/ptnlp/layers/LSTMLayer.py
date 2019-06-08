import torch
import torch.nn as nn
from torch.autograd import Variable

import sys 
sys.path.append("../../")

from ptnlp.utils.utils import argsort
from ptnlp.utils import logger

class LSTMLayer(torch.nn.Module):
    
    def __init__(self,D_in,D_out,n_layers = 1,dropout = 0, bidirectional = True):
        
        super(LSTMLayer, self).__init__()

        self.n_layers = n_layers
        self.D_in = D_in
        self.D_out = D_out
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        if self.bidirectional and self.D_out %2:
            logger.error( "Odd D_out in LSTMLayer with bidirectional ==True, Exit.")
            exit(-1)
        if self.bidirectional:
            self.D_out /=2

        self.rnn = nn.LSTM(input_size = self.D_in,
                           hidden_size = self.D_out,
                           num_layers = self.n_layers,
                           bias = True,
                           batch_first = True,
                           dropout = dropout,
                           bidirectional = bidirectional)
            
    def get_output_dim(self):
        if self.bidirectional:
            return self.D_out *2
        return self.D_out

    def get_zero_hidden(self,batch_size,device):
        n_layers = self.n_layers
        if self.bidirectional:
            n_layers *=2
        h0 = Variable(torch.zeros(n_layers, batch_size, self.D_out)).to(device = device)
        c0 = Variable(torch.zeros(n_layers, batch_size, self.D_out)).to(device = device)

        return (h0,c0)

    def sort_batch_by_len(self,x,seq_len,dim = 0):
        seq_len,order = torch.sort(seq_len,descending = True)
        #seq_len = [seq_len[i] for i in order]
        x = torch.index_select(x,dim=dim,index = order)
        return x,seq_len,order

    def recover_batch_order(self,x,seq_len,order,dim = 0):
        seq_len, original_order = torch.sort(order,descending = False)
        #seq_len = [seq_len[i] for i in original_order]
        x = torch.index_select(x,dim=dim,index = original_order)

        return x,seq_len



    def forward(self,x,seq_len,total_length = None,init_hidden = None,sort = False):
        
        if not total_length:
            total_length = max(seq_len)
        
        if not init_hidden:
            init_hidden = self.get_zero_hidden(len(seq_len),x.device)
        
        if not sort:
            x,new_seq_len, order = self.sort_batch_by_len(x,seq_len,0)
            if init_hidden:
                h_init,_,_ = self.sort_batch_by_len(init_hidden[0],seq_len,1)
                c_init,_,_ = self.sort_batch_by_len(init_hidden[1],seq_len,1)
            seq_len = new_seq_len

        pack = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True)
        out, (hidden,cell) = self.rnn(pack, init_hidden)
        
        out,_ = torch.nn.utils.rnn.pad_packed_sequence(out,batch_first = True,total_length = total_length)

        hidden = hidden.view(self.n_layers, 2 if self.bidirectional else 1,len(seq_len), self.D_out).permute([2,0,1,3])
        cell = cell.view(self.n_layers, 2 if self.bidirectional else 1,len(seq_len), self.D_out).permute([2,0,1,3])
        
        if not sort:
            out,new_seq_len = self.recover_batch_order(out,seq_len,order,0)
            hidden,_ = self.recover_batch_order(hidden,seq_len,order,0)
            cell,_ = self.recover_batch_order(cell,seq_len,order,0)

        # Output dim: (B*total_length * (direction * D_out))
        # hidden/cell dim: (B*n_layer*direction*D_out)
        return out,(hidden,cell)


if __name__ =="__main__":
    torch.cuda.set_device(4)
    device = torch.device("cuda:4")
    #device = torch.device("cpu")
    batch_size = 3
    max_length = 3
    hidden_size = 5
    n_layers =1

    # container
    batch_in = torch.zeros((batch_size, max_length, 1)).to(device = device)
    #print batch_in

    vec_1 = torch.FloatTensor([[1, 2, 3]]).t().to(device = device)
    vec_2 = torch.FloatTensor([[1, 2, 0]]).t().to(device = device)
    vec_3 = torch.FloatTensor([[1, 0, 0]]).t().to(device = device)

    batch_in[0] = vec_1
    batch_in[1] = vec_2
    batch_in[2] = vec_3

    seq_lengths = torch.tensor([3,1,2]).to(device = device)

    rnn = LSTMLayer(1,10,2).to(device = device)
    
    #init_state = rnn.get_zero_hidden(len(seq_lengths),device = device)
    batch_in = Variable(batch_in).to(device = device)
    output,hidden = rnn(batch_in,seq_lengths,5)
    print output
    print "----------------"    
    ret =  hidden[0][:,-1,:,:]
    ret = torch.cat((ret[:,0,:],ret[:,1,:]),dim = 1)
    print ret.shape
