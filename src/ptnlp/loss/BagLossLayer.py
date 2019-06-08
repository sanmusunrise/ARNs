import torch

class BagLossLayer(torch.nn.Module):

    def __init__(self,NIL_index = 0):
        super(BagLossLayer,self).__init__()
        self.NIL_index = NIL_index
        self.epsilon = 1e-8
        self.alpha = 1

    def forward(self,y_pred,targets,packages,seq_mask,weight = None):
        '''
        Parameters:
        y_pred      : [B,T,C] tensor saves the output score of [B,T] instance to each class **before softmax**.
        packages    : [B,T,T] tensor, for each of [B,T] instances, save a T dimension vecotor indicates whether each instance 
                      is in the same package of current instance.
        targets     : [B,T] tensor, the target labels of each instance
        seq_mask    : [B,T] tensor, indicates the instance is true instance or padding
        weight      : [C] tensor, indicates the weight of each class
        '''
        
        probs = torch.nn.functional.softmax(y_pred,dim=2)
        B,T,C = probs.shape
        
        golden_probs = torch.gather(probs,dim=2,index = targets.unsqueeze(dim=2))    #[B,T,1] stores the golden probs of each instance
        #NIL_probs = torch.index_select(probs,dim=2,index = torch.tensor([self.NIL_index],dtype = torch.long,device = probs.device))   #[B,T,1] stores the NIL probs of each instance
        #NIL_probs = probs[:, :, self.NIL_index].unsqueeze(dim=2) #[B,T,1] stores the NIL probs of each instance

        probs_in_package = golden_probs.expand(B,T,T).transpose(1,2)
        probs_in_package = probs_in_package * packages
        max_probs_in_package,_ = torch.max(probs_in_package,dim=2)            #[B,T] stores the max golden probs in the same package for each instance

        golden_probs = golden_probs.squeeze(dim =2)
        #print "golden_probs:",golden_probs.tolist()
        #print "max_probs:",max_probs_in_package.tolist()
        golden_weight = golden_probs / (max_probs_in_package)    #[B,T] indicates weight for each instance's golden answer
        
        golden_weight = golden_weight.view(-1)
        golden_weight = golden_weight.detach()
        y_pred = y_pred.view(-1,C)
        targets = targets.view(-1)        
        seq_mask = seq_mask.view(-1)

        if weight is None:
            weight = torch.ones(C,dtype = torch.float).to(device = y_pred.device)

        nil_label = torch.tensor([self.NIL_index] *(B*T),dtype = torch.long,device = y_pred.device)
        golden_loss = torch.nn.functional.cross_entropy(y_pred,targets,weight = weight, reduction = 'none')
        nil_loss = torch.nn.functional.cross_entropy(y_pred,nil_label,weight = weight,reduction = 'none')
        #print "weight:",golden_weight.tolist()
        #print "----------------"

        loss = golden_weight * golden_loss + (1- golden_weight) * nil_loss
        #loss = golden_loss
        loss = torch.dot(loss,seq_mask) / (torch.sum(seq_mask) + self.epsilon)

        return loss


if __name__ =="__main__":

    a = [[[1,2,3],[3,2,1]],
         [[4,6,3],[3,1,4]],
         [[5,7,8],[6,0,1]],
         [[7,3,2],[9,9,6]]]

    targets = [[2,2],
               [0,1],
               [1,1],
               [1,1]]
    packages = [[[1,1],[1,1]],
                [[1,0],[0,1]],
                [[1,1],[1,1]],
                [[1,0],[0,1]]]

    seq_mask = [[1,1],
                [1,0],
                [1,1],
                [1,1]]

    a = torch.tensor(a,dtype = torch.float)
    targets = torch.tensor(targets,dtype = torch.long)
    packages = torch.tensor(packages,dtype = torch.float)
    seq_mask = torch.tensor(seq_mask,dtype = torch.float)
    layer = AtLeastOneLossLayer(0)
    

    print "values:"
    print a
    print 
    print "packages:"
    print packages
    print 
    print "targets:"
    print targets
    print
    print layer(a,targets,packages,seq_mask)
