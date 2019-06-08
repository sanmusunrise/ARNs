import torch

class MaskedCrossEntropyLayer(torch.nn.Module):

    def __init__(self):
        super(MaskedCrossEntropyLayer,self).__init__()
        self.epsilon = 1e-8

    def forward(self,y_pred,targets,seq_mask,weight = None):
        shape = y_pred.size()
        label_size = shape[-1]
        y_pred = y_pred.view(-1,label_size)
        targets = targets.view(-1)
        seq_mask = seq_mask.view(-1)
        if weight is None:
            weight = torch.ones(label_size,dtype = torch.float).to(device = y_pred.device)
        #print seq_mask
        #weight = torch.tensor([1.0] *8,device = y_pred.get_device())
        #weight[0] = 0.2
        loss = torch.nn.functional.cross_entropy(y_pred,targets,weight = weight, reduction = 'none')
        loss = torch.dot(loss,seq_mask) / (torch.sum(seq_mask) + self.epsilon)

        return loss

