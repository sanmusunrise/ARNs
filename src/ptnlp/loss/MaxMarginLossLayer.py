import torch

class MaxMarginLossLayer(torch.nn.Module):

    def __init__(self,target_size,margin):
        super(MaxMarginLossLayer,self).__init__()
        self.epsilon = 1e-8
        self.margin  = margin
        self.target_size = target_size
        
        Parameter = torch.nn.Parameter
        target_select_matrix = Parameter(torch.eye(n=self.target_size,dtype = torch.float),requires_grad =False)
        MAX_VAL = Parameter(torch.tensor(99999.0),requires_grad =False)
        ZERO = Parameter(torch.tensor(0,dtype = torch.float),requires_grad =False)
        
        self.register_parameter("target_select_matrix",target_select_matrix)
        self.register_parameter("MAX_VAL",MAX_VAL)
        self.register_parameter("ZERO",ZERO)
        

    def create_mask(self,targets):
        return torch.index_select(self.target_select_matrix, 0, targets) #[B,C] matrix where the target of each instance is set to 1, otherwise 0.
    
    def golden_score(self,y_pred,targets):
        return torch.gather(y_pred,dim=1,index = targets.unsqueeze(dim=1)).squeeze_()
    
    def max_negative_score(self,y_pred,targets):
        mask = self.create_mask(targets)
        neg_pred = y_pred - self.MAX_VAL * mask
        max_neg,_ = torch.max(neg_pred,dim = 1)
        return max_neg
    
    def forward(self,y_pred,targets,reduction = True):
        '''
        Parameters:
        y_pred      : [B,C] tensor saves the output score for C choices of B instances*.
        targets     : [B] tensor, the target choice of each instance
        '''
        
        gol_pred = self.golden_score(y_pred,targets)
        max_neg_pred = self.max_negative_score(y_pred,targets)
        pred_margin = gol_pred - max_neg_pred
        
        if reduction:
            loss = torch.mean(torch.max(self.ZERO,self.margin - pred_margin))
        else:
            loss = torch.max(self.ZERO,self.margin - pred_margin)
        return loss


if __name__ =="__main__":
    
    y_pred = [[1,8,4,5],
              [2,9,0,3],
              [-1,0,2,1.0]]
    targets = [1,2,2]
    y_pred = torch.tensor(y_pred)
    targets = torch.tensor(targets)
    margin = 2
    
    
    layer = MaxMarginLossLayer(4,margin)
    print layer(y_pred,targets,False)
