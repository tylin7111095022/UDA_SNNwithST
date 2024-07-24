import torch
import torch.nn as nn
from torch.nn import L1Loss

class Distribution_loss(torch.nn.Module):
    """p is target distribution and q is predict distribution"""
    def __init__(self):
        super(Distribution_loss, self).__init__()
        self.metric = self.set_metric()

    def kl_divergence(self,p,q):
        """p and q are both a logit(before softmax function)"""
        prob_p = torch.softmax(p,dim=1)
        kl = (prob_p * torch.log_softmax(p,dim=1)) - (prob_p * torch.log_softmax(q,dim=1))
        # print(f"p*torch.log(p) is {torch.sum(p*torch.log(p))}")
        # print(f"p*torch.log(q) is {torch.sum(p*torch.log(q))}")
        # print(f"mean kl divergence: {torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])}")
        return torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])

    def cross_entropy(self,p,q,weight,is_hardlabel:bool=True):
        """p and q are both a logit(before softmax function)""" 
        if weight is not None:
            if is_hardlabel:
                ce = (-1* p * torch.log_softmax(q, dim=1))*weight # p is hard label
            else:
                ce = (-torch.softmax(p, dim=1) * torch.log_softmax(q, dim=1))*weight
        else:
            if is_hardlabel:
                ce = (-1* p * torch.log_softmax(q, dim=1)) # p is hard label
            else:
                ce = (-torch.softmax(p, dim=1) * torch.log_softmax(q, dim=1))
        # print(f"mean ce: {torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])}")
        return torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])
    
    def asl(self,p,q,weight=None):
        "p is label, q is logit"
        if weight is not None:
            if len(weight.shape) == 4:
                weight = weight.squeeze(1) 
        loss_fn = AsymmetricLoss(gamma_pos=0,gamma_neg=2,clip=0.1)
        loss = loss_fn(x=q,y=p,weight=weight)

        return loss
    
    def l1_loss(self,p,q):
        mae_fn = L1Loss()
        return mae_fn(p,q)

    
    def dice_loss(self,p,q):
        smooth = 1e-8
        prob_p = torch.softmax(p,dim=1)
        prob_q = torch.softmax(q,dim=1)

        inter = torch.sum(prob_p*prob_q) + smooth
        union = torch.sum(prob_p) + torch.sum(prob_q) + smooth
        loss = 1 - ((2*inter) / union)
        return  loss / p.size(0) # loss除以batch size

    def forward(self,p,q, weight=None):
        # assert p.dim() == 4, f"dimension of target distribution has to be 4, but get {p.dim()}"
        # assert p.dim() == q.dim(), f"dimension dismatch between p and q"
        if self.metric == 'kl_divergence':
            return self.kl_divergence(p,q)
        elif self.metric == "cross_entropy":
            return self.cross_entropy(p,q, weight)
        elif self.metric == "dice_loss":
            return self.dice_loss(p,q)
        elif self.metric == "mae":
            return self.l1_loss(p,q)
        elif self.metric == "asl":
            return self.asl(p,q,weight=weight)
        else:
            raise NotImplementedError("the loss metric has not implemented")
        
    def set_metric(self, metric:str="cross_entropy"):
        if metric in ["kl_divergence", "cross_entropy", "dice_loss", "mae", "contrastive", "asl"]:
            self.metric = metric
        else:
            raise NotImplementedError(f"the loss metric has not implemented. metric name must be in kl_divergence or cross_entropy")
        
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y, weight=None):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_prob = torch.softmax(x,dim=1)
        xs_pos = x_prob[:,1,:,:]
        xs_neg = x_prob[:,0,:,:]

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1) # 富樣本的機率如果高於(1-margin), 將其機率直接調成1，這樣計算loss的話便不會考慮到該樣本。

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  
            pt = pt0 + pt1 # pt = p if t > 0 else 1-p
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma) # 記住負樣本機率是(1-p)，所以對照ASL公式看1-(1-p) = p, ASL公式的p為正樣本的機率
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
            if weight is not None:
                loss *= weight

        return -loss.mean()