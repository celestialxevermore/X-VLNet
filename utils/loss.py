import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class KL_Divergence(nn.Module):
    def __init__(self,):
        super(KL_Divergence,self).__init__()
    def forward(self,sim_matrix):
        t2v_logpt = F.log_softmax(sim_matrix,dim=-1)
        v2t_logpt = F.log_softmax(sim_matrix.T,dim=-1)
        t2v_logpt = -torch.diag(t2v_logpt)
        v2t_logpt = -torch.diag(v2t_logpt)
        
        t2v_logpt /= t2v_logpt.sum()
        v2t_logpt /= v2t_logpt.sum()
        
        t2v_loss = sum(t2v_logpt * torch.log(t2v_logpt/v2t_logpt))
        v2t_loss = sum(v2t_logpt * torch.log(v2t_logpt/t2v_logpt))
        return t2v_loss,v2t_loss

class dual_softmax_loss_t(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss_t, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss