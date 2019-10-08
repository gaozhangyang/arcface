import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import globalvar as gl
import pdb

device=gl.get_value('device')
lbda=gl.get_value('lbda')

class GLinear(Function):
    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    def forward(ctx, input, weight):
        '''
        input: [n,latent_dim]
        weight: [c,latent_dim]
        '''
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        input, weight = ctx.saved_variables
        grad_input = grad_weight =  None

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight

def softFunc(x):
    f=-(x-0.5)*(x<0.5).float()+0+(x-1)*(x>1).float()
    return f

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.direct=gl.get_value('direct')
        self.fc=GLinear.apply
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        #todo 不一样的地方
        if gl.get_value('auxiliary')=='ring':
            Rloss=lbda*torch.mean((torch.norm(x,dim=1)-1)**2)/2
        elif gl.get_value('auxiliary')=='soft':
            Rloss=lbda*torch.mean(softFunc(torch.norm(x,dim=1)-1))/2
        else:
            raise Exception('方法未实现！')
                
        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x,self.direct)
        self.m=gl.get_value('m')
        self.s=gl.get_value('s')
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)#(wf.transpose(0, 1):[10,512] wf.transpose(0, 1)[labels]:[512,512]  torch.diagonal(wf.transpose(0, 1)[labels]):512
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)#exc1:[512,10]
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        
        return -torch.mean(L)+Rloss
