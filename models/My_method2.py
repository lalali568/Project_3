import torch.nn as nn
import torch.nn.functional as F
import torch


class My_model(nn.Module):
    """这是主要的函数"""
    def __init__(self, config):
        super().__init__()
        self.config=config
        features=config['input_size']*config['input_size']
        self.lin1=nn.Linear(features,features)
        self.lin2=nn.Linear(features,features)
        size=(config['window_size']-config['padding_size'])//config['padding_stride']
        self.map=nn.Parameter(torch.randn(size,size))




    def forward(self, x):
        bs=x.shape[0]
        t=x.shape[1]
        dim=x.shape[-1]
        x=x.reshape(bs,t,-1)
        x=self.lin1(x)
        x=torch.matmul(self.map,x)
        x=self.lin2(x)
        x=x.reshape(bs,t,dim,dim)
        return x






