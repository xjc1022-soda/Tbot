import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class PreHead(nn.Module):
    def __init__(self, student_net,):
        super(PreHead, self).__init__()
        self.student_net = student_net
        for param in self.student_net.parameters():
            param.requires_grad = False
        
    
    def fit(self, x):
        pass

    def forward(self, x):
        return self.linear(x)