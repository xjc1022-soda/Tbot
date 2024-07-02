import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class TiBot(nn.Module):
    """
    Use PatchTST (Transfomer) as encoder.
    This class will be used to train the model in train.py.
    """
    
    def __init__(self) -> None:
        pass