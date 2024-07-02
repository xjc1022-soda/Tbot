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
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        device='cuda',
        lr=0.001,
        batch_size=8,
        temporal_unit=0,
        after_epoch_callback=None,
    ):
        pass