import torch
import copy
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models.transformer import Patching
from models.transformer import TranEncoder
from models.transformer import TranDecoder
from models.transformer import MultiHeadAttention
from models.transformer import Merger_pool
from models.loss import DistillationLoss, ReconstructionLoss

from utils import visible_mask_div, patch_div

class TranBot(nn.Module):
    """
    Use PatchTST (Transfomer) or TS2Vec (dilated CNN) as encoder.
    This class will be used to train the model in main.py.
    """
    
    def __init__(
        self,
        ts_dim,
        n_patch,
        padding,
        batch_size,
        dropout=0,
        mask_ratio=0.2,
        d_model=64,
        n_hierarchy=3,
        n_layers=2, 
        d_k=64, 
        d_v=64, 
        d_ff=64, 
        n_heads=4,
        patch_size=24,
        stride=24,
        alpha=0.5,
        beta=0.5,
        device='cuda',
        lr=0.001,
        after_epoch_callback=None,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.n_patch = n_patch
        
        
        self.patch = Patching(patch_size, stride, padding).to(self.device)  
        self.student_net = TranEncoder(patch_size * ts_dim, d_model, n_hierarchy, n_layers, d_k, d_v, d_ff, n_heads, dropout).to(self.device)
        self.teacher_net = copy.deepcopy(self.student_net).to(self.device)
        for param in self.teacher_net.parameters():
            param.requires_grad = False
        self.decoder = TranDecoder(patch_size * ts_dim, d_model, n_hierarchy, d_k, d_v, d_ff, n_heads).to(self.device)
        self.merger = Merger_pool().to(self.device)
        self.m_query = torch.randn(batch_size, int(self.n_patch * mask_ratio) ,d_model).to(self.device)
        self.m_query.requires_grad = True
        self.cross_scale_layer = MultiHeadAttention(d_model, d_k, d_v, n_heads).to(self.device)
        self.patch_num = [int(n_patch / 2**i) for i in range(n_hierarchy)]
        print(self.patch_num)
        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0

        self.optimizer = torch.optim.Adam([*self.student_net.parameters(), self.m_query], lr=self.lr)

    def ema_update(self, alpha=0.99):
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = alpha * param_t.data + (1 - alpha) * param_s.data

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TiBot model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3

        if n_epochs is None:
            n_epochs = 40
        loss_log = []

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        
        for _ in range(n_epochs):
            for i, (batch,) in enumerate(train_loader):
                # self.teacher_net detach

                self.optimizer.zero_grad()
                
                # copy the student weights to the teacher 
                # self.teacher.load_state_dict(self.student.state_dict())

                x = batch.to(self.device)
                # print(x.shape)
                # print(f'x shape: {x.shape}')
                x_patch = self.patch(x)
                # print(x_patch.shape)
                visible_patch, mask_patch = visible_mask_div(x_patch, self.mask_ratio)
                # print(mask_patch.shape)
                # print(visible_patch.shape)
                # print(self.n_patch)
                assert visible_patch.shape[1] + mask_patch.shape[1] == self.n_patch              
                outputs, attens = self.student_net(visible_patch, )
                z, pred = self.decoder(self.m_query, outputs)
                z_hat, _ = self.teacher_net(mask_patch)

                # for i in range(3):
                #     print(outputs[i].shape, z[i].shape, z_hat[i].shape, pred[i].shape)
 
                d_loss = DistillationLoss(z, z_hat)
                r_loss = ReconstructionLoss(mask_patch, pred, self.merger)
                loss = self.alpha * d_loss + self.beta * r_loss
                loss.mean().backward()
                self.optimizer.step()
                self.teacher_net = copy.deepcopy(self.student_net).to(self.device)
                # self.ema_update()
                # import ipdb
                # ipdb.set_trace()
            
            print(f'Epoch {self.n_epochs+1} loss: {loss.mean().item()}')
            self.n_epochs += 1

        return loss
    
    def encode(self, x):
        x = torch.from_numpy(x).to(torch.float).to(self.device)
        x_patch = self.patch(x)
        outputs , attn = self.student_net(x_patch)
        shape = []
        idx = 0
        idx_list = [0]
        all_rep = []
        for output in outputs:
            idx += output.shape[1]
            shape.append(output.shape[1])
            idx_list.append(idx)

        linear_list = []
        for i in range(len(shape)):
            linear = nn.Linear(shape[i], x.shape[1]).to(self.device)
            linear_list.append(linear)
        
        outputs_joint = torch.cat(outputs, dim=1)
        rep_joint, _ = self.cross_scale_layer(outputs_joint, outputs_joint, outputs_joint, None)
        for i in range(len(shape)):
            linear = nn.Linear(shape[i], x.shape[1]).to(self.device)
            rep_joint = rep_joint[:, idx_list[i]:idx_list[i+1]]
            rep = linear(rep_joint.transpose(1, 2)).transpose(1, 2)
            all_rep.append(rep)
        all_rep = torch.sum(torch.stack(all_rep, dim=0), dim=0)

        return all_rep.to('cpu').detach().numpy()
    
    def save(self, path):
        torch.save(self.state_dict(), path)