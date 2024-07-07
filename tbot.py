import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from models import TSEncoder
from models import TBotLoss
from models import TBotHead

class TBot(nn.Module):
    """
    Use PatchTST (Transfomer) or TS2Vec (dilated CNN) as encoder.
    This class will be used to train the model in main.py.
    """
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=8,
        temporal_unit=0, # control the minimum length of the time series
        after_epoch_callback=None,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.temporal_unit = temporal_unit

        self._teacher_net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self._student_net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)

        # shared cls and patch headers for teacher and student
        self.teacher_head = TBotHead(output_dims, cls_dims=320, patch_dims=320, device=self.device).to(self.device)
        self.student_head = TBotHead(output_dims, cls_dims=320, patch_dims=320, device=self.device).to(self.device)

        # combine the encoder and the header
        self.teacher = nn.Sequential(self._teacher_net, self.teacher_head).to(self.device)
        self.student = nn.Sequential(self._student_net, self.student_head).to(self.device)

        self.tbotloss = TBotLoss()

        self.after_epoch_callback = after_epoch_callback
        self.n_epochs = 0

        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)

    def ema_update(self, alpha=0.99):
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = alpha * param_t.data + (1 - alpha) * param_s.data

    def fit(self, train_data, n_epochs=None, verbose=False):
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
                for param in self.teacher.parameters():
                    param.requires_grad = False
                self.optimizer.zero_grad()
                
                # copy the student weights to the teacher
                self.teacher.load_state_dict(self.student.state_dict())

                x = batch.to(self.device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
               
                def take_per_row(A, indx, num_elem):
                    all_indx = indx[:,None] + np.arange(num_elem)
                    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

                out1 = take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft)
                # add padding to 24
                out1 = F.pad(out1, (0, 0, 0, 24 - out1.size(1)))
                # add learnable cls token at the beginning
                out1 = F.pad(out1, (0, 0, 1, 0))
                # out1 = out1[:, -crop_l:]
                out1_masked, mask1 = self._student_net.mask(out1)

                out2 = take_per_row(x, crop_offset + crop_left, crop_eright - crop_left)
                # add padding to 24
                out2 = F.pad(out2, (0, 0, 0, 24 - out2.size(1)))
                # add learnable cls token at the beginning
                out2 = F.pad(out2, (0, 0, 1, 0))
                # out2 = out2[:, :crop_l]
                out2_masked, mask2 = self._student_net.mask(out2)

                assert out1.size() == out2.size()
                # print(out1.size(), out2.size())

                teacher_out1 = self.teacher(out1)
                teacher_out2 = self.teacher(out2)
                student_out1 = self.student(out1_masked)
                student_out2 = self.student(out2_masked)

                loss = self.tbotloss(student_out1, teacher_out1, 
                                    student_out2, teacher_out2,
                                    mask1, mask2)
                
                loss.mean().backward()
                self.optimizer.step()
                self.ema_update()

            print(f'Epoch {self.n_epochs+1} loss: {loss.mean().item()}')
            self.n_epochs += 1

        return loss


"""
记录
python -u main.py ERing UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42
Dataset: ERing
Arguments: Namespace(batch_size=8, dataset='ERing', epochs=40, eval=False, gpu=0, loader='UEA', lr=0.001, max_threads=8, repr_dims=320, run_name='UEA', save_every=None, seed=42)
cuda:0
Loading data... done
torch.Size([8, 6, 320]) torch.Size([8, 6, 320])
torch.Size([8, 37, 320]) torch.Size([8, 37, 320])
torch.Size([8, 50, 320]) torch.Size([8, 50, 320])
Finished.
"""