import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

class TBotLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_out1, teacher_out1, 
                    student_out2, teacher_out2, 
                    student_mask1, student_mask2): # train by epoch
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student1_cls, student1_patch = student_out1 # student1_cls: [B, C], student1_patch: [B, T, C]
        student2_cls, student2_patch = student_out2
        teacher1_cls, teacher1_patch = teacher_out1 # teacher1_cls: [B, C], teacher1_patch: [B, T, C]
        teacher2_cls, teacher2_patch = teacher_out2

        # contrastive loss for student1_cls and teacher2_cls
        loss_cls_1 = torch.sum(-teacher1_cls * F.log_softmax(student2_cls, dim=1), dim=-1)
        # contrastive loss for student2_cls and teacher1_cls
        loss_cls_2 = torch.sum(-teacher2_cls * F.log_softmax(student1_cls, dim=1), dim=-1)
        loss_cls = (loss_cls_1 + loss_cls_2) / 2

        # contrastive loss for student1_patch and teacher1_patch
        # sum over dimension 1
        loss_patch_1 = torch.sum(-teacher1_patch * F.log_softmax(student1_patch, dim=1), dim=-1)
        loss_patch_1 = torch.sum(loss_patch_1, dim=1)
        # loss_patch_1 = torch.sum(loss_patch_1 * student_mask1, dim=1)
        # loss_patch_1 = torch.sum(loss_patch_1) / torch.sum(student_mask1)

        loss_patch_2 = torch.sum(-teacher2_patch * F.log_softmax(student2_patch, dim=1), dim=-1)
        loss_patch_2 = torch.sum(loss_patch_2, dim=1)
        # loss_patch_2 = torch.sum(loss_patch_2 * student_mask2, dim=1)
        # loss_patch_2 = torch.sum(loss_patch_2) / torch.sum(student_mask2)
        loss_patch = (loss_patch_1 + loss_patch_2) / 2
            
        loss = loss_cls + loss_patch

        return loss
    
# add a log
# python -u main.py ERing UEA --loader UEA --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42
# Dataset: ERing
# Arguments: Namespace(batch_size=8, dataset='ERing', epochs=40, eval=False, gpu=0, loader='UEA', lr=0.001, max_threads=8, repr_dims=320, run_name='UEA', save_every=None, seed=42)
# cuda:0
# Loading data... done
# Epoch 1 loss: 15.088810920715332
# Epoch 2 loss: -6.889143466949463
# Epoch 3 loss: -2.575761318206787
# Epoch 4 loss: 0.7212203145027161
# Epoch 5 loss: -18.003435134887695
# Epoch 6 loss: -6.908088684082031
# Epoch 7 loss: -26.769227981567383
# Epoch 8 loss: -5.729512691497803
# Epoch 9 loss: -23.660396575927734
# Epoch 10 loss: -14.486408233642578
# Epoch 11 loss: -6.4370951652526855
# Epoch 12 loss: 14.927998542785645
# Epoch 13 loss: 19.904356002807617
# Epoch 14 loss: 26.191434860229492
# Epoch 15 loss: -4.921065330505371
# Epoch 16 loss: -5.744189262390137
# Epoch 17 loss: -32.732269287109375
# Epoch 18 loss: 2.110478401184082
# Epoch 19 loss: -16.34660530090332
# Epoch 20 loss: -1.9561625719070435
# Epoch 21 loss: -34.58295440673828
# Epoch 22 loss: -7.064224720001221
# Epoch 23 loss: -28.78524398803711
# Epoch 24 loss: -21.57451057434082
# Epoch 25 loss: 74.02960968017578
# Epoch 26 loss: 15.768782615661621
# Epoch 27 loss: -42.40218734741211
# Epoch 28 loss: -15.369378089904785
# Epoch 29 loss: -65.58644104003906
# Epoch 30 loss: 6.055628299713135
# Epoch 31 loss: 128.15724182128906
# Epoch 32 loss: 56.72381591796875
# Epoch 33 loss: -32.94186782836914
# Epoch 34 loss: -41.28746795654297
# Epoch 35 loss: -19.21023941040039
# Epoch 36 loss: 26.030311584472656
# Epoch 37 loss: -97.62760162353516
# Epoch 38 loss: -51.95337677001953
# Epoch 39 loss: -41.578250885009766
# Epoch 40 loss: -17.401899337768555
# Finished.