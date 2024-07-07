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
# (ts2vec) ➜  self-distillation-time-series-learning git:(main) ✗ python -u main.py ERing UEA --loader UEA --batch-size 8 --repr-dims 64 --max-threads 8 --seed 42
# Dataset: ERing
# Arguments: Namespace(batch_size=8, dataset='ERing', epochs=40, eval=False, gpu=0, loader='UEA', lr=0.001, max_threads=8, repr_dims=64, run_name='UEA', save_every=None, seed=42)
# cuda:0
# Loading data... done
# Epoch 1 loss: -2.797661781311035
# Epoch 2 loss: 1.3645538091659546
# Epoch 3 loss: -1.607981562614441
# Epoch 4 loss: 3.587738037109375
# Epoch 5 loss: -2.515519618988037
# Epoch 6 loss: -4.065597057342529
# Epoch 7 loss: 3.7022693157196045
# Epoch 8 loss: 2.640960693359375
# Epoch 9 loss: -10.208735466003418
# Epoch 10 loss: -6.974784851074219
# Epoch 11 loss: -3.786545991897583
# Epoch 12 loss: -7.514355182647705
# Epoch 13 loss: -1.9308971166610718
# Epoch 14 loss: -25.068405151367188
# Epoch 15 loss: -3.202470064163208
# Epoch 16 loss: 7.651269912719727
# Epoch 17 loss: 20.949249267578125
# Epoch 18 loss: -8.607158660888672
# Epoch 19 loss: 4.7553229331970215
# Epoch 20 loss: 2.1125221252441406
# Epoch 21 loss: -6.21744966506958
# Epoch 22 loss: 14.087311744689941
# Epoch 23 loss: 7.840858459472656
# Epoch 24 loss: 9.254195213317871
# Epoch 25 loss: -6.784298419952393
# Epoch 26 loss: -2.130847692489624
# Epoch 27 loss: -5.9547648429870605
# Epoch 28 loss: -7.950315952301025
# Epoch 29 loss: -15.204230308532715
# Epoch 30 loss: 13.998777389526367
# Epoch 31 loss: 10.91220474243164
# Epoch 32 loss: -13.378812789916992
# Epoch 33 loss: -0.3950973451137543
# Epoch 34 loss: -20.930023193359375
# Epoch 35 loss: -16.056617736816406
# Epoch 36 loss: 7.598066806793213
# Epoch 37 loss: -12.6756591796875
# Epoch 38 loss: -1.201253056526184
# Epoch 39 loss: -8.209927558898926
# Epoch 40 loss: -19.077362060546875
# Finished.