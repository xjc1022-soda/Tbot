import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

class TBotLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy(self):
        pass

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