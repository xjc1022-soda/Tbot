import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

class TBotLoss(nn.Module):
    def __init__(self, student_temp=2, teacher_temp=5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def cross_entropy(self, student, teacher):
        student = student / self.student_temp
        teacher = teacher / self.teacher_temp
        # print(teacher < 0)
        # print((F.log_softmax(student, dim=1) > 0).sum())
        return torch.sum(-teacher * F.log_softmax(student, dim=1), dim=-1)

    def forward(self, student_out1, teacher_out1, 
                    student_out2, teacher_out2, 
                    student_mask1, student_mask2):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student1_cls, student1_patch = student_out1 # student1_cls: [B, C], student1_patch: [B, T, C]
        student2_cls, student2_patch = student_out2
        teacher1_cls, teacher1_patch = teacher_out1 # teacher1_cls: [B, C], teacher1_patch: [B, T, C]
        teacher2_cls, teacher2_patch = teacher_out2

        # contrastive loss for student1_cls and teacher2_cls
        loss_cls_1 = self.cross_entropy(student2_cls, teacher1_cls)
        # print(f'cross entropy is {loss_cls_1}')
        # contrastive loss for student2_cls and teacher1_cls
        loss_cls_2 = self.cross_entropy(student1_cls, teacher2_cls)
        loss_cls = (loss_cls_1 + loss_cls_2) / 2

        if torch.sum(student_mask1) == 0 or torch.sum(student_mask2) == 0:
            loss_patch = torch.tensor(0.0, device=student1_cls.device)
            return loss_cls + loss_patch
        
        elif torch.sum(student_mask1) == 0:
            loss_patch_2 = self.cross_entropy(student2_patch, teacher2_patch)
            loss_patch_2 = torch.sum(loss_patch_2 * student_mask2, dim=1)
            loss_patch_2 = loss_patch_2 / torch.sum(student_mask2, dim=1)
            return loss_cls + loss_patch_2
        
        elif torch.sum(student_mask2) == 0:
            loss_patch_1 = self.cross_entropy(student1_patch, teacher1_patch)
            loss_patch_1 = torch.sum(loss_patch_1 * student_mask1, dim=1)
            loss_patch_1 = loss_patch_1 / torch.sum(student_mask1, dim=1)
            return loss_cls + loss_patch_1
        
        else:
            loss_patch_1 = self.cross_entropy(student1_patch, teacher1_patch)
            loss_patch_1 = torch.sum(loss_patch_1 * student_mask1, dim=1)
            loss_patch_1 = loss_patch_1 / torch.sum(student_mask1, dim=1)

            loss_patch_2 = self.cross_entropy(student2_patch, teacher2_patch)
            loss_patch_2 = torch.sum(loss_patch_2 * student_mask2, dim=1)
            loss_patch_1 = loss_patch_1 / torch.sum(student_mask2, dim=1)

            loss_patch = (loss_patch_1 + loss_patch_2) / 2
            return loss_cls + loss_patch