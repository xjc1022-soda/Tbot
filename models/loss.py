import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

class TBotLoss(nn.Module):
    def __init__(self, student_temp=2, teacher_temp=5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp


    #  center is not used in the cross_entropy function
    def cross_entropy(self, student, teacher, center=None):
        teacher = teacher.detach()
        student = F.log_softmax(student / self.student_temp, dim=1)
        teacher = F.softmax(teacher / self.teacher_temp - center, dim=1)
        return torch.sum(- student * teacher, dim=-1)


    def forward(self, student_out1, teacher_out1, 
                    student_out2, teacher_out2, 
                    student_mask1, student_mask2, cls_center, patch_center):

        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student1_cls, student1_patch = student_out1 # student1_cls: [B, C], student1_patch: [B, T, C]
        student2_cls, student2_patch = student_out2
        teacher1_cls, teacher1_patch = teacher_out1 # teacher1_cls: [B, C], teacher1_patch: [B, T, C]
        teacher2_cls, teacher2_patch = teacher_out2
        
        # cls_center and patch_center dont need grad
        cls_center = cls_center.detach()
        patch_center = patch_center.detach()

        if torch.all(cls_center == torch.zeros_like(cls_center)) and torch.all(patch_center == torch.zeros_like(patch_center)):
            cls_center = torch.cat([teacher1_cls, teacher2_cls], dim=0).mean(dim=0).to(student1_cls.device)
            patch_center = torch.cat([teacher1_patch, teacher2_patch], dim=0).mean(dim=1).mean(dim=0).to(student1_patch.device)



        loss_cls_1 = self.cross_entropy(student2_cls, teacher1_cls, cls_center)
        loss_cls_2 = self.cross_entropy(student1_cls, teacher2_cls, cls_center)

        loss_cls = (loss_cls_1 + loss_cls_2) / 2

        if torch.sum(student_mask1) == 0 or torch.sum(student_mask2) == 0:
            loss_patch = torch.tensor(0.0).to(student1_patch.device)
        
        elif torch.sum(student_mask1) == 0:
            loss_patch_2 = self.cross_entropy(student2_patch, teacher2_patch, patch_center)
            loss_patch_2 = torch.sum(loss_patch_2 * student_mask2, dim=1)
            loss_patch = loss_patch_2 / torch.sum(student_mask2, dim=1)
        
        elif torch.sum(student_mask2) == 0:
            loss_patch_1 = self.cross_entropy(student1_patch, teacher1_patch, patch_center)
            loss_patch_1 = torch.sum(loss_patch_1 * student_mask1, dim=1)
            loss_patch = loss_patch_1 / torch.sum(student_mask1, dim=1)
        
        else:
            loss_patch_1 = self.cross_entropy(student1_patch, teacher1_patch, patch_center)
            loss_patch_1 = torch.sum(loss_patch_1 * student_mask1, dim=1)
            loss_patch_1 = loss_patch_1 / torch.sum(student_mask1, dim=1)

            loss_patch_2 = self.cross_entropy(student2_patch, teacher2_patch, patch_center)
            loss_patch_2 = torch.sum(loss_patch_2 * student_mask2, dim=1)
            loss_patch_2 = loss_patch_2 / torch.sum(student_mask2, dim=1)

            print(loss_patch_1, loss_patch_2)

            loss_patch = (loss_patch_1 + loss_patch_2) / 2
        
        # whether it is fair to average the loss by 2
        cls_center = cls_center * 0.9 + torch.cat([teacher1_cls, teacher2_cls], dim=0).mean(dim=0) * 0.1
        patch_center = patch_center * 0.9 + torch.cat([teacher1_patch, teacher2_patch], dim=0).mean(dim=1).mean(dim=0) * 0.1

        return loss_cls , loss_patch, cls_center, patch_center
    

def DistillationLoss(student_out, teacher_out):
    mse = nn.MSELoss()
    loss = 0 
    for i in range(len(student_out)):
        loss += mse(student_out[i], teacher_out[i])
    return loss

def ReconstructionLoss(mask_patch, pred_list, merger):
    mse = nn.MSELoss()
    loss = 0 
    for pred in pred_list:
        loss += mse(mask_patch, pred)
        mask_patch = merger(mask_patch)
    return loss
