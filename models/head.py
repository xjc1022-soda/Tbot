import torch
import torch.nn as nn

class TBotHead(nn.Module):
    def __init__(self, output_dims, cls_dims, patch_dims, device):
        super().__init__()
        self.output_dims = output_dims
        self.bottleneck_dims = 128
        self.cls_dims = cls_dims
        self.patch_dims = patch_dims
        self.device = device

    def forward(self, x):
        # common first two layers for cls and patch
        head = nn.Sequential(
            nn.Linear(self.output_dims, self.bottleneck_dims),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dims, self.bottleneck_dims),
            nn.ReLU(),
            nn.Linear(self.bottleneck_dims, self.output_dims),
        ).to(self.device)

        # patch_head = nn.Sequential(
        #     nn.Linear(self.output_dims, self.bottleneck_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.bottleneck_dims, self.bottleneck_dims),
        #     nn.ReLU(),
        #     nn.Linear(self.bottleneck_dims, self.output_dims),
        # ).to(self.device)

        x_cls = head(x[:,0,:])
        x_patch = []
        for i in range(1, x.size(1)):
            x_patch.append(head(x[:,i,:]))
        x_patch = torch.stack(x_patch, dim=1)

        return x_cls, x_patch

