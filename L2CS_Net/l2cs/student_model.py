import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from l2cs.model import L2CS

class SlimStudentL2CS(nn.Module):
    def __init__(self, num_bins=90):
        super(SlimStudentL2CS, self).__init__()
        self.model = L2CS(BasicBlock, [1,1,1,1],  num_bins)

    def forward(self, x):
        return self.model(x)