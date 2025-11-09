import torch
import torch.nn as nn

from models.backbone import Backbone
from models.depth_decoder import NNConv


class LSDANet(nn.Module):
    def __init__(self):
        super(LSDANet, self).__init__()
        self.backbone = Backbone()
        self.decoder = NNConv(in_channels=576, kernel_size=3, dw='dw')

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x
