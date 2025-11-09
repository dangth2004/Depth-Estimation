from pyexpat import features

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features

    def forward(self, x):
        return self.mobilenet(x)
