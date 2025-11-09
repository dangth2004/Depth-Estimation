import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def depthwise(in_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    assert 2 * padding == kernel_size - 1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, bias=False, groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
    )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def output_layer(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )


class NNConv(nn.Module):
    def __init__(self, in_channels=1024, kernel_size=3, dw='dw'):
        super(NNConv, self).__init__()
        self.max_depth = 10.0
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(in_channels, kernel_size),
                pointwise(in_channels, 512))
            self.conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.conv6 = output_layer(32, 1)
        else:
            self.conv1 = conv(1024, 512, kernel_size)
            self.conv2 = conv(512, 256, kernel_size)
            self.conv3 = conv(256, 128, kernel_size)
            self.conv4 = conv(128, 64, kernel_size)
            self.conv5 = conv(64, 32, kernel_size)
            self.conv6 = output_layer(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv6(x)
        return x
