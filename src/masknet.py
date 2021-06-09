import torch
import torch.nn as nn


class MaskNet(nn.Module):
    def __init__(self, in_channels=3):
        super(MaskNet, self).__init__()
        self.darknet53 = Darknet53(in_channels)
        self.bbox = nn.Linear(1024, 4)
        self.mask = nn.Sequential(
            nn.Linear(1024 + 4, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        features = self.darknet53(image)
        bbox = self.bbox(features)
        mask = self.mask(torch.cat([features, bbox], dim=1)).squeeze()
        return bbox, mask


class Darknet53(nn.Module):
    def __init__(self, in_channels=3):
        super(Darknet53, self).__init__()
        self.conv1 = Darknet53.dark_conv(in_channels, 32)
        self.conv2 = Darknet53.dark_conv(32, 64, stride=2)
        self.residual1 = Darknet53.residual(64, 1)
        self.conv3 = Darknet53.dark_conv(64, 128, stride=2)
        self.residual2 = Darknet53.residual(128, 2)
        self.conv4 = Darknet53.dark_conv(128, 256, stride=2)
        self.residual3 = Darknet53.residual(256, 8)
        self.conv5 = Darknet53.dark_conv(256, 512, stride=2)
        self.residual4 = Darknet53.residual(512, 8)
        self.conv6 = Darknet53.dark_conv(512, 1024, stride=2)
        self.residual5 = Darknet53.residual(1024, 4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual1(out)
        out = self.conv3(out)
        out = self.residual2(out)
        out = self.conv4(out)
        out = self.residual3(out)
        out = self.conv5(out)
        out = self.residual4(out)
        out = self.conv6(out)
        out = self.residual5(out)
        out = self.global_avg_pool(out)
        return out.view(out.size(0), 1024)

    @staticmethod
    def dark_conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    @staticmethod
    def residual(in_channels, n_residual):
        return nn.Sequential(*[DarkResidualBlock(in_channels) for _ in range(n_residual)])


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        out_channels = in_channels // 2
        self.layer1 = Darknet53.dark_conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.layer2 = Darknet53.dark_conv(out_channels, in_channels)

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x
