import torch
import torch.nn as nn


class MaskNet(nn.Module):
    def __init__(self):
        super(MaskNet, self).__init__()
        self.features = darknet53()
        self.bbox = nn.Sequential(
            nn.Conv2d(1024, 5, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def forward(self, image: torch.Tensor):
        features = self.features(image)
        out = self.bbox(features).flatten(start_dim=1)
        mask = out[:, 0]
        bbox = out[:, 1:] * image.size(2)
        bbox[:, :2] = bbox[:, :2] - bbox[:, 2:] / 2
        return bbox, mask


def darknet_conv(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        out_channels = in_channels // 2
        self.layer1 = darknet_conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.layer2 = darknet_conv(out_channels, in_channels)

    def forward(self, x):
        return self.layer2(self.layer1(x)) + x


def darknet_residual(in_channels, n_residual):
    return nn.Sequential(*[Bottleneck(in_channels) for _ in range(n_residual)])


def darknet53():
    return nn.Sequential(
        darknet_conv(3, 32),
        darknet_conv(32, 64, stride=2),
        darknet_residual(64, 1),
        darknet_conv(64, 128, stride=2),
        darknet_residual(128, 2),
        darknet_conv(128, 256, stride=2),
        darknet_residual(256, 8),
        darknet_conv(256, 512, stride=2),
        darknet_residual(512, 8),
        darknet_conv(512, 1024, stride=2),
        darknet_residual(1024, 4)
    )
