import torch
import torch.nn as nn

from utils.HDPA import HDPA
from utils.InverseConv2DLayer import InverseConv2DLayer


class LinearHead(nn.Module):

    def __init__(
        self,
        in_ch: int,
        n_classes: int = 1000,
    ):
        super().__init__()

        # self.hdpa = HDPA(in_ch)
        self.inv_conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1),
            nn.GroupNorm(16, in_ch // 2),
            nn.GELU(),
            InverseConv2DLayer(in_channels=in_ch // 2,
                               out_channels=in_ch // 2,
                               kernel_size=5,
                               scale=2),
        )

        self.inv_conv_2 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch // 4, 3, padding=1),
            nn.GroupNorm(16, in_ch // 4),
            nn.GELU(),
            InverseConv2DLayer(in_channels=in_ch // 4,
                               out_channels=in_ch // 4,
                               kernel_size=5,
                               scale=2),
        )

        self.inv_conv_3 = nn.Sequential(
            nn.Conv2d(in_ch // 4, in_ch // 8, 3, padding=1),
            nn.GroupNorm(16, in_ch // 8),
            nn.GELU(),
            InverseConv2DLayer(in_channels=in_ch // 8,
                               out_channels=in_ch // 8,
                               kernel_size=5,
                               scale=2),
        )

        self.proj = nn.Conv2d(in_ch // 8, n_classes, 1)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=False)

    def forward(self, fmap):  # fmap: [B, C, H, W]（步长16）
        # fmap = self.hdpa(fmap)
        fmap = self.inv_conv_1(fmap)
        fmap = self.inv_conv_2(fmap)
        fmap = self.inv_conv_3(fmap)
        return self.up(self.proj(fmap))  # 输入分辨率的logits
