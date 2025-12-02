import torch
import torch.nn as nn


class LinearHead(nn.Module):

    def __init__(
        self,
        in_ch: int,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_ch, n_classes, 1)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=False)

    def forward(self, fmap):  # fmap: [B, C, H, W]（步长16）
        return self.up(self.proj(fmap))  # 输入分辨率的logits
