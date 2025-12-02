import torch
import torch.nn as nn

from .prn_decoder import SEFusion


class LinearHead(nn.Module):

    def __init__(
        self,
        in_ch: int,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.fusion = SEFusion(in_ch)

        self.proj = nn.Conv2d(in_ch, n_classes, 1)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=False)

    def forward(self, x, y=None):  # fmap: [B, C, H, W]（步长16）
        if y is None:
            return self.up(self.proj(x))  # 输入分辨率的logits
        else:
            return self.up(self.proj(self.fusion(x, y)))
