import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.InverseConv2DLayer import InverseConv2DLayer
from utils.FrequencyChannelAttention import FrequencyChannelAttention
from utils.HaarFusion import Fusion as HaarFusion
from utils.GCBAM import GCBAM
from utils.Net import FeatureReinforcementModule, Decoder


class ConvBNReLU(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 1,
    ):
        layers = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      pad,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class FPNDecoder(nn.Module):
    """The Feature Pyramid Network (FPN) decoder is used as a module for
    decoders in image segmentation to do multiscale spatial pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        # inter_layers: int,
        patch_h: int = 16,
        patch_w: int = 16,
        n_classes: int = 100,
    ):
        super().__init__()
        self.width = patch_w
        self.height = patch_h
        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        # self.inter_layers = inter_layers

        inner_channels = [
            out_channels,
            out_channels // 2,
            out_channels // 4,
            out_channels // 8,
        ]

        # self.upsample = nn.Upsample(scale_factor=2)

        # FPN Module
        # self.conv1 = ConvBNReLU(in_channels, out_channels, 3)
        self.conv2 = ConvBNReLU(out_channels, inner_channels[1], 3)
        self.conv3 = ConvBNReLU(inner_channels[1], inner_channels[2], 3)
        self.conv4 = ConvBNReLU(inner_channels[2], inner_channels[3], 3)
        self.conv5 = ConvBNReLU(inner_channels[3], n_classes, 1, pad=0)

        self.fusion1 = HaarFusion(out_channels, wave='haar')
        self.fusion2 = HaarFusion(inner_channels[1], wave='haar')
        self.fusion3 = HaarFusion(inner_channels[2], wave='haar')

        self.gcbam0 = GCBAM(in_channels)
        # self.gcbam1 = GCBAM(out_channels)
        # self.gcbam2 = GCBAM(inner_channels[1])
        # self.gcbam3 = GCBAM(inner_channels[2])
        # self.gcbam4 = GCBAM(inner_channels[3])

        # Intermediate layers
        # self.inter_conv0 = nn.Sequential(
        #     nn.Conv2d(in_channels, inner_channels[0], 1),
        #     # FrequencyChannelAttention(inner_channels[1]),
        # )
        self.inter_conv1 = nn.Sequential(
            nn.Conv2d(out_channels, inner_channels[1], 1),
            # FrequencyChannelAttention(inner_channels[1]),
        )
        self.inter_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, inner_channels[2], 1),
            # FrequencyChannelAttention(inner_channels[2]),
        )
        # self.inter_conv3 = nn.Sequential(
        #     nn.Conv2d(out_channels, inner_channels[3], 1),
        #     # FrequencyChannelAttention(inner_channels[3]),
        # )

        # InverseConv2DLayer
        # self.inv_conv = InverseConv2DLayer(in_channels=inner_channels[3],
        #                                    out_channels=inner_channels[3],
        #                                    kernel_size=5,
        #                                    scale=2)
        # self.inv_conv_2 = InverseConv2DLayer(in_channels=inner_channels[1],
        #                                      out_channels=inner_channels[1],
        #                                      kernel_size=5,
        #                                      scale=2)
        # self.inv_conv_3 = InverseConv2DLayer(in_channels=inner_channels[2],
        #                                      out_channels=inner_channels[2],
        #                                      kernel_size=5,
        #                                      scale=2)

        self.frm = FeatureReinforcementModule([in_channels] * 5, out_channels)
        self.decoder = Decoder(out_channels)

    # def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    #     x = features[-1]

    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     # x = self.upsample(x)
    #     x = self.inv_conv_2(x)  # 上采样替换为反卷积

    #     inter_fpn = self.inter_conv1(features[0])
    #     x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
    #     x = self.conv3(x)
    #     # x = self.upsample(x)
    #     x = self.inv_conv_3(x)  # 上采样替换为反卷积

    #     inter_fpn = self.inter_conv2(features[1])
    #     x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")
    #     x = self.conv4(x)
    #     # x = self.inv_conv(x)  # 使用反卷积增强特征 & 提升输出的分辨率

    #     inter_fpn = self.inter_conv3(features[2])
    #     x = x + F.interpolate(inter_fpn, size=x.shape[-2:], mode="nearest")

    #     x = self.inv_conv(x)
    #     return self.conv5(x)

    # def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    #     for i, feature in enumerate(features):
    #         features[i] = self.gcbam0(feature)

    #     x = features[-1]

    #     x = self.conv1(x)  # 256
    #     # x = self.gcbam1(x)

    #     x = self.fusion1(self.inter_conv0(features[-2]), x)  # 256
    #     x = self.conv2(x)
    #     # x = self.gcbam2(x)
    #     x = self.fusion2(self.inter_conv1(features[-3]), x)  # 128
    #     x = self.conv3(x)
    #     # x = self.gcbam3(x)
    #     x = self.fusion3(self.inter_conv2(features[-4]), x)  # 64
    #     x = self.conv4(x)
    #     # x = self.gcbam4(x)

    #     return self.conv5(x)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # for i, feature in enumerate(features):
        #     features[i] = self.gcbam0(feature)

        features = self.frm(*features)

        x = features[-1]

        # x = self.conv1(x)  # 256
        # x = self.gcbam1(x)

        x = self.fusion1(features[-2], x)  # 256
        x = self.conv2(x)
        # x = self.gcbam2(x)
        x = self.fusion2(self.inter_conv1(features[-3]), x)  # 128
        x = self.conv3(x)
        # x = self.gcbam3(x)
        x = self.fusion3(self.inter_conv2(features[-4]), x)  # 64
        x = self.conv4(x)
        # x = self.gcbam4(x)

        return self.conv5(x)

    # def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
    #     features = self.frm(*features)

    #     gc = self.fusion1(features[-2], features[-1])
    #     m_1, m_2, m_3 = self.decoder(*features, gc)

    #     return m_3
