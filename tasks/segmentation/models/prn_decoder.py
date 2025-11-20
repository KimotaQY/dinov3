import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.Net import FeatureReinforcementModule
from utils.HaarFusion import Fusion as HaarFusion
from utils.GCBAM import GCBAM
from utils.PKIBlock import Poly_Kernel_Inception_Block
from utils.CBAM import CBAM


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


class ProgressiveRefinementNeck(nn.Module):

    def __init__(self, channels_list, num_stages=1):
        """
        PRN模块初始化 - 对应论文中的P2-P5三个尺度
        
        Args:
            channels_list: 各层级特征图的通道数列表 [C2, C3, C4, C5]
            num_stages: 渐进精化阶段数
        """
        super(ProgressiveRefinementNeck, self).__init__()
        self.num_stages = num_stages

        # 初始特征融合卷积 (对应公式1)
        self.td_convs = nn.ModuleList([
            nn.Conv2d(channels_list[i + 1] + channels_list[i],
                      channels_list[i],
                      3,
                      padding=1) for i in range(len(channels_list) - 1)
        ])

        # 主干特征重用机制的卷积层
        self.reuse_convs = nn.ModuleList([
            nn.Conv2d(channels_list[0] + channels_list[1],
                      channels_list[1],
                      3,
                      padding=1),  # P2->P3
            nn.Conv2d(channels_list[1] * 2 + channels_list[0],
                      channels_list[0],
                      3,
                      padding=1)  # P3->P2
        ])

        # 输出卷积层
        # self.out_convs = nn.ModuleList([
        #     nn.Conv2d(channels_list[i], channels_list[i], 3, padding=1)
        #     for i in range(len(channels_list))
        # ])
        self.out_convs_2 = nn.Conv2d(channels_list[0] * 2 + channels_list[1],
                                     channels_list[0],
                                     3,
                                     padding=1)
        self.out_convs_3 = nn.Conv2d(channels_list[1] + channels_list[2],
                                     channels_list[1],
                                     3,
                                     padding=1)
        self.out_convs_4 = nn.Conv2d(channels_list[2],
                                     channels_list[2],
                                     3,
                                     padding=1)

    def resize(self, x, size):
        """分辨率匹配操作"""
        return F.interpolate(x, size=size, mode='nearest')

    def forward(self, backbone_features):
        """
        PRN前向传播 - 严格对应论文中的P2-P5流程
        
        Args:
            backbone_features: 主干网络特征 [P2_in, P3_in, P4_in, P5_in]
            
        Returns:
            output_features: 输出特征 [P2_out, P3_out, P4_out]
        """
        P2_in, P3_in, P4_in, P5_in = backbone_features

        # === 步骤1: 初始特征融合 (公式1) ===
        # 自上而下路径: P5 -> P4 -> P3 -> P2
        P5_td = P5_in
        P4_td = self.td_convs[2](torch.cat(
            [self.resize(P5_td, P4_in.shape[-2:]), P4_in], dim=1))
        P3_td = self.td_convs[1](torch.cat(
            [self.resize(P4_td, P3_in.shape[-2:]), P3_in], dim=1))
        P2_td = self.td_convs[0](torch.cat(
            [self.resize(P3_td, P2_in.shape[-2:]), P2_in], dim=1))

        # === 步骤2: 主干特征重用和渐进融合 ===
        P2_refine = P2_td

        for stage in range(self.num_stages):
            # 下采样P2_td并与P3_in拼接 (对应公式2)
            P3_td1 = self.reuse_convs[0](torch.cat(
                [self.resize(P2_refine, P3_in.shape[-2:]), P3_in], dim=1))

            # 上采样P3_td1并与P2_td、P2_in拼接 (对应公式3)
            P2_refine = self.reuse_convs[1](torch.cat(
                [self.resize(P3_td1, P2_in.shape[-2:]), P2_td, P2_in], dim=1))

        # === 步骤3: 输出生成 (对应公式4-6) ===
        # P4_out: 从精化后的P2下采样得到
        P4_out = self.out_convs_4(self.resize(P2_refine, P4_in.shape[-2:]))

        # P3_out: 融合P4_out和原始P3_in
        P3_out = self.out_convs_3(
            torch.cat([self.resize(P4_out, P3_in.shape[-2:]), P3_in], dim=1))

        # P2_out: 融合P3_out、精化P2和原始P2_in
        P2_out = self.out_convs_2(
            torch.cat(
                [self.resize(P3_out, P2_in.shape[-2:]), P2_refine, P2_in],
                dim=1))

        return [P2_out, P3_out, P4_out]


class PRNDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_classes):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        # self.inter_layers = inter_layers

        inner_channels = [
            out_channels,
            out_channels // 2,
            out_channels // 4,
            out_channels // 8,
        ]

        # TODO: change the input channels
        self.frm = FeatureReinforcementModule([in_channels[0]] + in_channels,
                                              out_channels)

        self.neck = ProgressiveRefinementNeck(channels_list=[out_channels] * 4,
                                              num_stages=1)

        # self.gcbam0 = GCBAM(out_channels, group=1)
        self.cbam = CBAM(out_channels)

        # self.pki_block = Poly_Kernel_Inception_Block(out_channels,
        #                                              out_channels)

        # self.upsample = nn.Upsample(scale_factor=2)

        # self.conv2 = ConvBNReLU(out_channels, inner_channels[1], 3)
        # self.conv3 = ConvBNReLU(inner_channels[1], inner_channels[2], 3)
        # self.conv4 = ConvBNReLU(inner_channels[2], inner_channels[3], 3)
        # # self.conv5 = ConvBNReLU(inner_channels[3], n_classes, 1, pad=0)

        # self.fusion1 = HaarFusion(out_channels, wave='haar')
        # self.fusion2 = HaarFusion(inner_channels[1], wave='haar')
        # self.fusion3 = HaarFusion(inner_channels[2], wave='haar')

        # self.inter_conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels, inner_channels[1], 1),
        #     # FrequencyChannelAttention(inner_channels[2]),
        # )
        # self.inter_conv3 = nn.Sequential(
        #     nn.Conv2d(out_channels, inner_channels[2], 1),
        #     # FrequencyChannelAttention(inner_channels[2]),
        # )

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, features):
        features = self.frm(*features)

        # features = [self.pki_block(f) for f in features]

        # features = [self.gcbam0(f) for f in features]
        features = [self.cbam(f) for f in features]

        p2, p3, p4 = self.neck(features)

        # 提升输出结果分辨率
        # x = self.fusion1(p3, p4)
        # x = self.conv2(x)
        # x = self.fusion2(self.inter_conv2(p2), x)
        # x = self.conv3(x)
        # x = self.conv4(self.inter_conv3(features[0]) + x)

        # x = self.fusion3(x, self.conv4(features[-1]))

        # x = features[-1]

        # x = self.fusion1(features[-2], x)  # 256
        # x = self.conv2(x)
        # x = self.fusion2(self.inter_conv1(features[-3]), x)  # 128
        # x = self.conv3(x)
        # x = self.fusion3(self.inter_conv2(features[-4]), x)  # 64
        # x = self.conv4(x)

        # return self.conv5(x)
        x = p2

        return self.out_conv(x)


# 使用示例 - 严格对应论文描述
if __name__ == "__main__":
    # 论文中实际使用的P2-P5三个尺度
    batch_size = 4
    channels = [256] * 4  # P2, P3, P4, P5通道数
    feature_sizes = [(320, 320), (160, 160), (80, 80), (40, 40)]  # 对应640输入的下采样

    P2 = torch.randn(batch_size, channels[0], *feature_sizes[0])
    P3 = torch.randn(batch_size, channels[1], *feature_sizes[1])
    P4 = torch.randn(batch_size, channels[2], *feature_sizes[2])
    P5 = torch.randn(batch_size, channels[3], *feature_sizes[3])

    prn = ProgressiveRefinementNeck(channels_list=channels, num_stages=1)
    output_features = prn([P2, P3, P4, P5])

    print("PRN输入输出对应关系:")
    print(f"输入: P2{P2.shape}, P3{P3.shape}, P4{P4.shape}")
    print(
        f"输出: P2{output_features[0].shape}, P3{output_features[1].shape}, P4{output_features[2].shape}"
    )
    # print(channels[::1])
