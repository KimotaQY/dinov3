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


class SqueezeAndExcitation(nn.Module):

    def __init__(self,
                 channel,
                 reduction=16,
                 activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,
                      kernel_size=1), activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SEFusion(nn.Module):

    def __init__(self,
                 channels_in,
                 activation=nn.ReLU(inplace=True),
                 num_modalities: int = 2):
        super(SEFusion, self).__init__()

        # 动态创建指定数量的SqueezeAndExcitation模块
        self.se_modules = nn.ModuleList([
            SqueezeAndExcitation(channels_in, activation=activation)
            for _ in range(num_modalities)
        ])

    def forward(self, *modalities):
        # 检查输入模态数量是否与初始化时定义的数量一致
        if len(modalities) != len(self.se_modules):
            raise ValueError(
                f"Expected {len(self.se_modules)} modalities, got {len(modalities)}"
            )

        # 对每个模态应用对应的SE模块并累加
        outputs = []
        for se_module, modality in zip(self.se_modules, modalities):
            outputs.append(se_module(modality))

        # 将所有处理后的模态相加
        out = sum(outputs)
        return out


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
            nn.Conv2d(channels_list[1] + channels_list[0] * 2,
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
        self.out_convs_4 = nn.Conv2d(channels_list[0],
                                     channels_list[2],
                                     3,
                                     padding=1)

    def resize(self, x, size):
        """分辨率匹配操作"""
        return F.interpolate(x, size=size, mode='nearest')

    def forward(self, backbone_features):
        """
        PRN前向传播 - 对应论文中的P2-P5流程
        
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
            resize_ = self.resize(P3_td1, P2_in.shape[-2:])
            cat_feature = torch.cat([resize_, P2_td, P2_in], dim=1)
            P2_refine = self.reuse_convs[1](cat_feature)
            # P2_refine = self.reuse_convs[1](torch.cat(
            #     [self.resize(P3_td1, P2_in.shape[-2:]), P2_td, P2_in], dim=1))

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


class Decoder(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        # TODO: change the input channels
        self.frm = FeatureReinforcementModule([in_channels[0]] + in_channels,
                                              out_channels)

        if num_modalities > 1:
            # 当有多个模态时，为每一层创建对应的SEFusion模块
            self.fusion1 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion2 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion3 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion4 = SEFusion(out_channels,
                                    num_modalities=num_modalities)

        self.neck = ProgressiveRefinementNeck(channels_list=[out_channels] * 4,
                                              num_stages=1)

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            x = modalities[0]
            features = self.frm(*x)
            p2, p3, p4 = self.neck(features)
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = self.frm(*modality)
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = self.fusion1(*all_x2)
            ff2 = self.fusion2(*all_x3)
            ff3 = self.fusion3(*all_x4)
            ff4 = self.fusion4(*all_x5)

            features = (ff1, ff2, ff3, ff4)

            p2, p3, p4 = self.neck(features)
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


class Decoder_FRM(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        # TODO: change the input channels
        self.frm = FeatureReinforcementModule([in_channels[0]] + in_channels,
                                              out_channels)

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            x = modalities[0]
            features = self.frm(*x)
            x1, x2, x3, x4 = features

            H, W = x1.shape[2:]
            p2 = x1
            for _x in [x2, x3, x4]:
                p2 = p2 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = self.frm(*modality)
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = sum(all_x2)
            ff2 = sum(all_x3)
            ff3 = sum(all_x4)
            ff4 = sum(all_x5)

            H, W = ff1.shape[2:]
            p2 = ff1
            for _x in [ff2, ff3, ff4]:
                p2 = p2 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


class Decoder_FRM_MMFF(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        # TODO: change the input channels
        self.frm = FeatureReinforcementModule([in_channels[0]] + in_channels,
                                              out_channels)

        if num_modalities > 1:
            # 当有多个模态时，为每一层创建对应的SEFusion模块
            self.fusion1 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion2 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion3 = SEFusion(out_channels,
                                    num_modalities=num_modalities)
            self.fusion4 = SEFusion(out_channels,
                                    num_modalities=num_modalities)

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            x = modalities[0]
            features = self.frm(*x)
            x1, x2, x3, x4 = features

            H, W = x1.shape[2:]
            p2 = x1
            for _x in [x2, x3, x4]:
                p2 = p2 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = self.frm(*modality)
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = self.fusion1(*all_x2)
            ff2 = self.fusion2(*all_x3)
            ff3 = self.fusion3(*all_x4)
            ff4 = self.fusion4(*all_x5)

            H, W = ff1.shape[2:]
            p2 = ff1
            for _x in [ff2, ff3, ff4]:
                p2 = p2 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


class Decoder_PRN(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        self.neck = ProgressiveRefinementNeck(channels_list=in_channels,
                                              num_stages=1)

        self.out_conv = ConvBNReLU(in_channels[0], n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            features = modalities[0]
            p2, p3, p4 = self.neck(features)
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = modality
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = sum(all_x2)
            ff2 = sum(all_x3)
            ff3 = sum(all_x4)
            ff4 = sum(all_x5)

            features = (ff1, ff2, ff3, ff4)

            p2, p3, p4 = self.neck(features)
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


class Decoder_PRN_MMFF(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        if num_modalities > 1:
            # 当有多个模态时，为每一层创建对应的SEFusion模块
            self.fusion1 = SEFusion(in_channels[0],
                                    num_modalities=num_modalities)
            self.fusion2 = SEFusion(in_channels[1],
                                    num_modalities=num_modalities)
            self.fusion3 = SEFusion(in_channels[2],
                                    num_modalities=num_modalities)
            self.fusion4 = SEFusion(in_channels[3],
                                    num_modalities=num_modalities)

        self.neck = ProgressiveRefinementNeck(channels_list=in_channels,
                                              num_stages=1)

        self.out_conv = ConvBNReLU(in_channels[0], n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            features = modalities[0]
            p2, p3, p4 = self.neck(features)
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = modality
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = self.fusion1(*all_x2)
            ff2 = self.fusion2(*all_x3)
            ff3 = self.fusion3(*all_x4)
            ff4 = self.fusion4(*all_x5)

            features = (ff1, ff2, ff3, ff4)

            p2, p3, p4 = self.neck(features)
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


class Decoder_MMFF(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
        num_modalities: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.num_modalities = num_modalities  # 存储模态数量

        if num_modalities > 1:
            # 当有多个模态时，为每一层创建对应的SEFusion模块
            self.fusion1 = SEFusion(in_channels[0],
                                    num_modalities=num_modalities)
            self.fusion2 = SEFusion(in_channels[1],
                                    num_modalities=num_modalities)
            self.fusion3 = SEFusion(in_channels[2],
                                    num_modalities=num_modalities)
            self.fusion4 = SEFusion(in_channels[3],
                                    num_modalities=num_modalities)

        self.conv_1 = ConvBNReLU(in_channels[0], out_channels, 1, pad=0)
        self.conv_2 = ConvBNReLU(in_channels[1], out_channels, 1, pad=0)
        self.conv_3 = ConvBNReLU(in_channels[2], out_channels, 1, pad=0)
        self.conv_4 = ConvBNReLU(in_channels[3], out_channels, 1, pad=0)

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, *modalities):
        if len(modalities) == 1:
            # 单模态情况：仅使用第一个模态
            x = modalities[0]

            x_1 = self.conv_1(x[0])
            x_2 = self.conv_2(x[1])
            x_3 = self.conv_3(x[2])
            x_4 = self.conv_4(x[3])

            H, W = x_1.shape[2:]
            for _x in [x_2, x_3, x_4]:
                x_1 = x_1 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )

            p2 = x_1
        elif len(modalities) > 1 and self.num_modalities > 1 and len(
                modalities) == self.num_modalities:
            features_list = []
            for modality in modalities:
                features = modality
                features_list.append(features)

            all_x2 = [features[0] for features in features_list]
            all_x3 = [features[1] for features in features_list]
            all_x4 = [features[2] for features in features_list]
            all_x5 = [features[3] for features in features_list]

            ff1 = self.fusion1(*all_x2)
            ff2 = self.fusion2(*all_x3)
            ff3 = self.fusion3(*all_x4)
            ff4 = self.fusion4(*all_x5)

            ff1 = self.conv_1(ff1)
            ff2 = self.conv_2(ff2)
            ff3 = self.conv_3(ff3)
            ff4 = self.conv_4(ff4)

            H, W = ff1.shape[2:]
            for _x in [ff2, ff3, ff4]:
                ff1 = ff1 + F.interpolate(
                    _x,
                    size=(H, W),
                    mode="bilinear",
                )
            p2 = ff1
        else:
            raise ValueError(
                f"Invalid number of modalities: {len(modalities)}, expected {self.num_modalities}"
            )

        return self.out_conv(p2)


# 使用示例
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
