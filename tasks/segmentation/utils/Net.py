import torch
import torch.nn as nn
import torch.nn.functional as F

from .CBAM import ChannelAttention, SpatialAttention


class FeatureReinforcementModule(nn.Module):

    def __init__(self, in_d=None, out_d=64, drop_rate=0):
        super(FeatureReinforcementModule, self).__init__()
        if in_d is None:
            raise ValueError("in_d must be provided")
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d

        # Define all conv_scale modules dynamically using a loop
        self.conv_scales = nn.ModuleDict()
        for scale in range(2, 6):  # For scales 2 to 5
            for i in range(2, 6):  # For each conv_scale1_c2 ... conv_scale5_c5
                key = f"conv_scale{i}_c{scale}"
                self.conv_scales[key] = self._create_conv_block(
                    self.in_d[scale - 1],
                    self.mid_d,
                    scale=i,
                    orig_scale=scale)

        # Fusion layers
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[1],
                                                       self.out_d, drop_rate)
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[2],
                                                       self.out_d, drop_rate)
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[3],
                                                       self.out_d, drop_rate)
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[4],
                                                       self.out_d, drop_rate)

    def _create_conv_block(self, in_channels, mid_channels, scale, orig_scale):
        layers = []
        if scale > orig_scale:  # Pooling for scales > 1
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2**(scale - orig_scale),
                    stride=2**(scale - orig_scale),
                ))
            # 使用小波下采样保留细节
            # for i in range(scale - orig_scale):
            #     layers.append(HWD(in_channels, in_channels))

        if scale == orig_scale:
            layers.extend([
                nn.Conv2d(in_channels,
                          mid_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
        elif scale != orig_scale:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channels,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])

        return nn.Sequential(*layers)

    def forward(self, c2, c3, c4, c5):
        # Handle each scale's forward pass dynamically
        def process_scale(c, scale_idx):
            scale_outputs = []
            for i in range(2, 6):  # For scales 2 to 5
                key = f"conv_scale{i}_c{scale_idx + 2}"
                output = self.conv_scales[key](c)
                if i < scale_idx + 2:  # Interpolate as needed
                    output = F.interpolate(
                        output,
                        scale_factor=(
                            2**(scale_idx + 2 - i),
                            2**(scale_idx + 2 - i),
                        ),
                        mode="bilinear",
                    )
                scale_outputs.append(output)
            return scale_outputs

        # Get outputs for all input features
        c2_scales = process_scale(c2, 0)
        c3_scales = process_scale(c3, 1)
        c4_scales = process_scale(c4, 2)
        c5_scales = process_scale(c5, 3)

        # Aggregation and fusion
        s2 = self.conv_aggregation_s2(
            torch.cat([c2_scales[0], c3_scales[0], c4_scales[0], c5_scales[0]],
                      dim=1),
            c2,
        )
        s3 = self.conv_aggregation_s3(
            torch.cat([c2_scales[1], c3_scales[1], c4_scales[1], c5_scales[1]],
                      dim=1),
            c3,
        )
        s4 = self.conv_aggregation_s4(
            torch.cat([c2_scales[2], c3_scales[2], c4_scales[2], c5_scales[2]],
                      dim=1),
            c4,
        )
        s5 = self.conv_aggregation_s5(
            torch.cat([c2_scales[3], c3_scales[3], c4_scales[3], c5_scales[3]],
                      dim=1),
            c5,
        )

        return s2, s3, s4, s5


class FeatureFusionModule(nn.Module):

    def __init__(self, fuse_d, in_d, out_d, drop_rate):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.in_d = in_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.fuse_d,
                self.fuse_d,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.fuse_d,
            ),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            # nn.Dropout(drop_rate),
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # # 自适应权重模块
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.softmax = nn.Softmax(dim=2)
        # self.Sigmoid = nn.Sigmoid()

        # self.Conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         self.fuse_d // 4,
        #         self.fuse_d // 4,
        #         kernel_size=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False,
        #     ))
        # self.Conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         self.fuse_d // 4,
        #         self.fuse_d // 4,
        #         kernel_size=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False,
        #     ))
        # self.Conv4 = nn.Sequential(
        #     nn.Conv2d(
        #         self.fuse_d // 4,
        #         self.fuse_d // 4,
        #         kernel_size=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False,
        #     ))
        # self.Conv5 = nn.Sequential(
        #     nn.Conv2d(
        #         self.fuse_d // 4,
        #         self.fuse_d // 4,
        #         kernel_size=1,
        #         padding=0,
        #         dilation=1,
        #         bias=False,
        #     ))

    def forward(self, c_fuse, c):
        # # =====自适应权重计算=====
        # c2, c3, c4, c5 = torch.chunk(c_fuse, chunks=4, dim=1)

        # # 计算多尺度权重
        # c2_weight = self.Conv2(self.gap(c2))
        # c3_weight = self.Conv3(self.gap(c3))
        # c4_weight = self.Conv4(self.gap(c4))
        # c5_weight = self.Conv5(self.gap(c5))

        # weight = torch.cat([c2_weight, c3_weight, c4_weight, c5_weight], 2)
        # weight = self.softmax(self.Sigmoid(weight))

        # # 调整权重维度
        # c2_weight = torch.unsqueeze(weight[:, :, 0], 2)
        # c3_weight = torch.unsqueeze(weight[:, :, 1], 2)
        # c4_weight = torch.unsqueeze(weight[:, :, 2], 2)
        # c5_weight = torch.unsqueeze(weight[:, :, 3], 2)

        # c_fuse = torch.cat(
        #     [c2 * c2_weight, c3 * c3_weight, c4 * c4_weight, c5 * c5_weight],
        #     dim=1)
        # # ======自适应权重 End======

        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out


class DecoderBlock(nn.Module):

    def __init__(self, mid_d):
        super(DecoderBlock, self).__init__()
        self.mid_d = mid_d
        self.conv_high = nn.Conv2d(self.mid_d,
                                   self.mid_d,
                                   kernel_size=1,
                                   stride=1)
        self.conv_global = nn.Conv2d(self.mid_d,
                                     self.mid_d,
                                     kernel_size=1,
                                     stride=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.mid_d,
                self.mid_d,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.mid_d,
            ),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, x_low, x_high, global_context):
        batch, channels, height, width = x_low.shape
        x_high = F.interpolate(self.conv_high(x_high),
                               size=(height, width),
                               mode="bilinear")
        global_context = F.interpolate(self.conv_global(global_context),
                                       size=(height, width),
                                       mode="bilinear")
        x_low = self.fusion(x_low + x_high + global_context)

        # 消融实验，去掉gc
        # x_low = self.fusion(x_low + x_high)

        mask = self.cls(x_low)
        # 消融实验，baseline
        # mask = self.cls(self.fusion(x_low) + x_low)
        return x_low, mask


class ChannelReferenceAttention(nn.Module):

    def __init__(self, in_d):
        super(ChannelReferenceAttention, self).__init__()
        self.in_d = in_d
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1),
        )
        self.high_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1),
        )
        self.low_conv = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1),
        )
        self.spatial_attention = SpatialAttention()
        self.out_conv = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, low_context, high_context, global_context):
        b, c, h, w = low_context.shape
        # 池化操作
        low_context_pool = self.low_conv(low_context)
        high_context_pool = self.high_conv(high_context)
        global_context_pool = self.global_conv(global_context)
        low_context_pool = low_context_pool.squeeze(dim=-1)
        high_context_pool = high_context_pool.squeeze(dim=-1).permute(0, 2, 1)
        global_context_pool = global_context_pool.squeeze(dim=-1).permute(
            0, 2, 1)

        att_l_h = torch.bmm(low_context_pool, high_context_pool)
        att_l_g = torch.bmm(low_context_pool, global_context_pool)
        att = torch.sigmoid(att_l_h + att_l_g)
        out = torch.bmm(att, low_context.view(b, c, -1))
        out = out.view(b, c, h, w)

        spatial_att = self.spatial_attention(out)
        out = out * spatial_att

        out = self.out_conv(out) + low_context
        return out


class Decoder(nn.Module):

    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        self.channel_attention = ChannelReferenceAttention(self.mid_d)
        self.db_p4 = DecoderBlock(self.mid_d)
        self.db_p3 = DecoderBlock(self.mid_d)
        self.db_p2 = DecoderBlock(self.mid_d)

    def forward(self, d2, d3, d4, d5, gc_d4):
        p4 = self.channel_attention(d4, d5, gc_d4)
        p4, mask_p4 = self.db_p4(p4, d5, gc_d4)
        p3 = self.channel_attention(d3, p4, gc_d4)
        p3, mask_p3 = self.db_p3(p3, p4, gc_d4)
        p2 = self.channel_attention(d2, p3, gc_d4)
        p2, mask_p2 = self.db_p2(p2, p3, gc_d4)

        # # p4 = self.channel_attention(d4, d5, gc_d4)
        # p4, mask_p4 = self.db_p4(d4, d5, gc_d4)
        # # p3 = self.channel_attention(d3, p4, gc_d4)
        # p3, mask_p3 = self.db_p3(d3, p4, gc_d4)
        # # p2 = self.channel_attention(d2, p3, gc_d4)
        # p2, mask_p2 = self.db_p2(d2, p3, gc_d4)
        return mask_p2, mask_p3, mask_p4
