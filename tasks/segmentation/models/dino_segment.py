import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from .linear_decoder import LinearHead
from .fpn_decoder import FPNDecoder
from .prn_decoder import PRNDecoder
from .backbone.dinov3_adapter import DINOv3_Adapter
from .heads.mask2former_head import Mask2FormerHead
from .heads.pixel_decoder import MSDeformAttnPixelDecoder
from .lora import LoRA

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.hub.backbones import dinov3_vitl16

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class DINOSegment(nn.Module):

    def __init__(
            self,
            model_name,
            backbone_weights=None,
            n_classes: int = 1000,
            hidden_dim=2048,
            window_size=(224, 224),
            use_lora: bool = False,
            r: int = 3,
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if window_size is not None:
            self.processor.size = {
                'height': window_size[0],
                'width': window_size[1]
            }

        if backbone_weights is not None:
            self.backbone = dinov3_vitl16(weights=backbone_weights,
                                          pretrained=True)
        else:
            self.backbone = dinov3_vitl16(pretrained=False)

        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.backbone = DINOv3_Adapter(
            self.backbone,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS["dinov3_vitl16"],
        )

        embed_dim = self.backbone.backbone.embed_dim
        patch_size = self.backbone.patch_size
        # self.decoder = Mask2FormerHead(
        #     input_shape={
        #         "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
        #         "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
        #         "3": [embed_dim, patch_size, patch_size, 4],
        #         "4": [embed_dim,
        #               int(patch_size / 2),
        #               int(patch_size / 2), 4],
        #     },
        #     hidden_dim=hidden_dim,
        #     num_classes=n_classes,
        #     ignore_value=255,
        # )
        # self.pixel_decoder = MSDeformAttnPixelDecoder(
        #     input_shape={
        #         "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
        #         "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
        #         "3": [embed_dim, patch_size, patch_size, 4],
        #         "4": [embed_dim,
        #               int(patch_size / 2),
        #               int(patch_size / 2), 4],
        #     },
        #     transformer_dropout=0.0,
        #     transformer_nheads=16,
        #     transformer_dim_feedforward=4096,
        #     transformer_enc_layers=6,
        #     conv_dim=hidden_dim,
        #     mask_dim=hidden_dim,
        #     norm="GN",
        #     transformer_in_features=["1", "2", "3", "4"],
        #     common_stride=4,
        # )
        self.decoder = PRNDecoder(in_channels=[embed_dim] * 4,
                                  out_channels=256,
                                  n_classes=n_classes)
        # self.fpn = FPNDecoder(in_channels=embed_dim,
        #                       out_channels=256,
        #                       n_classes=n_classes)
        # for param in self.decoder.parameters():
        #     param.requires_grad = True

        # self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, inputs):
        _, _, H, W = inputs.shape

        inputs = self.processor(inputs, return_tensors="pt")

        with torch.autocast("cuda"):
            outputs = self.backbone.forward(inputs.data['pixel_values'])

        # pred = self.decoder(outputs['3'])
        # pred = self.decoder(outputs)

        # 仅使用FPN
        multi_scale_features = [
            outputs["1"],  # 高分辨率特征
            outputs["2"],
            outputs["3"],
            outputs["4"]  # 低分辨率特征
        ]
        # logits = self.fpn(multi_scale_features)
        logits = self.decoder(multi_scale_features)

        # pixel_decoder + fpn
        # _, _, multi_scale_features = self.pixel_decoder.forward_features(
        #     outputs)
        # logits = self.fpn(multi_scale_features[::-1])

        # 确保输出大小与输入一致
        pred = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
        )

        # mask_pred, mask_cls = pred["pred_masks"], pred["pred_logits"]
        # mask_pred = F.interpolate(
        #     mask_pred,
        #     size=(H, W),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        # mask_pred = mask_pred.sigmoid()
        # pred = torch.einsum("bqc,bqhw->bchw", mask_cls.to(torch.float),
        #                     mask_pred.to(torch.float))

        return pred
