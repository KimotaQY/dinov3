import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from .linear_decoder import LinearHead
from .fpn_decoder import FPNDecoder
from .backbone.dinov3_adapter import DINOv3_Adapter
from .heads.mask2former_head import Mask2FormerHead

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

    def __init__(self,
                 model_name,
                 backbone_weights=None,
                 n_classes: int = 1000,
                 hidden_dim=2048,
                 window_size=(224, 224)):
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
        # patch_size = self.backbone.patch_size
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
        self.fpn = FPNDecoder(in_channels=embed_dim,
                              out_channels=embed_dim // 8,
                              n_classes=n_classes)
        # for param in self.decoder.parameters():
        #     param.requires_grad = True

        # self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)

    def forward(self, inputs):
        _, _, H, W = inputs.shape

        inputs = self.processor(inputs, return_tensors="pt")

        with torch.autocast("cuda"):
            outputs = self.backbone.forward(inputs.data['pixel_values'])

        # pred = self.decoder(outputs['3'])
        # pred = self.decoder(outputs)

        # 仅使用FPN
        multi_scale_features = []
        for i, key in enumerate(outputs):
            multi_scale_features.append(outputs[key])
        fpn_out = self.fpn(multi_scale_features)

        # _, _, multi_scale_features = self.decoder.pixel_decoder.forward_features(
        #     outputs)
        # fpn_out = self.fpn(multi_scale_features[::-1])

        # 确保输出大小与输入一致
        pred = F.interpolate(
            fpn_out,
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
