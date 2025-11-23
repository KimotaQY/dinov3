import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..backbone.dinov3_adapter import DINOv3_Adapter
# from ..heads.mask2former_head import Mask2FormerHead

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.hub.backbones import dinov3_vitl16
from dinov3.eval.segmentation.models.heads.mask2former_head import Mask2FormerHead
from .dinov3_adapter import DINOv3_Adapter

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class Mask2Former(nn.Module):

    def __init__(
        self,
        backbone_weights=None,
        n_classes: int = 1000,
        hidden_dim=2048,
        use_lora: bool = False,
        r: int = 3,
    ):
        super().__init__()

        if backbone_weights is not None:
            self.backbone = dinov3_vitl16(weights=backbone_weights,
                                          pretrained=True)
        else:
            self.backbone = dinov3_vitl16(pretrained=False)

        embed_dim = self.backbone.embed_dim
        patch_size = self.backbone.patch_size

        # self.backbone = DINOv3_Adapter(
        #     self.backbone,
        #     interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS["dinov3_vitl16"],
        # )

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

        self.backbone = DINOv3_Adapter(
            self.backbone,
            interaction_indexes=BACKBONE_INTERMEDIATE_LAYERS["dinov3_vitl16"],
        )
        self.backbone.eval()
        self.decoder = Mask2FormerHead(
            input_shape={
                "1": [embed_dim, patch_size * 4, patch_size * 4, 4],
                "2": [embed_dim, patch_size * 2, patch_size * 2, 4],
                "3": [embed_dim, patch_size, patch_size, 4],
                "4": [embed_dim,
                      int(patch_size / 2),
                      int(patch_size / 2), 4],
            },
            hidden_dim=hidden_dim,
            num_classes=n_classes,
            ignore_value=255,
        )

    def forward(self, inputs):
        _, _, H, W = inputs.shape
        patch_h, patch_w = inputs.shape[-2] // 16, inputs.shape[-1] // 16

        outputs = self.backbone.forward(inputs)

        logits = self.decoder(outputs)

        semantic_logits = self.semantic_segmentation(inputs)

        semantic_logits = F.interpolate(semantic_logits,
                                        size=(H, W),
                                        mode='bilinear',
                                        align_corners=False)

        return logits

    def semantic_segmentation(self, inputs):
        """
        从Mask2Former的输出中提取语义分割结果
        
        Args:
            inputs: 输入图像张量
            
        Returns:
            semantic_logits: 语义分割logits，形状为(B, n_classes, H, W)
        """
        outputs = self.forward(inputs)

        # 提取预测结果
        pred_logits = outputs["pred_logits"]  # (B, 100, n_classes)
        pred_masks = outputs["pred_masks"]  # (B, 100, H_mask, W_mask)

        # 使用einsum将分类概率与mask结合得到语义分割结果
        # 注意: 这里假设pred_masks已经被调整到适当的分辨率
        semantic_logits = torch.einsum("bqc,bqhw->bchw", pred_logits,
                                       pred_masks)

        # 如果需要，可以对输出进行插值以匹配输入尺寸
        # if semantic_logits.shape[2:] != inputs.shape[2:]:
        #     semantic_logits = F.interpolate(
        #         semantic_logits,
        #         size=inputs.shape[2:],
        #         mode='bilinear',
        #         align_corners=False
        #     )

        return semantic_logits
