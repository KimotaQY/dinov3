import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

from .linear_decoder import LinearHead


class DINOSegment(nn.Module):

    def __init__(self,
                 model_name,
                 emb_dim: int = 1024,
                 n_classes: int = 1000,
                 window_size=(224, 224)):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if window_size is not None:
            self.processor.size = {
                'height': window_size[0],
                'width': window_size[1]
            }
        self.encoder = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
        )
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = LinearHead(in_ch=emb_dim, n_classes=n_classes)
        self.decoder = self.decoder.to(self.encoder.device)

    def forward(self, image):
        inputs = self.processor(image,
                                return_tensors="pt").to(self.encoder.device)
        with torch.autocast("cuda"):
            with torch.no_grad():
                outputs = self.encoder(**inputs)

        # pooled_output = outputs.pooler_output
        # print("Pooled output shape:", pooled_output.shape)

        # cls = outputs.last_hidden_state[:, 0]  # 全局（[CLS]）
        num_regs = self.encoder.config.num_register_tokens
        patch_flat = outputs.last_hidden_state[:, 1 + num_regs:, :]
        # 重塑为[B, C, H, W]，步长=16
        B, N, C = patch_flat.shape
        H = W = int((N)**0.5)
        feat_map = patch_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        logits = self.decoder(feat_map)

        return logits
