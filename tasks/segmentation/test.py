import io
import os
import pickle
import tarfile
import urllib

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

os.environ.setdefault("DINOV3_LOCATION", "/home/yyyjvm/SS-projects/dinov3")
# DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"

# if os.getenv("DINOV3_LOCATION") is not None:
#     DINOV3_LOCATION = os.getenv("DINOV3_LOCATION")
# else:
#     DINOV3_LOCATION = DINOV3_GITHUB_LOCATION

# print(f"DINOv3 location set to {DINOV3_LOCATION}")

# # examples of available DINOv3 models:
# MODEL_DINOV3_VITS = "dinov3_vits16"
# MODEL_DINOV3_VITSP = "dinov3_vits16plus"
# MODEL_DINOV3_VITB = "dinov3_vitb16"
# MODEL_DINOV3_VITL = "dinov3_vitl16"
# MODEL_DINOV3_VITHP = "dinov3_vith16plus"
# MODEL_DINOV3_VIT7B = "dinov3_vit7b16"

# MODEL_NAME = MODEL_DINOV3_VITL

# model = torch.hub.load(
#     repo_or_dir=DINOV3_LOCATION,
#     model=MODEL_NAME,
#     source="local" if DINOV3_LOCATION != DINOV3_GITHUB_LOCATION else "github",
#     # weights="/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
# )
# model.cuda()

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
print(f"图像加载完成，尺寸: {image.size}")

pretrained_model_name = "/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(
    pretrained_model_name, 
    device_map="auto", 
)

inputs = processor(images=image, return_tensors="pt").to(model.device)
print(f"输入预处理完成，输入形状: {inputs['pixel_values'].shape}")
with torch.inference_mode():
    outputs = model(**inputs)

pooled_output = outputs.pooler_output
print("Pooled output shape:", pooled_output.shape)

cls = outputs.last_hidden_state[:, 0]  # 全局（[CLS]）
num_regs = model.config.num_register_tokens  
patch_flat = outputs.last_hidden_state[:, 1 + num_regs:, :]  
# 重塑为[B, C, H, W]，步长=16
B, N, C = patch_flat.shape  
H = W = int((N) ** 0.5)  
feat_map = patch_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
print("feat map shape:", feat_map.shape)

# 打印模型配置信息
config = model.config
print(f"- 图像尺寸: {config.image_size}x{config.image_size}")