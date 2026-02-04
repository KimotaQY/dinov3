#!/usr/bin/env python
"""批量运行多个模型的训练"""

import subprocess
import sys
import os

import torch

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from tasks.segmentation.models.My_DINO.dino_segment import build_model


def run_model_training(model_name):
    """运行单个模型的训练"""
    cmd = [
        "torchrun", "--nproc_per_node=2",
        "./tasks/segmentation/train_multi.py", "--model-name", model_name,
        "--num-modalities", "1"
    ]

    print(f"开始训练模型: {model_name}")
    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"模型 {model_name} 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"模型 {model_name} 训练失败，错误码: {e.returncode}")
        return False


def main():
    # models = [
    #     "DINOv3_Baseline",
    #     "DINOv3_Adapter",
    #     "DINOv3_FRM",
    #     "DINOv3_MMFF",
    #     "DINOv3_PRN",
    #     "DINOv3_FRM_MMFF",
    #     "DINOv3_FRM_PRN",
    #     "DINOv3_PRN_MMFF",
    #     "DINOv3_Adapter_FRM",
    #     "DINOv3_Adapter_MMFF",
    #     "DINOv3_Adapter_PRN",
    #     "DINOv3_FRM_MMFF_PRN",
    #     "DINOv3_Adapter_FRM_MMFF",
    #     "DINOv3_Adapter_PRN_MMFF",
    #     "DINOv3_Adapter_FRM_PRN",
    #     "DINOv3",
    # ]
    models = [
        # "DINOv3_Baseline",
        # "DINOv3_Adapter",
        # "DINOv3_FRM",
        # "DINOv3_PRN",
        # "DINOv3_FRM_PRN",
        # "DINOv3_Adapter_FRM",
        # "DINOv3_Adapter_PRN",
        "DINOv3",
    ]

    success_count = 0
    total_count = len(models)

    for model in models:
        if run_model_training(model):
            success_count += 1
        print("-" * 50)

    print(f"\n训练完成: {success_count}/{total_count} 个模型成功训练")


def test_model():
    models = [
        "DINOv3_Baseline",
        "DINOv3_Adapter",
        "DINOv3_FRM",
        "DINOv3_PRN",
        "DINOv3_MMFF",
        "DINOv3_FRM_MMFF",
        "DINOv3_PRN_MMFF",
        "DINOv3_FRM_PRN",
        "DINOv3_FRM_MMFF_PRN",
        "DINOv3_Adapter_FRM",
        "DINOv3_Adapter_PRN",
        "DINOv3_Adapter_MMFF",
        "DINOv3_Adapter_FRM_MMFF",
        "DINOv3_Adapter_PRN_MMFF",
        "DINOv3_Adapter_FRM_PRN",
        "DINOv3",
    ]

    num_modalities = 2
    for model_name in models:
        backbone_weights = "/home/yyyj/Checkpoints/facebook/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        model = build_model(model_name=model_name,
                            backbone_weights=backbone_weights,
                            n_classes=6,
                            use_lora=False,
                            num_modalities=num_modalities)
        model.cuda()

        if num_modalities == 1:
            input = torch.randn(1, 3, 224, 224)
            input = input.cuda()
            output = model(input)
            print(f"模型 {model_name} 输出形状: {output.shape}")
        elif num_modalities > 1:
            inputs = [
                torch.randn(1, 3, 224, 224) for _ in range(num_modalities)
            ]
            inputs = [input.cuda() for input in inputs]
            output = model(*inputs)
            print(f"模型 {model_name} 输出形状: {output.shape}")


if __name__ == "__main__":
    main()
    # test_model()
    # from PIL import Image
    # from skimage.io import imread
    # import numpy as np

    # img = imread(
    #     "/home/yyyj/SS-datasets/EarthMiss/Peru-Callao/images/RGB/Peru-Callao_clip_9_5.tif"
    #     # "/home/yyyj/SS-datasets/EarthMiss/America-Eugene/masks/America-Eugene_clip_1_6.tif"
    # )
    # print(img.shape)
    # img = img.transpose((2, 0, 1))

    # # # 获取最小值和最大值
    # # min_val = np.min(img)
    # # max_val = np.max(img)

    # # # 防止除零错误
    # # if max_val > min_val:
    # #     dsm = (img - min_val) / (max_val - min_val)
    # # else:
    # #     # 如果所有像素值都相同，设置为0（或保持原值）
    # #     dsm = np.full_like(img, 0.0)  # 或者 dsm = np.full_like(dsm, min_val)

    # img = Image.fromarray(img)

    # img_2 = Image.open(
    #     "/home/yyyj/SS-datasets/ISPRS_dataset/Potsdam/2_Ortho_RGB/top_potsdam_2_10_RGB.tif"
    # )
    # img_2 = img_2.convert('RGB')
    # print(img_2.shape)
