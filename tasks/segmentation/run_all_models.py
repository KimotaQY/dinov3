#!/usr/bin/env python
"""批量运行多个模型的训练"""

import subprocess
import sys
import os


def run_model_training(model_name):
    """运行单个模型的训练"""
    cmd = [
        "torchrun", "--nproc_per_node=2",
        "./tasks/segmentation/train_multi.py", "--model-name", model_name
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
    models = [
        # "DINOv3_Baseline",
        "DINOv3_FRM",
        "DINOv3_PRN",
        "DINOv3_MMFF",
        "DINOv3_FRM_MMFF",
        "DINOv3_PRN_MMFF",
        "DINOv3_Adapter_FRM",
        "DINOv3_Adapter_PRN",
        "DINOv3_Adapter_MMFF",
        "DINOv3_Adapter_FRM_MMFF",
        "DINOv3_Adapter_PRN_MMFF",
        "DINOv3",
    ]

    success_count = 0
    total_count = len(models)

    for model in models:
        if run_model_training(model):
            success_count += 1
        print("-" * 50)

    print(f"\n训练完成: {success_count}/{total_count} 个模型成功训练")


if __name__ == "__main__":
    main()
