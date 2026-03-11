import os
import sys
import torch
import numpy as np

from tqdm import tqdm

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dinov3.distributed as distributed

from utils.metrics import metrics_print_version as metrics
from utils.inference import slide_inference

# 选择对应检查点 - 使用变量方式导入
# 从环境变量获取项目ID，如果没有设置则使用默认值
DATASET_NAME = ""
MODEL_NAME = ""
NUM_MODALITIES = -1

# 从动态导入的模块中获取需要的变量和类
from tasks.segmentation.datasets import build_dataset
from configs import get_cfg
from configs.common_cfg import MS_ROOT_DIR

from PIL import Image


def save_prediction_as_image(pred, label, save_dir, index):
    """将预测结果保存为图像"""

    if DATASET_NAME == "Vaihingen" or DATASET_NAME == "Potsdam":
        palette = np.array(
            [
                [255, 255, 255],  # 0: Impervious surfaces (white)
                [0, 0, 255],  # 1: Buildings (blue)
                [0, 255, 255],  # 2: Low vegetation (cyan)
                [0, 255, 0],  # 3: Trees (green)
                [255, 255, 0],  # 4: Cars (yellow)
                [255, 0, 0],  # 5: Clutter (red)
                [0, 0, 0]  # 6: Undefined (black)
            ],
            dtype=np.uint8)
    elif DATASET_NAME == "YYYJ":
        palette = np.array(
            [
                # 地基建设 → 棕色系
                [102, 50, 18],  # 0: 深褐色
                [175, 117, 71],  # 1: 赭石色
                [214, 171, 131],  # 2: 浅土黄色
                [231, 212, 190],  # 3: 极浅的米黄色

                # 施工道路 → 灰色系
                [64, 64, 64],  # 4: 深灰色
                [153, 153, 153],  # 5: 中灰色

                # 风电施工 → 蓝色系
                [18, 74, 143],  # 6: 深蓝色
                [125, 177, 230],  # 7: 天蓝色

                # 独立类别 - 高区分度配色
                [255, 191, 0],  # 8: 琥珀色/金黄色
                [54, 140, 48],  # 9: 深绿色
                [212, 50, 125],  # 10: 洋红色/品红色
                [128, 78, 191],  # 11: 中等深度的紫色
                [188, 155, 218],  # 12: 浅薰衣草紫
                [220, 60, 60],  # 13: 鲜红色
                [255, 255, 255],  # 14: 白色
                [0, 0, 0]  # 15: 黑色
            ],
            dtype=np.uint8)
    elif DATASET_NAME == "WHU":
        palette = np.array(
            [
                [204, 102, 1],  # 0: farmland
                [255, 0, 0],  # 1: city
                [255, 255, 0],  # 2: village
                [0, 0, 255],  # 3: water
                [85, 166, 1],  # 4: forest
                [93, 255, 255],  # 5: road
                [152, 102, 153],  # 6: others
                [0, 0, 0]
            ],
            dtype=np.uint8)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 应用颜色映射到预测结果
    pred_colored = palette[pred]
    label_colored = palette[label]

    # 转换为PIL图像并保存
    pred_img = Image.fromarray(pred_colored.squeeze().astype(np.uint8), 'RGB')
    label_img = Image.fromarray(label_colored.squeeze().astype(np.uint8),
                                'RGB')

    pred_img.save(os.path.join(save_dir, f"{index}_prediction.png"))
    label_img.save(os.path.join(save_dir, f"{index}_ground_truth.png"))


def get_local_rank():
    """获取本地rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def main(**kwargs):
    try:
        # 初始化分布式训练环境
        distributed.enable(overwrite=True)
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Falling back to single GPU training")
        # 手动设置环境以进行单GPU训练
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    # 获取模型配置
    # cfg = cfg_module.get_cfg(MODEL_NAME, DATASET_NAME)
    cfg = get_cfg(MODEL_NAME, DATASET_NAME, **kwargs)
    window_size = cfg.get('window_size')
    model = cfg.get('model')

    test_dataset = build_dataset(
        DATASET_NAME,
        "test",
        window_size=window_size,
        model_name=MODEL_NAME,
        modality="multi" if NUM_MODALITIES > 1 else None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # 将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load(classification_model_path)["model"],
                          strict=False)

    # 如果分布式训练可用，则包装为分布式模型
    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_local_rank()]
            if torch.cuda.is_available() else None,
            output_device=get_local_rank()
            if torch.cuda.is_available() else None,
            find_unused_parameters=True,  # 这将允许模型在某些参数未参与损失计算时仍能正常工作
        )

    test(model, test_loader, cfg)


def test(model, test_loader, cfg):
    # 清理缓存
    torch.cuda.empty_cache()
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    labels = []

    window_size = cfg.get("window_size")
    classes = cfg.get("labels")

    # 创建保存结果的目录
    modality = "uni" if NUM_MODALITIES == 1 else "multi"
    save_dir = f"./vis_results/{MODEL_NAME}_{DATASET_NAME}_{modality}"
    os.makedirs(save_dir, exist_ok=True)
    sample_index = 0

    iterations = tqdm(test_loader, disable=not distributed.is_main_process())
    for batch in iterations:
        if NUM_MODALITIES > 1:
            input, dsm, label = batch
            input, dsm = input.to(device), dsm.to(device)

            with torch.no_grad():
                s_w = int(window_size[0] * 2 / 3)
                pred = slide_inference(input,
                                       model,
                                       dsm=dsm,
                                       n_output_channels=len(classes),
                                       crop_size=window_size,
                                       stride=(s_w, s_w),
                                       batch_size=cfg.get("batch_size", 4))
        else:
            input, label = batch
            input = input.to(device)

            with torch.no_grad():
                s_w = int(window_size[0] * 2 / 3)
                pred = slide_inference(input,
                                       model,
                                       n_output_channels=len(classes),
                                       crop_size=window_size,
                                       stride=(s_w, s_w),
                                       batch_size=cfg.get("batch_size", 4))

        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label)

        # 保存预测结果为图像
        save_prediction_as_image(pred, label.numpy(), save_dir, sample_index)
        sample_index += 1

    MIoU, F1, Kappa, Acc = metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([p.ravel() for p in labels]).ravel(), classes)

    # 构建详细指标字典
    detailed_metrics = {"MIoU": MIoU, "F1": F1, "Kappa": Kappa, "Acc": Acc}

    return detailed_metrics


if __name__ == "__main__":
    test_models_list = [
        # {
        #     "model_name":
        #     "DINOv3_Baseline",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_Baseline/Vaihingen_20260201_062327/DINOv3_Baseline_Vaihingen_e50_mIoU69.22.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_Adapter",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_Adapter/Vaihingen_20260204_151138/DINOv3_Adapter_Vaihingen_e50_mIoU71.68.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_FRM",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_FRM/Vaihingen_20260201_104742/DINOv3_FRM_Vaihingen_e45_mIoU80.65.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_MMFF",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_MMFF/Vaihingen_20260201_183541/DINOv3_MMFF_Vaihingen_e50_mIoU80.72.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_PRN",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_PRN/Vaihingen_20260201_125125/DINOv3_PRN_Vaihingen_e50_mIoU79.25.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_Adapter_FRM",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_Adapter_FRM/Vaihingen_20260204_165757/DINOv3_Adapter_FRM_Vaihingen_e20_mIoU82.82.pth"
        # },
        # {
        #     "model_name":
        #     "DINOv3_Adapter_FRM_MMFF",
        #     "dataset_name":
        #     "Vaihingen",
        #     "modality":
        #     2,
        #     "classification_model_path":
        #     f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3_Adapter_FRM_MMFF/Vaihingen_20260204_232845/DINOv3_Adapter_FRM_MMFF_Vaihingen_e40_mIoU83.04.pth"
        # },
        {
            "model_name":
            "DINOv3",
            "dataset_name":
            "Vaihingen",
            "modality":
            2,
            "classification_model_path":
            f"/home/{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation/logs/DINOv3/Vaihingen_20251123_043510/DINOv3_Vaihingen_e40_mIoU83.6.pth"
        },
    ]

    for model_cfg in test_models_list:
        MODEL_NAME = model_name = model_cfg.get("model_name")
        DATASET_NAME = dataset_name = model_cfg.get("dataset_name")
        NUM_MODALITIES = modality = model_cfg.get("modality")
        classification_model_path = model_cfg.get("classification_model_path")

        print("=" * 50)
        print(f"Testing model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Modality: {modality}")
        print("=" * 50)

        main(num_modalities=NUM_MODALITIES)
