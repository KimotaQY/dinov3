import importlib
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
# DATASET_NAME = "Potsdam"
DATASET_NAME = "Vaihingen"
MODEL_NAME = "DINOv3"
# MODALITY = "uni"
MODALITY = "multi"

PROJECT_ID = os.environ.get('DINO_PROJECT_ID',
                            MODEL_NAME + ".Potsdam_20251126_090536")
# 动态导入模块
train_distr_module = importlib.import_module(
    f'logs.{PROJECT_ID}.proj_files.train_distr')
datasets_module = importlib.import_module(
    f'logs.{PROJECT_ID}.proj_files.datasets')
# cfg_module = importlib.import_module(f'logs.{PROJECT_ID}.proj_files.configs')

# 从动态导入的模块中获取需要的变量和类
from tasks.segmentation.datasets import build_dataset
from configs import get_cfg

classification_model_path = "/home/yyyj/SS-projects/dinov3/tasks/segmentation/logs/DINOv3/Potsdam_20251126_090536/DINOv3_Potsdam_e40_mIoU86.02.pth"


def get_local_rank():
    """获取本地rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def main():
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
    cfg = get_cfg(MODEL_NAME, DATASET_NAME)
    window_size = cfg.get('window_size')
    model = cfg.get('model')

    test_dataset = build_dataset(DATASET_NAME,
                                 "test",
                                 window_size=window_size,
                                 model_name=MODEL_NAME,
                                 modality=MODALITY)
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

    iterations = tqdm(test_loader, disable=not distributed.is_main_process())
    if MODALITY == "multi":
        for input, dsm, label in iterations:
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

            pred = np.argmax(pred, axis=1)
            preds.append(pred)
            labels.append(label)
    else:
        for input, label in iterations:
            input = input.to(device)
            with torch.no_grad():
                s_w = int(window_size[0] * 2 / 3)
                pred = slide_inference(input,
                                       model,
                                       n_output_channels=len(classes),
                                       crop_size=window_size,
                                       stride=(s_w, s_w),
                                       batch_size=(cfg.get("batch_size", 4)) *
                                       4)

            pred = np.argmax(pred, axis=1)
            preds.append(pred)
            labels.append(label)

    MIoU, F1, Kappa, Acc = metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([p.ravel() for p in labels]).ravel(), classes)

    # 构建详细指标字典
    detailed_metrics = {"MIoU": MIoU, "F1": F1, "Kappa": Kappa, "Acc": Acc}

    return detailed_metrics


# 设置调试环境变量
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from tasks.segmentation.models.UMFormer.UMFormer import UMFormer
if __name__ == "__main__":
    main()
    # local_model_dir = "/home/yyyj/Checkpoints/timm/resnet18.fb_swsl_ig1b_ft_in1k"
    # net = UMFormer(6, local_model_dir=local_model_dir)
    # net = net.cuda(0)
    # dummy_input = torch.randn(1, 3, 1024, 1024).cuda(0).contiguous()
    # # flops, params = profile(net, (dummy_input, ))
    # # print('flops: ', flops, 'params: ', params)
    # # print('flops: %.2f M, params: %.2f M' %
    # #       (flops / 1000000.0, params / 1000000.0))
    # # print('***************************')
    # # # x = torch.rand(2, 3, 512, 512).to(device)
    # # total = sum([param.nelement() for param in net.parameters()])  # 计算总参数量
    # # print("model size:", total / 1000 / 1000, "M")
    # # print('***************************')
    # y = net(dummy_input)
    # print(y.size())
