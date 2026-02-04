import torch.optim as optim
import argparse

from losses import *
from .common_cfg import *
from models.ESANet import ArgumentParserRGBDSegmentation, build_model

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()

    base_lr = args.lr
    batch_size = 8
    epochs = 500
    window_size = (512, 512)
    labels = get_labels(dataset_name)
    ignore_index = len(labels) - 1
    loss_fn = SoftCrossEntropyLoss(smooth_factor=0.05,
                                   ignore_index=ignore_index)

    model = build_model(
        args,
        n_classes=len(labels),
    )[0]

    # 根据GPU数量调整学习率
    if distributed.is_enabled():
        base_lr = base_lr * distributed.get_world_size()

    # 分别为backbone和其他部分设置不同的学习率
    backbone_params = []
    other_params = []

    # 如果backbone中有需要训练的参数（如LoRA参数）
    if hasattr(model, 'backbone'):
        backbone_params = [
            p for p in model.backbone.parameters() if p.requires_grad
        ]

    # 其他所有需要训练的参数
    other_params = []
    for name, param in model.named_parameters():
        # 排除backbone中的参数，剩下的都是其他参数
        if not name.startswith('backbone') and param.requires_grad:
            other_params.append(param)

    # 为不同部分设置不同的学习率
    param_groups = [
        {
            'params': backbone_params,
            'lr': base_lr
        },
        {
            'params': other_params,
            'lr': base_lr
        }  # 其他部分使用正常学习率
    ]

    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[i['lr'] for i in optimizer.param_groups],
        total_steps=epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e-4)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)
