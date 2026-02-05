import numpy as np
import torch.optim as optim
import torch.nn as nn

from losses import *
from .common_cfg import *
from models.FTransUNet import VisionTransformer as ViT_seg
from models.FTransUNet import CONFIGS as CONFIGS_ViT_seg

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = 0.01
    batch_size = 10
    epochs = 50
    window_size = (256, 256)
    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = SoftCrossEntropyLoss(smooth_factor=0.05,
                                   ignore_index=ignore_index)

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = len(labels)
    config_vit.n_skip = 3
    config_vit.patches.grid = (int(window_size[0] / 16),
                               int(window_size[1] / 16))
    model = ViT_seg(config_vit,
                    img_size=window_size[0],
                    num_classes=len(labels)).cuda()
    model.load_from(weights=np.load(config_vit.pretrained_path))

    # 根据GPU数量调整学习率
    if distributed.is_enabled():
        base_lr = base_lr * distributed.get_world_size()

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params': [value], 'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params': [value], 'lr': base_lr / 2}]

    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45],
                                               gamma=0.1)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)
