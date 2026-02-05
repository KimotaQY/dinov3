import torch.optim as optim
import torch.nn as nn

from losses import *
from .common_cfg import *
from models.CMGFNet import CMGFNet

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = 0.001
    batch_size = 16
    epochs = 30
    window_size = (256, 256)
    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = LossFn(ignore_index)

    model = CMGFNet(num_classes=len(labels), pretrained=True)

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

    optimizer = optim.Adamax(param_groups, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer,
                                                total_iters=epochs,
                                                power=0.9)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)


class LossFn(nn.Module):

    def __init__(self, ignore_index):
        super(LossFn, self).__init__()
        self.loss_fn = JointLoss(
            SoftCrossEntropyLoss(smooth_factor=0.05,
                                 ignore_index=ignore_index),
            DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    def forward(self, pred, label):
        output1, output2, output3 = pred
        loss1 = self.loss_fn(output1, label)
        loss2 = self.loss_fn(output2, label)
        loss3 = self.loss_fn(output3, label)

        return (loss1 + loss2 + loss3) / 3
