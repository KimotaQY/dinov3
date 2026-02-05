import math
import torch.optim as optim
import torch.nn as nn

from losses import *
from .common_cfg import *
from models.MultiSenseSeg import Build_MultiSenseSeg

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(dataset_name=None):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = 0.0002
    batch_size = 8
    epochs = 100
    window_size = (512, 512)
    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = LossFn(epochs=epochs, ignore_index=ignore_index)

    model = Build_MultiSenseSeg(n_classes=len(labels), in_chans=(3, 3))

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
            'lr': base_lr * 1
        },
        {
            'params': other_params,
            'lr': base_lr
        }  # 其他部分使用正常学习率
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    e = warm_epochs = 5
    lf = lambda x: (((1 + math.cos((x - e + 1) * math.pi /
                                   (epochs - e))) / 2)**1.0
                    ) * 0.95 + 0.02 if x >= e else 0.95 / (e - 1) * x + 0.02
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=lf,
                                            last_epoch=-1)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)


class LossFn(nn.Module):

    def __init__(self, epochs, ignore_index):
        super(LossFn, self).__init__()

        a_begin, a_end = 0.5, 0.4
        self.factor = lambda x: a_begin + (a_end - a_begin) / epochs * x

        self.loss_fn = SoftCrossEntropyLoss(smooth_factor=0.05,
                                            ignore_index=ignore_index)

    def forward(self, pred, label, epoch):
        a = self.factor(epoch)

        loss_main = self.loss_fn(pred[0], label)
        loss_aux = self.loss_fn(pred[1], label)
        loss = loss_main + loss_aux * a

        return loss
