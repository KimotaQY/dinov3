import torch.optim as optim

from losses import *
from .common_cfg import *
from tasks.segmentation.models.My_DINO.dino_segment import DINOSegment, DINOSegment_Linear, DINOSegment_Adapter, DINOSegment_PRNDecoder, DINOSegment_Decoder_PRN, DINOSegment_Decoder_FRM

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(model_name=None, dataset_name=None):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = 1e-4
    batch_size = 8
    epochs = 50
    window_size = (512, 512)
    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    # pretrained_model_name = "/home/yyyj/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
    backbone_weights = "/home/yyyj/Checkpoints/facebook/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    if model_name == 'DINOv3':
        model = DINOSegment(backbone_weights=backbone_weights,
                            n_classes=len(labels),
                            window_size=window_size,
                            use_lora=False)
    elif model_name == 'DINOv3_baseline':
        model = DINOSegment_Linear(backbone_weights=backbone_weights,
                                   n_classes=len(labels),
                                   use_lora=False)
    elif model_name == 'DINOv3_Adapter':
        model = DINOSegment_Adapter(backbone_weights=backbone_weights,
                                    n_classes=len(labels),
                                    use_lora=False)
    elif model_name == 'DINOv3_PRN':
        model = DINOSegment_PRNDecoder(backbone_weights=backbone_weights,
                                       n_classes=len(labels),
                                       use_lora=False)
    elif model_name == 'DINOv3_PRN_only':
        model = DINOSegment_Decoder_PRN(backbone_weights=backbone_weights,
                                        n_classes=len(labels),
                                        use_lora=False)
    elif model_name == 'DINOv3_FRM_only':
        model = DINOSegment_Decoder_FRM(backbone_weights=backbone_weights,
                                        n_classes=len(labels),
                                        use_lora=False)

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
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=epochs,
                                                     eta_min=1e-7)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)
