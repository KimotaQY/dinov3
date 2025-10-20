import logging
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from datetime import datetime
from datasets import build_dataset

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.logging import setup_logging
# 导入分布式训练相关模块
import dinov3.distributed as distributed

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)
from models.dino_segment import DINOSegment
from utils.metrics import CrossEntropy2d, DiceLoss, metrics, CombinedLoss
from utils.inference import slide_inference
from utils.utils import set_seed
from utils.move_files import move_files

BATCH_SIZE = 6
LABELS = ["roads", "buildings", "low veg.", "trees", "cars",
          "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)
EPOCHS = 50
WINDOW_SIZE = (512, 512)
DATASET_NAME = "Vaihingen"


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

    set_seed(42)
    pretrained_model_name = "/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
    train_dataset = build_dataset(DATASET_NAME,
                                  "train",
                                  window_size=WINDOW_SIZE)

    # 根据分布式训练设置调整采样器
    if distributed.is_enabled():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = BATCH_SIZE  # 保持batch size不变
        shuffle = False  # 使用sampler时需要设置为False
    else:
        train_sampler = None
        batch_size = BATCH_SIZE
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               sampler=train_sampler,
                                               num_workers=8,
                                               pin_memory=False)

    test_dataset = build_dataset(DATASET_NAME, "test", window_size=WINDOW_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    backbone_weights = "/home/yyyjvm/Checkpoints/facebook/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    model = DINOSegment(pretrained_model_name,
                        backbone_weights=backbone_weights,
                        n_classes=N_CLASSES,
                        window_size=WINDOW_SIZE)

    # 将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    # 创建日志目录
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_dict = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation"
    dst_dict = f"{src_dict}/logs/{date_time}"

    # 只在主进程上创建目录和保存文件
    if distributed.is_main_process():
        print(f"正在将文件移动到 {dst_dict}...")
        move_files(src_dict, os.path.join(dst_dict, 'proj_files'),
                   ['logs', '__pycache__', '.pyc'])
        print("=====文件移动完成=====")
        # 初始化日志系统
        setup_logging(output=dst_dict, level=logging.INFO, name='dinov3seg')

    # 根据GPU数量调整学习率
    base_lr = 1e-4
    if distributed.is_enabled():
        base_lr = base_lr * distributed.get_world_size()

    # 打印所有可学习参数
    # if distributed.is_main_process():
    #     print("Trainable parameters:")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"  {name}: {param.shape}")

    # 修改优化器设置，确保所有需要训练的参数都被包含
    # 原来的filter可能会遗漏一些参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # optimizer = optim.SGD(trainable_params,
    #                       lr=base_lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45],
    #                                            gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=EPOCHS,
                                                     eta_min=1e-7)

    train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          save_dir=dst_dict if distributed.is_main_process() else None)
    # test(model, test_loader)


def train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          save_dir,
          weights=WEIGHTS):
    logger = logging.getLogger("dinov3seg")
    epochs = EPOCHS
    best_IoU = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLoss = CombinedLoss()

    for e in range(1, epochs + 1):
        model.train()

        # 在分布式训练中设置采样器
        if distributed.is_enabled():
            train_loader.sampler.set_epoch(e)

        total_loss = 0.0
        num_batches = 0

        iterations = tqdm(train_loader,
                          disable=not distributed.is_main_process())
        for input, label in iterations:
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(input)

            # loss_ce = CrossEntropy2d(logits, label, weight=weights.to(device))
            # loss_dice = DiceLoss(logits, label)
            loss_ce, loss_dice = CLoss(logits,
                                       label,
                                       weight=weights.to(device))

            loss = loss_ce + loss_dice

            # 添加调试信息来帮助定位问题
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf detected in loss:")
                print(f"logits shape: {logits.shape}")
                print(f"label shape: {label.shape}")
                print(f"label min: {label.min()}, label max: {label.max()}")
                print(f"unique labels: {torch.unique(label)}")
                print(f"weights: {weights}")
                sys.exit(1)

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.data
            num_batches += 1

            if distributed.is_main_process():
                iterations.set_description(
                    "Epoch: {}/{} Loss: {:.4f} + {:.4f}".format(
                        e, epochs, loss_ce.data, loss_dice.data))
            # print('loss:', loss.data)

        # 计算并打印epoch的平均loss
        avg_loss = total_loss / num_batches

        # 在分布式训练中同步损失
        if distributed.is_enabled():
            avg_loss_tensor = torch.tensor(
                avg_loss, device=device) if not isinstance(
                    avg_loss,
                    torch.Tensor) else avg_loss.detach().clone().to(device)
            torch.distributed.all_reduce(avg_loss_tensor)
            avg_loss_tensor /= distributed.get_world_size()
            avg_loss = avg_loss_tensor.item()

        if distributed.is_main_process():
            logger.info(f"Epoch {e}/{epochs} - Average Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        # 每隔5个epoch保存一次模型
        if e % 5 == 0 and distributed.is_main_process():
            mIoU = test(model, test_loader)
            if mIoU > best_IoU:
                best_IoU = mIoU
                # 保存模型时考虑分布式包装
                model_state = model.module.state_dict() if hasattr(
                    model, 'module') else model.state_dict()
                torch.save({
                    "model": model_state
                }, f"{save_dir}/dinoseg_{DATASET_NAME}_e{e}_mIoU{round(mIoU*100, 2)}.pth"
                           )

            # 清理多余的 .pth 文件
            if save_dir is not None:
                model_files = [
                    f for f in os.listdir(save_dir) if f.endswith(".pth")
                ]
                if len(model_files) > 5:  # 设置最大保留的模型数量
                    # 按文件创建时间排序，保留最新的 5 个模型
                    model_files.sort(key=lambda x: os.path.getmtime(
                        os.path.join(save_dir, x)))
                    for file_name in model_files[:-5]:
                        os.remove(os.path.join(save_dir, file_name))
                        print(f"Deleted old model: {file_name}")

        # 保存检查点
        if save_dir is not None and distributed.is_main_process():
            model_path = save_dir + "/" + DATASET_NAME + "_checkpoint.pth"
            # 保存模型时考虑分布式包装
            model_state = model.module.state_dict() if hasattr(
                model, 'module') else model.state_dict()
            torch.save(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": e,
                },
                model_path,
            )


def test(model, test_loader):
    # 清理缓存
    torch.cuda.empty_cache()
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    labels = []

    iterations = tqdm(test_loader, disable=not distributed.is_main_process())
    for input, label in iterations:
        input = input.to(device)
        with torch.no_grad():
            pred = slide_inference(input, model, n_output_channels=N_CLASSES)

        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label)

    acc = metrics(np.concatenate([p.ravel() for p in preds]),
                  np.concatenate([p.ravel() for p in labels]).ravel(), LABELS)

    return acc


if __name__ == "__main__":
    main()
