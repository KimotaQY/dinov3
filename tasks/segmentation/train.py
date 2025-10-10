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

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)
from model.dino_segment import DINOSegment
from utils.metrics import CrossEntropy2d, metrics
from utils.inference import slide_inference
from utils.utils import set_seed
from utils.move_files import move_files

BATCH_SIZE = 8
LABELS = ["roads", "buildings", "low veg.", "trees", "cars",
          "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)
EPOCHS = 50
WINDOW_SIZE = (512, 512)
DATASET_NAME = "Vaihingen"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_seed(42)
    train_dataset = build_dataset(DATASET_NAME,
                                  "train",
                                  window_size=WINDOW_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_dataset = build_dataset(DATASET_NAME, "test", window_size=WINDOW_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    pretrained_model_name = "/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
    model = DINOSegment(pretrained_model_name,
                        n_classes=N_CLASSES,
                        window_size=WINDOW_SIZE)
    model.to(device)

    # 创建日志目录
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_dict = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation"
    dst_dict = f"{src_dict}/logs/{date_time}"
    move_files(src_dict, os.path.join(dst_dict, 'proj_files'),
               ['logs', '__pycache__', '.pyc'])
    # 初始化日志系统
    setup_logging(output=dst_dict, level=logging.INFO, name='dinov3seg')

    base_lr = 0.1
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                 model.parameters()),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45],
                                               gamma=0.1)

    train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          save_dir=dst_dict)
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
    for e in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        num_batches = 0

        iterations = tqdm(train_loader)
        for input, label in iterations:
            optimizer.zero_grad()

            logits = model(input.to(device))

            loss = CrossEntropy2d(logits,
                                  label.to(device),
                                  weight=weights.to(device))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
            num_batches += 1

            iterations.set_description("Epoch: {}/{} Loss: {:.4f}".format(
                e, epochs, loss.data))
            # print('loss:', loss.data)

        # 计算并打印epoch的平均loss
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {e}/{epochs} - Average Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        # 记录控制台输出

        # 每隔5个epoch保存一次模型
        if e % 5 == 0:
            mIoU = test(model, test_loader)
            if mIoU > best_IoU:
                best_IoU = mIoU
                torch.save({"model": model.state_dict()},
                           f"{save_dir}/dinoseg_{DATASET_NAME}_mIoU{mIoU}.pth")

            # 清理多余的 .pth 文件
            model_files = [
                f for f in os.listdir(save_dir) if f.endswith(".pth")
            ]
            if len(model_files) > 5:  # 设置最大保留的模型数量
                # 按文件创建时间排序，保留最新的 5 个模型
                model_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(save_dir, x)))
                for file_name in model_files[:-5]:
                    os.remove(os.path.join(save_dir, file_name))
                    print(f"Deleted old model: {file_name}")

        if save_dir is not None:
            model_path = save_dir + "/" + DATASET_NAME + "_checkpoint.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": e + 1,
                },
                model_path,
            )


def test(model, test_loader):
    model.eval()

    preds = []
    labels = []

    iterations = tqdm(test_loader)
    for input, label in iterations:
        with torch.no_grad():
            pred = slide_inference(input.to(device),
                                   model,
                                   n_output_channels=N_CLASSES)

        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label)

    acc = metrics(np.concatenate([p.ravel() for p in preds]),
                  np.concatenate([p.ravel() for p in labels]).ravel(), LABELS)

    return acc


if __name__ == "__main__":
    main()
