import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from datasets import build_dataset

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)
from model.dino_segment import DINOSegment
from utils.metrics import CrossEntropy2d, metrics
from utils.inference import slide_inference
from utils.utils import set_seed

BATCH_SIZE = 8
LABELS = ["roads", "buildings", "low veg.", "trees", "cars",
          "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)
EPOCHS = 50
WINDOW_SIZE = (512, 512)


def main():
    set_seed(42)
    train_dataset = build_dataset("Vaihingen",
                                  "train",
                                  window_size=WINDOW_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_dataset = build_dataset("Vaihingen", "test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    pretrained_model_name = "/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
    model = DINOSegment(pretrained_model_name,
                        n_classes=N_CLASSES,
                        window_size=WINDOW_SIZE)
    # model.to(model.encoder.device)

    base_lr = 0.1
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                 model.parameters()),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45],
                                               gamma=0.1)

    train(model, train_loader, test_loader, optimizer, scheduler)
    # test(model, test_loader)


def train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          weights=WEIGHTS):
    epochs = EPOCHS
    for e in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        num_batches = 0

        iterations = tqdm(train_loader)
        for input, label in iterations:
            optimizer.zero_grad()
            logits = model(input)

            loss = CrossEntropy2d(logits,
                                  label.to(logits.device),
                                  weight=weights.to(logits.device))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
            num_batches += 1

            iterations.set_description("Epoch: {}/{} Loss: {:.4f}".format(
                e, epochs, loss.data))
            # print('loss:', loss.data)

        # 计算并打印epoch的平均loss
        avg_loss = total_loss / num_batches
        print(f"Epoch {e}/{epochs} - Average Loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        test(model, test_loader)


def test(model, test_loader):
    model.eval()

    preds = []
    labels = []

    iterations = tqdm(test_loader)
    for input, label in iterations:
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
