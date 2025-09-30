import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)  # 设置 Python 内部的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的随机种子（单 GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU 设置所有 GPU 的随机种子

    # 确保 CuDNN 的确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
