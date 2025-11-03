import os
import sys

# fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

# 添加matplotlib用于可视化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import numpy as np
import torch
from PIL import Image

# 将项目根目录添加到路径中
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)

from models.dino_segment import DINOSegment
from datasets import build_dataset
from utils.inference import slide_inference

# 选择对应检查点
from logs.YYYJ_20251031_062617.proj_files.train_distr import LABELS, N_CLASSES, WINDOW_SIZE, DATASET_NAME, WEIGHTS, test

# 模型配置
MODEL = None
pretrained_model_name = "/home/yyyjvm/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
backbone_weights = "/home/yyyjvm/Checkpoints/facebook/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

classification_model_path = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation/logs/YYYJ_20251031_062617/dinoseg_YYYJ_e95_mIoU28.02.pth"

# 定义配色方案
PALETTE = {
    # 【地基建设】→ 棕色系
    0: (102, 50, 18),  # 深褐色
    1: (175, 117, 71),  # 赭石色
    2: (214, 171, 131),  # 浅土黄色
    3: (231, 212, 190),  # 极浅的米黄色

    # 【施工道路】→ 灰色系
    4: (64, 64, 64),  # 深灰色
    5: (153, 153, 153),  # 中灰色

    # 【独立类别 - 高区分度配色】
    6: (54, 140, 48),  # 深绿色
    7: (212, 50, 125),  # 洋红色/品红色
    8: (128, 78, 191),  # 中等深度的紫色
    9: (188, 155, 218),  # 浅薰衣草紫
    10: (220, 60, 60),  # 鲜红色
    11: (255, 255, 255)  # 白色
}

app = FastAPI(title="DINOv3 Segmentation API",
              description="API for image segmentation using DINOv3 model",
              version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """在应用启动时加载模型"""
    load_model()
    print("Model loaded successfully")


@app.get("/")
async def root():
    """根路径，用于检查 API 是否正常运行"""
    return {"message": "DINOv3 Segmentation API is running!"}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    model_status = "loaded" if MODEL is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "classes": int(N_CLASSES) if N_CLASSES else None
    }


@app.post("/segment/add")
async def segment_image(task_id: str | int = Form(...),
                        img_path: str = Form(...)):
    """
    接收上传的图像文件并返回分割结果
    
    参数:
    - task_id: 任务ID (字符串或整数)
    - img_path: 图像文件路径
    
    返回:
    - JSON 格式的分割结果
    """
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not isinstance(task_id, (str, int)):
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id. Must be a string or integer.")

    if not isinstance(img_path, str):
        raise HTTPException(status_code=400,
                            detail="Invalid img_path. Must be a string.")

    try:
        # 检查文件是否存在
        # print(img_path)
        # if not os.path.exists(img_path):
        #     raise HTTPException(status_code=404,
        #                         detail=f"Image file not found: {img_path}")

        # # 读取图像
        # image = Image.open(img_path).convert('RGB')

        # # 转换为模型所需的格式
        # data = np.array(image, dtype='float32').transpose((2, 0, 1)) / 255.0
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data = torch.from_numpy(data).unsqueeze(0).to(device)

        # # 进行推理
        # model = MODEL
        # pred_np = predict(data, model)

        # 返回结果
        print("返回结果")
        return JSONResponse(content={
            "task_id": task_id,
            "message": "Segmentation started"
        })
        # return JSONResponse(
        #     content={
        #         "task_id": task_id,
        #         "image_path": img_path,
        #         "message": "Segmentation successful",
        #         "prediction_shape": [int(x) for x in pred_np.shape],
        #         "classes": int(N_CLASSES),
        #         "unique_classes_predicted":
        #         [int(x) for x in np.unique(pred_np)]
        #     })

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing image: {str(e)}")


# 图像分类
def image_classification(img_path):
    # 检查模型是否加载
    load_model()

    # 检查路径中tif文件是否存在
    if not os.path.exists(img_path):
        print("影像不存在")
        return

    # 读取tif
    data = Image.open(img_path).convert('RGB')
    data = np.array(data, dtype='float32').transpose((2, 0, 1)) / 255.0
    # 给data增加一个batch维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.from_numpy(data).unsqueeze(0).to(device)

    predict(data, MODEL)

    # ### 测试模型是否可用 ###
    # test_dataset = build_dataset(DATASET_NAME, "test", window_size=WINDOW_SIZE)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # MIoU = test(MODEL, test_dataloader)
    # print(f"mIoU: {MIoU:.2f}")
    # ### 测试模型是否可用 End ###


# 加载模型
def load_model():
    global MODEL
    if MODEL is not None:
        return

    MODEL = DINOSegment(pretrained_model_name,
                        backbone_weights=backbone_weights,
                        n_classes=N_CLASSES,
                        window_size=WINDOW_SIZE)

    MODEL.load_state_dict(torch.load(classification_model_path)["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(device)


def colorize_mask(mask, palette):
    """将mask应用配色方案转换为彩色图像"""
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)

    for label_id, color in palette.items():
        r[mask == label_id] = color[0]
        g[mask == label_id] = color[1]
        b[mask == label_id] = color[2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def visualize_prediction(image,
                         pred,
                         label=None,
                         palette=None,
                         alpha=0.5,
                         save_path=None):
    """可视化预测结果
    
    Args:
        image: 原始图像 (C, H, W)
        pred: 预测结果 (H, W)
        label: 真实标签 (H, W)，可选
        palette: 配色方案
        alpha: mask透明度
        save_path: 保存路径，可选
    """
    # 确保image是(H, W, C)格式
    if image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    # 归一化图像到0-255范围
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # 创建彩色mask
    if palette is not None:
        colored_pred = colorize_mask(pred, palette)
    else:
        # 默认使用随机颜色
        cmap = plt.get_cmap('tab20')
        colored_pred = cmap(pred / pred.max())[:, :, :3]
        colored_pred = (colored_pred * 255).astype(np.uint8)

    # 创建叠加图像
    overlay = image.copy()
    overlay = np.where(pred[..., None] > 0,
                       (1 - alpha) * image + alpha * colored_pred,
                       image).astype(np.uint8)

    # 创建显示图像
    if label is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # 显示原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示预测结果
        axes[1].imshow(colored_pred)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # 显示真实标签
        if palette is not None:
            colored_label = colorize_mask(label, palette)
        else:
            colored_label = cmap(label / label.max())[:, :, :3]
            colored_label = (colored_label * 255).astype(np.uint8)

        axes[2].imshow(colored_label)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')

        # 显示叠加效果
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay Prediction')
        axes[3].axis('off')

        # 添加图例
        if palette is not None:
            patches = []
            for i, label_name in enumerate(LABELS):
                if i in palette:
                    color = np.array(palette[i]) / 255.0
                    patches.append(
                        mpatches.Patch(color=color, label=label_name))
            fig.legend(handles=patches,
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 显示原始图像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 显示预测结果
        axes[1].imshow(colored_pred)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # 显示叠加效果
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Prediction')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
        print("Done")


def predict(input, model):
    model.eval()
    with torch.no_grad():
        pred = slide_inference(input, model, n_output_channels=N_CLASSES)

    pred = np.argmax(pred, axis=1)

    # 将input从tensor转为numpy并移除batch维度
    input_np = input.cpu().numpy()[0]
    pred_np = pred.cpu().numpy()[0]

    visualize_prediction(image=input_np,
                         pred=pred_np,
                         label=None,
                         palette=PALETTE,
                         alpha=0.6,
                         save_path=None)


if __name__ == "__main__":
    print(f"N_CLASSES: {N_CLASSES}")
    image_classification(
        img_path=
        "/home/yyyjvm/SS-datasets/YYYJ_dataset/test/label_masks/1638431685714513920GF720201114.tif"
    )
