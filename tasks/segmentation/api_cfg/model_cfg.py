# 模型配置
MODEL = None
pretrained_model_name = "/home/yyyj/Checkpoints/facebook/dinov3-vitl16-pretrain-sat493m"
backbone_weights = "/home/yyyj/Checkpoints/facebook/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"

classification_model_path = "/home/yyyj/SS-projects/dinov3/tasks/segmentation/logs/YYYJ_20251031_062617/dinoseg_YYYJ_e95_mIoU28.02.pth"

LOG_FOLDER = "YYYJ_20251031_062617"  # 对应的模型检查点

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

# 影像文件路径
FILE_ROOT = "/home/yyyj/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105/"
