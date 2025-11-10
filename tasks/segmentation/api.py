import asyncio
import importlib
import logging
import logging.config
import os
import pickle
import sys
import time

# fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse

import requests

# 添加matplotlib用于可视化
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# 添加GeoTIFF处理和WKT生成所需库
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely import wkt
import json

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

from utils.inference import slide_inference

# 配置文件
from api_cfg import *

# 选择对应检查点 - 使用变量方式导入
# 从环境变量获取项目ID，如果没有设置则使用默认值
PROJECT_ID = os.environ.get('DINO_PROJECT_ID', LOG_FOLDER)
# 动态导入模块
train_distr_module = importlib.import_module(
    f'logs.{PROJECT_ID}.proj_files.train_distr')
models_dino_segment_module = importlib.import_module(
    f'logs.{PROJECT_ID}.proj_files.models.dino_segment')

# 从动态导入的模块中获取需要的变量和类
LABELS = train_distr_module.LABELS
N_CLASSES = train_distr_module.N_CLASSES
WINDOW_SIZE = train_distr_module.WINDOW_SIZE
DATASET_NAME = train_distr_module.DATASET_NAME
WEIGHTS = train_distr_module.WEIGHTS
test = train_distr_module.test
DINOSegment = models_dino_segment_module.DINOSegment

# 配置日志
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("dino_segmentation_api")
task_logger = logging.getLogger("dino_segmentation_tasks")

app = FastAPI(title="DINOv3 Segmentation API",
              description="API for image segmentation using DINOv3 model",
              version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """在应用启动时加载模型"""
    logger.info("Starting up DINOv3 Segmentation API")
    load_model()
    # 从磁盘恢复队列和任务存储
    restore_queue_from_disk()
    # 启动后台队列处理任务
    asyncio.create_task(process_queue())


@app.on_event("shutdown")
async def shutdown_event():
    """在应用关闭时保存队列和任务存储到磁盘"""
    save_queue_to_disk()
    logger.info("Queue and task store saved to disk")
    logger.info("Shutting down DINOv3 Segmentation API")


@app.get("/")
async def root():
    """根路径，用于检查 API 是否正常运行"""
    logger.info("Root endpoint accessed")
    return {"message": "DINOv3 Segmentation API is running!"}


@app.get("/health")
async def health_check():
    """健康检查端点"""
    model_status = "loaded" if MODEL is not None else "not loaded"
    logger.info(f"Health check performed - Model status: {model_status}")
    return {
        "status": "healthy",
        "model_status": model_status,
        "classes": int(N_CLASSES) if N_CLASSES else None,
        "queue_size": len(QUEUE)
    }


@app.get("/queue/status")
async def queue_status():
    """获取队列当前状态"""
    pending_tasks = []
    for task_id in QUEUE:
        if task_id in TASK_STORE:
            task_info = TASK_STORE[task_id].copy()
            # 转换时间戳为可读格式
            if 'added_time' in task_info:
                task_info['added_time'] = time.ctime(task_info['added_time'])
            pending_tasks.append(task_info)

    logger.info(f"Queue status requested - Queue length: {len(QUEUE)}")
    return {
        "queue_length": len(QUEUE),
        "pending_tasks": pending_tasks,
        "task_store_size": len(TASK_STORE)
    }


# 加入队列
@app.post("/segment/add")
async def add_task(task_id: str | int = Form(...),
                   img_path: str = Form(...),
                   geojson_roi: str = Form(None)):
    """
    接收上传的任务信息并加入处理队列
    
    参数:
    - task_id: 任务ID (字符串或整数)
    - img_path: 图像文件路径
    - geojson_roi: 可选的GeoJSON格式的感兴趣区域范围
    
    返回:
    - JSON 格式的结果
    """
    logger.info(f"Adding task {task_id} to queue")
    load_model()
    global MODEL
    if MODEL is None:
        logger.error(f"Model not loaded for task {task_id}")
        fail_event(task_id, "Model not loaded", 500, TaskStatus.FAILED)

    # 验证1: 队列长度是否超过限制
    if len(QUEUE) >= QUEUE_CONFIG["queue_size"]:
        logger.warning(f"Queue is full when adding task {task_id}")
        fail_event(task_id, "Queue is full", 429, TaskStatus.FAILED)

    # 验证2: 队列中是否已存在相同task_id的任务
    if str(task_id) in QUEUE:
        logger.warning(f"Task with id {task_id} already exists in queue")
        fail_event(task_id, f"Task with id {task_id} already exists in queue",
                   409, TaskStatus.FAILED)

    if not isinstance(task_id, (str, int)):
        logger.error(f"Invalid task_id for task {task_id}")
        fail_event(task_id, "Invalid task_id. Must be a string or integer.",
                   400, TaskStatus.FAILED)

    if not isinstance(img_path, str):
        logger.error(f"Invalid img_path for task {task_id}")
        fail_event(task_id, "Invalid img_path. Must be a string.", 400,
                   TaskStatus.FAILED)

    # 如果提供了geojson_roi，验证其格式
    roi_geometry = None
    if geojson_roi:
        try:
            import json
            from shapely.geometry import shape
            geojson_data = json.loads(geojson_roi)
            roi_geometry = shape(geojson_data)
            # 验证几何类型
            if not roi_geometry.is_valid:
                logger.error(f"Invalid GeoJSON geometry for task {task_id}")
                fail_event(task_id, "Invalid GeoJSON geometry: not valid", 400,
                           TaskStatus.FAILED)
        except Exception as e:
            logger.error(
                f"Invalid GeoJSON format for task {task_id}: {str(e)}")
            fail_event(task_id, f"Invalid GeoJSON format: {str(e)}", 400,
                       TaskStatus.FAILED)

    # 检查文件是否存在
    file_path = os.path.join(FILE_ROOT, img_path)
    if not os.path.exists(file_path):
        logger.error(f"Image file not found for task {task_id}: {img_path}")
        fail_event(task_id, f"Image file not found: {img_path}", 404,
                   TaskStatus.FAILED)

    # 将任务添加到队列中
    task_info = {
        "task_id": task_id,
        "img_path": file_path,  # 使用完整路径
        "geojson_roi": geojson_roi,
        "added_time": time.time(),
        "status": TaskStatus.NOT_STARTED
    }

    # 存储任务信息
    TASK_STORE[str(task_id)] = task_info
    # 添加到处理队列
    QUEUE.append(str(task_id))

    # 保存队列和任务存储到磁盘
    save_queue_to_disk()

    update_task(task_id, "Task added to queue", TaskStatus.NOT_STARTED)
    logger.info(f"Task {task_id} successfully added to queue")

    return JSONResponse(content={
        "task_id": task_id,
        "message": "Task added to queue."
    })


def save_queue_to_disk():
    """将队列和任务存储保存到磁盘"""
    try:
        # 保存队列
        with open(QUEUE_PERSISTENCE_FILE, 'wb') as f:
            pickle.dump(QUEUE, f)

        # 保存任务存储
        with open(TASK_STORE_PERSISTENCE_FILE, 'wb') as f:
            pickle.dump(TASK_STORE, f)

        logger.info(
            f"Queue and task store saved. Queue size: {len(QUEUE)}, Task store size: {len(TASK_STORE)}"
        )
    except Exception as e:
        logger.error(f"Error saving queue to disk: {e}")


def restore_queue_from_disk():
    """从磁盘恢复队列和任务存储"""
    try:
        # 恢复队列
        if os.path.exists(QUEUE_PERSISTENCE_FILE):
            with open(QUEUE_PERSISTENCE_FILE, 'rb') as f:
                restored_queue = pickle.load(f)
                QUEUE.clear()
                QUEUE.extend(restored_queue)
            logger.info(f"Restored queue with {len(QUEUE)} tasks")

        # 恢复任务存储
        if os.path.exists(TASK_STORE_PERSISTENCE_FILE):
            with open(TASK_STORE_PERSISTENCE_FILE, 'rb') as f:
                restored_task_store = pickle.load(f)
                TASK_STORE.clear()
                TASK_STORE.update(restored_task_store)
            logger.info(f"Restored task store with {len(TASK_STORE)} tasks")

        # 删除已保存的文件（可选，根据需求决定是否删除）
        # os.remove(QUEUE_PERSISTENCE_FILE)
        # os.remove(TASK_STORE_PERSISTENCE_FILE)

    except Exception as e:
        logger.error(f"Error restoring queue from disk: {e}")
        # 初始化为空队列和任务存储
        QUEUE.clear()
        TASK_STORE.clear()


async def process_queue():
    """后台任务：周期性检查并处理队列中的任务"""
    while True:
        try:
            if QUEUE:  # 如果队列不为空
                task_id = QUEUE.pop(0)  # 取出第一个任务
                # 每次队列变化后保存到磁盘
                save_queue_to_disk()

                if task_id in TASK_STORE:
                    task_info = TASK_STORE[task_id]
                    try:
                        task_logger.info(f"Processing task {task_id}")
                        # 更新任务状态
                        task_info['status'] = TaskStatus.PROCESSING
                        # 每次任务状态变化后保存到磁盘
                        save_queue_to_disk()

                        await segment_image(
                            task_id=task_info['task_id'],
                            img_path=task_info['img_path'],
                            geojson_roi=task_info['geojson_roi'])
                        # 任务完成后从存储中移除
                        if task_id in TASK_STORE:
                            del TASK_STORE[task_id]
                        # 任务完成后再保存一次
                        save_queue_to_disk()
                        task_logger.info(
                            f"Task {task_id} completed successfully")
                    except Exception as e:
                        error_msg = f"Error processing task {task_id}: {str(e)}"
                        task_logger.error(error_msg)
                        if task_id in TASK_STORE:
                            TASK_STORE[task_id]['status'] = TaskStatus.FAILED
                        # 发生错误也保存状态
                        save_queue_to_disk()
                else:
                    logger.warning(f"Task {task_id} not found in task store")
            else:
                # 如果队列为空，等待一段时间再检查
                await asyncio.sleep(QUEUE_CONFIG.get("check_interval", 5))
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}")
            await asyncio.sleep(QUEUE_CONFIG.get("check_interval", 5))


async def segment_image(task_id: str | int = Form(...),
                        img_path: str = Form(...),
                        geojson_roi: str = Form(None)):
    """
    处理分割任务
    
    参数:
    - task_id: 任务ID (字符串或整数)
    - img_path: 图像文件路径
    - geojson_roi: 可选的GeoJSON格式的感兴趣区域范围
    
    """
    global MODEL

    try:
        task_logger.info(f"Starting segmentation for task {task_id}")
        # 检查文件是否存在
        task_logger.debug(f"Checking image file: {img_path}")
        if not os.path.exists(img_path):
            task_logger.error(f"Image file not found: {img_path}")
            update_task(task_id, f"Image file not found: {img_path}",
                        TaskStatus.FAILED)

        # 读取GeoTIFF图像
        update_task(task_id, "Loading Image", TaskStatus.PROCESSING)
        task_logger.debug("Loading image data")
        with rasterio.open(img_path) as src:
            # 读取图像数据
            image_data = src.read()  # 读取所有波段
            profile = src.profile.copy()  # 获取图像的元数据
            task_logger.debug(f"Image metadata: {profile}")

            # 如果是多波段图像，选择前3个波段作为RGB
            if image_data.shape[0] >= 3:
                image_data = image_data[:3, :, :]  # 取前3个波段
            elif image_data.shape[0] == 1:
                # 如果是单波段，复制为3个波段
                image_data = np.repeat(image_data, 3, axis=0)

            # 转换为模型所需的格式 (H, W, C) -> (C, H, W)
            image_data = np.transpose(image_data, (1, 2, 0))

            # 归一化到0-1范围
            image_data = image_data.astype(
                'float32') / 255.0 if image_data.max(
                ) > 1.0 else image_data.astype('float32')

            # 转换为模型所需的格式
            data = np.transpose(image_data,
                                (2, 0, 1))  # (H, W, C) -> (C, H, W)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            data = torch.from_numpy(data).unsqueeze(0).to(device)

        # 进行推理
        update_task(task_id, "Predicting", TaskStatus.PROCESSING)
        task_logger.debug("Starting prediction")
        model = MODEL
        # colored_pred, overlay = predict(data, model)
        pred = predict(data, model)

        # 将预测结果转换为GeoTIFF格式并保存
        pred_np = pred.cpu().numpy()[0]  # 移除batch维度
        pred_class = np.argmax(pred_np, axis=0)  # 获取每个像素的分类结果

        # 更新profile以适应单波段分类结果
        profile.update({'count': 1, 'dtype': rasterio.uint8})

        # 生成输出文件路径
        # base_name = os.path.splitext(os.path.basename(img_path))[0]
        # output_geotiff_path = os.path.join(os.path.dirname(img_path),
        #                                    f"{base_name}_classified.tif")
        # output_wkt_path = os.path.join(os.path.dirname(img_path),
        #                                f"{base_name}_polygons.json")

        # 保存分类结果为GeoTIFF
        # with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
        #     dst.write(pred_class.astype(rasterio.uint8), 1)

        # 生成WKT格式的多边形
        wkt_polygons = generate_wkt_polygons(pred_class, profile)

        # 如果提供了geojson_roi，则裁剪WKT多边形
        if geojson_roi:
            wkt_polygons = clip_wkt_with_geojson(wkt_polygons, geojson_roi)

        # 每类只保留一个多边形 TODO
        # for key, value in wkt_polygons.items():
        #     wkt_polygons[key] = value[:1]
        # print(f"wkt_polygons: {wkt_polygons}")

        # 提交结果
        try:
            task_logger.debug("Posting results")
            post_results(task_id, wkt_polygons)
        except Exception as e:
            error_msg = f"Upload results failed: {str(e)}"
            task_logger.error(error_msg)
            update_task(task_id, f"Upload results failed: {str(e)}",
                        TaskStatus.FAILED)

    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        task_logger.error(error_msg)
        update_task(task_id, f"Error processing image: {str(e)}",
                    TaskStatus.FAILED)


def fail_event(task_id,
               message,
               status_code=500,
               task_status=TaskStatus.FAILED):
    """
    统一错误处理函数
    
    Args:
        task_id: 任务ID
        message: 错误消息
        status_code: HTTP状态码
        task_status: 任务状态
    """
    task_logger.error(f"Task {task_id} failed: {message}")
    update_task(task_id, message, task_status)
    raise HTTPException(status_code=status_code, detail=message)


# 加载模型
def load_model():
    global MODEL
    if MODEL is not None:
        return

    task_logger.info("Loading model")
    MODEL = DINOSegment(pretrained_model_name,
                        backbone_weights=backbone_weights,
                        n_classes=N_CLASSES,
                        window_size=WINDOW_SIZE)

    MODEL.load_state_dict(torch.load(classification_model_path)["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(device)
    task_logger.info("Model loaded successfully")


def predict(input, model):
    model.eval()
    with torch.no_grad():
        pred = slide_inference(input, model, n_output_channels=N_CLASSES)

    return pred


def generate_wkt_polygons(classification_result, profile):
    """将分类结果转换为WKT多边形格式
    
    Args:
        classification_result: 分类结果数组 (H, W)
        profile: GeoTIFF元数据
        
    Returns:
        dict: 每个类别的WKT多边形列表
    """
    task_logger.debug("Generating WKT polygons from classification results")
    # 获取变换矩阵和坐标系
    transform = profile['transform']
    crs = profile['crs']

    polygons_by_class = {}

    # 为每个类别生成多边形
    for class_id in np.unique(classification_result):
        class_id = int(class_id)  # 转换为Python原生int类型
        # if class_id == 0:  # 跳过背景类（如果需要）
        #     continue

        # 创建该类别的二值掩码
        mask = (classification_result == class_id).astype(np.uint8)

        # 生成多边形
        results = ({
            'properties': {
                'class_id': class_id,
                'id': i
            },
            'geometry': s
        } for i, (
            s, v) in enumerate(shapes(mask, mask=mask, transform=transform)))

        # 转换为WKT格式
        polygons = []
        for result in results:
            geom = shape(result['geometry'])
            polygons.append({
                'wkt': wkt.dumps(geom),
                'class_id': class_id,
                'id': int(result['properties']['id'])
            })

        if polygons:
            polygons_by_class[LABELS[int(class_id)]] = polygons

    task_logger.debug(
        f"Generated WKT polygons for {len(polygons_by_class)} classes")
    return polygons_by_class


def clip_wkt_with_geojson(wkt_polygons, geojson_roi):
    """使用GeoJSON ROI裁剪WKT多边形
    
    Args:
        wkt_polygons: WKT多边形字典
        geojson_roi: GeoJSON格式的ROI
        
    Returns:
        dict: 裁剪后的WKT多边形字典
    """
    if not geojson_roi:
        return wkt_polygons

    try:
        task_logger.debug("Clipping WKT polygons with GeoJSON ROI")
        # 解析GeoJSON ROI
        geojson_data = json.loads(geojson_roi)
        roi_geometry = shape(geojson_data)

        # 如果ROI无效，则返回原始多边形
        if not roi_geometry.is_valid:
            return wkt_polygons

        clipped_polygons = {}

        # 对每个类别的多边形进行裁剪
        for class_name, polygons in wkt_polygons.items():
            clipped_polygons[class_name] = []

            for polygon_data in polygons:
                # 解析WKT多边形
                polygon_geom = wkt.loads(polygon_data['wkt'])

                # 执行裁剪操作
                if roi_geometry.intersects(polygon_geom):
                    intersection = roi_geometry.intersection(polygon_geom)

                    # 如果有交集，添加裁剪后的结果
                    if not intersection.is_empty:
                        clipped_polygons[class_name].append({
                            'wkt':
                            wkt.dumps(intersection),
                            'class_id':
                            polygon_data['class_id'],
                            'id':
                            polygon_data['id']
                        })

        task_logger.debug("WKT polygons clipped successfully")
        return clipped_polygons
    except Exception as e:
        # 出现任何错误时返回原始多边形
        error_msg = f"Error clipping WKT polygons with GeoJSON ROI: {e}"
        task_logger.error(error_msg)
        return wkt_polygons


def update_task(id: int, content: str, status: int):
    url = f"{BASE_URL}/app/AiModel/updateTask"

    global TOKEN
    if TOKEN is None:
        TOKEN = get_token()
    headers = {"FAUTH": TOKEN}

    data = {
        "id": id,
        "content": content,
        "status": status,
    }

    response = requests.post(url, headers=headers, data=data)

    result = response.json()
    if response.status_code != 200:
        if result["code"] == 1002 and result["msg"] == "请先登录":
            token = get_token()
            TOKEN = token
            if token is not None:
                update_task(id, content, status)
            else:
                raise RuntimeError("token获取失败")
        else:
            raise RuntimeError(result["msg"])

    return result


def get_token():
    url = f"{BASE_URL}/access/applogin"

    params = {
        "username": "admin",
        "password": "25563918aabbd32dc10158b88fac41ca",
        "userTypeCode": "leader"
    }

    response = requests.post(url, params=params)

    result = response.json()
    message = result.get("msg")
    if response.status_code != 200:
        task_logger.error(f"Token获取失败：{message}")
        return None

    token = result.get("data").get("token")
    task_logger.info(f"Token获取成功：{message}")

    return token


def post_results(task_id, results):
    url = f"{BASE_URL}/app/AiModel/addPointChange"

    global TOKEN
    if TOKEN is None:
        TOKEN = get_token()
    headers = {"FAUTH": TOKEN}

    json_data = json.dumps(results, ensure_ascii=False)
    data = {"taskId": task_id, "geom": json_data}

    response = requests.post(url, headers=headers, data=data)

    result = response.json()
    if response.status_code != 200:
        raise RuntimeError(result["msg"])
    else:
        update_task(task_id, "Segmentation completed.", TaskStatus.COMPLETED)

    return result


if __name__ == "__main__":
    # print(f"N_CLASSES: {N_CLASSES}")
    # image_classification(
    #     img_path=
    #     "/home/yyyj/SS-datasets/YYYJ_dataset/test/label_masks/1638431685714513920GF720201114.tif"
    # )
    # result = update_task(1, "开始预测", 1)
    # print(f"result: {result}")
    # token = get_token()
    # print(f"token获取结果：{token}")

    print("测试异步")

    img_path = "/home/yyyj/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105/1910247754676965376（厂房）/1863848133197434880BJ320240211.tif"
    if not os.path.exists(img_path):
        print("图像不存在")

    load_model()
    # 读取GeoTIFF图像
    with rasterio.open(img_path) as src:
        # 读取图像数据
        image_data = src.read()  # 读取所有波段
        profile = src.profile.copy()  # 获取图像的元数据
        print(f"tif元数据：{profile}")

        # 如果是多波段图像，选择前3个波段作为RGB
        if image_data.shape[0] >= 3:
            image_data = image_data[:3, :, :]  # 取前3个波段
        elif image_data.shape[0] == 1:
            # 如果是单波段，复制为3个波段
            image_data = np.repeat(image_data, 3, axis=0)

        # 转换为模型所需的格式 (H, W, C) -> (C, H, W)
        image_data = np.transpose(image_data, (1, 2, 0))

        # 归一化到0-1范围
        image_data = image_data.astype('float32') / 255.0 if image_data.max(
        ) > 1.0 else image_data.astype('float32')

        # 转换为模型所需的格式
        data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.from_numpy(data).unsqueeze(0).to(device)

    # 进行推理
    model = MODEL
    # colored_pred, overlay = predict(data, model)
    pred = predict(data, model)

    # 将预测结果转换为GeoTIFF格式并保存
    pred_np = pred.cpu().numpy()[0]  # 移除batch维度
    pred_class = np.argmax(pred_np, axis=0)  # 获取每个像素的分类结果

    # 更新profile以适应单波段分类结果
    profile.update({'count': 1, 'dtype': rasterio.uint8})

    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_geotiff_path = os.path.join(os.path.dirname(img_path),
                                       f"{base_name}_classified.tif")
    output_wkt_path = os.path.join(os.path.dirname(img_path),
                                   f"{base_name}_polygons.json")

    # 保存分类结果为GeoTIFF
    with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
        dst.write(pred_class.astype(rasterio.uint8), 1)

    # 生成WKT格式的多边形
    wkt_polygons = generate_wkt_polygons(pred_class, profile)

    # 每类只保留一个多边形
    # for key, value in wkt_polygons.items():
    #     wkt_polygons[key] = value[:1]

    post_results(1, wkt_polygons)

    # url = f"{BASE_URL}/basic/liangsuiji/uploadResult"

    # if TOKEN is None:
    #     TOKEN = get_token()
    # headers = {"FAUTH": TOKEN}

    # # json_data = json.dumps(results, ensure_ascii=False)
    # # json_data = json_data.replace("'", "")
    # # json_data_format = json_data[1:-1]
    # # data = {"taskId": 1, "geom": '{"地基建设":[]}'}
    # data = {"resultData": '{"地基建设":[]}'}

    # response = requests.post(url, headers=headers, data=data)

    # result = response.json()
    # if response.status_code != 200:
    #     raise RuntimeError(result["msg"])
