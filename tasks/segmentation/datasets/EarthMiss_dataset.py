import os
import random
import sys
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
import torchvision.transforms.functional as TF

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)
from utils.transform import *

palette = {
    0: (255, 255, 255),  # Background (white)
    1: (255, 0, 0),  # Building (red)
    2: (255, 255, 0),  # Road (yellow)
    3: (0, 0, 255),  # Water (blue)
    4: (159, 129, 183),  # Barren (purple)
    5: (0, 255, 0),  # Forest (green)
    6: (255, 195, 128),  # Agricultural (deeper yellow)
    7: (165, 0, 165),  # Playground (deeper purple)
    8: (0, 0, 0)
}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


class EarthMiss_Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            citys,
            rgb_dir,
            label_dir,
            data_type,
            window_size=(224, 224),
            normalize_type=None,
            sar_dir=None,
    ):
        super(EarthMiss_Dataset, self).__init__()

        self.data_type = data_type
        self.window_size = window_size

        # List of files
        self.rgb_files = []
        self.sar_files = []
        self.label_files = []
        for city in citys:
            data_rgb_dir = rgb_dir.format(city)
            data_sar_dir = sar_dir.format(
                city) if sar_dir is not None else None
            data_label_dir = label_dir.format(city)
            # 读取data_dir下的所有文件名
            self.rgb_files.extend([
                os.path.join(data_rgb_dir, f) for f in os.listdir(data_rgb_dir)
                if f.endswith(".tif")
            ])
            self.sar_files.extend([
                os.path.join(data_sar_dir, f) for f in os.listdir(data_sar_dir)
                if f.endswith(".tif")
            ])
            self.label_files.extend([
                os.path.join(data_label_dir, f)
                for f in os.listdir(data_label_dir) if f.endswith(".tif")
            ])

            # # 预检查文件有效性，过滤掉损坏的文件
            # valid_indices = []
            # for i, (rgb_file, label_file) in enumerate(
            #         zip(self.rgb_files, self.label_files)):
            #     try:
            #         # 尝试打开文件检查是否有效
            #         with imread(rgb_file) as img:
            #             img.verify()
            #         with imread(label_file) as img:
            #             img.verify()
            #         if self.sar_files and i < len(self.sar_files):
            #             with imread(self.sar_files[i]) as img:
            #                 img.verify()
            #         valid_indices.append(i)
            #     except Exception:
            #         # 跳过损坏的文件
            #         print(f"Warning: Skipping corrupted file {rgb_file}")
            #         continue

        # Sanity check : raise an error if some files do not exist
        for file in self.rgb_files + self.label_files + self.sar_files:
            if not os.path.exists(file) and not os.path.isfile(file):
                raise ValueError(f"File {file} does not exist")

        # Initialize cache dicts
        self.rgb_cache = {}
        self.label_cache = {}
        self.sar_cache = {}

        if normalize_type == "geo":
            self.imagenet_mean = (0.430, 0.411, 0.296)
            self.imagenet_std = (0.213, 0.156, 0.143)
        elif normalize_type == "common":
            self.imagenet_mean = (0.485, 0.456, 0.406)
            self.imagenet_std = (0.229, 0.224, 0.225)
        else:
            self.imagenet_mean = None
            self.imagenet_std = None

    def __len__(self):
        interval_num = (256**2 / self.window_size[0]**2) * 16  # 256尺寸时为*16
        data_len = len(self.rgb_files
                       ) * interval_num if self.data_type == 'train' else len(
                           self.rgb_files)
        return int(data_len)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            random_idx = random.randint(0, len(self.rgb_files) - 1)

            if random_idx in self.rgb_cache.keys():
                data = self.rgb_cache[random_idx]
            else:
                data = imread(self.rgb_files[random_idx])
                self.rgb_cache[random_idx] = data

            if random_idx in self.label_cache.keys():
                label = self.label_cache[random_idx]
            else:
                label = imread(self.label_files[random_idx]).astype(np.int64)
                label = label - 1

                self.label_cache[random_idx] = label

            sar = None
            if random_idx in self.sar_cache.keys():
                sar = self.sar_cache[random_idx]
            elif len(self.sar_files) > 0:
                sar = imread(self.sar_files[random_idx])

                self.sar_cache[random_idx] = sar

            # Get a random patch
            # data = data.transpose((2, 0, 1))
            x1, x2, y1, y2 = self.get_random_pos(data, self.window_size)
            if isinstance(data, np.ndarray):
                data = data[x1:x2, y1:y2, :]
                label = label[x1:x2, y1:y2]
                sar = sar[x1:x2, y1:y2] if sar is not None else None
            elif isinstance(data, Image.Image):
                data = data.crop(
                    (y1, x1, y2, x2))  # PIL使用(left, upper, right, lower)
                label = label.crop((y1, x1, y2, x2))
                sar = sar.crop((y1, x1, y2, x2)) if sar is not None else None

            # 弱增强
            # data = data.transpose((1, 2, 0))
            data, label, sar = resize(data, label, sar, ratio_range=(0.5, 2.0))
            data, label, sar = crop(data, label, sar, size=self.window_size[0])
            data, label, sar = hflip(data, label, sar, p=0.5)
            data, label, sar = vflip(data, label, sar, p=0.5)
            # data, label = rotate(data, label, p=0.5)

            # data = color_jitter(data, p=0.8)
            # data = grayscale(data, p=0.2)
            # data = blur(data, p=0.5)

            # convert to np.array
            # data = np.array(data, dtype='float32').transpose((2, 0, 1))
            # label = np.array(label)
            # label = np.asarray(self.convert_from_color(label), dtype='int64')
        else:
            data = imread(self.rgb_files[idx])
            # data = np.array(data, dtype='float32').transpose((2, 0, 1))

            label = imread(self.label_files[idx]).astype(np.int64)
            label = label - 1
            # label_arr = np.array(label_img)
            # label = np.asarray(self.convert_from_color(label_arr),
            #                    dtype='int64')

            sar = imread(self.sar_files[idx]) if len(
                self.sar_files) > 0 else None

        # 最终转换为tensor前确保数据连续且格式正确
        if isinstance(data, np.ndarray):
            # 确保数据连续且转换为(C,H,W)格式用于tensor转换
            data = np.ascontiguousarray(data)
            label = np.ascontiguousarray(label)

        data = TF.to_tensor(data)  # Convert image to tensor
        if self.imagenet_mean is not None:
            data = TF.normalize(
                data, self.imagenet_mean,
                self.imagenet_std)  # Normalize with ImageNet mean and std

        # 确保标签值在有效范围内
        # invalid_mask = (label < 0)
        # if invalid_mask.any():
        #     # print(f"Found {invalid_mask.sum().item()} invalid label values")
        #     # print(f"Invalid values: {label[invalid_mask].unique()}")
        #     # 将无效值替换为0或其他默认值
        #     label = label.clone()  # 创建副本避免就地修改
        #     label[invalid_mask] = 255

        if sar is not None:
            if isinstance(sar, np.ndarray):
                # 获取最小值和最大值
                min_val = np.min(sar)
                max_val = np.max(sar)
                # 防止除零错误
                if max_val > min_val:
                    sar = (sar - min_val) / (max_val - min_val)
                else:
                    # 如果所有像素值都相同，设置为0（或保持原值）
                    sar = np.full_like(
                        sar, 0.0)  # 或者 sar = np.full_like(sar, min_val)

                sar = np.ascontiguousarray(sar)
            else:
                # 处理 PIL Image 对象
                min_val, max_val = sar.getextrema()
                if max_val > min_val:
                    sar = Image.eval(
                        sar, lambda x: (x - min_val) / (max_val - min_val))
                else:
                    sar = Image.eval(sar, lambda x: min_val)

            sar = TF.to_tensor(sar)
            return data, sar, label
        else:
            return data, label

    @staticmethod
    def convert_from_color(arr_3d, palette=invert_palette):
        """ RGB-color encoding to grayscale labels """
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    @staticmethod
    def get_random_pos(img, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        if isinstance(img, np.ndarray):
            W, H = img.shape[:2]
        elif isinstance(img, Image.Image):
            W, H = img.size

        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2
