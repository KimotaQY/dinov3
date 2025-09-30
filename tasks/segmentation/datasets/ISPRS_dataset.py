import os
import random
import numpy as np
import torch
from PIL import Image

palette = {
    0: (255, 255, 255),  # Impervious surfaces (white)
    1: (0, 0, 255),  # Buildings (blue)
    2: (0, 255, 255),  # Low vegetation (cyan)
    3: (0, 255, 0),  # Trees (green)
    4: (255, 255, 0),  # Cars (yellow)
    5: (255, 0, 0),  # Clutter (red)
    6: (0, 0, 0)
}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


class ISRPS_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 ids,
                 data_dir,
                 label_dir,
                 dataset_name,
                 data_type,
                 window_size=(224, 224)):
        super(ISRPS_Dataset, self).__init__()

        self.dataset_name = dataset_name
        self.data_type = data_type
        self.window_size = window_size

        # List of files
        self.data_files = [data_dir.format(id) for id in ids]
        self.label_files = [label_dir.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for file in self.data_files + self.label_files:
            if not os.path.exists(file) and not os.path.isfile(file):
                raise ValueError(f"File {file} does not exist")

        # Initialize cache dicts
        self.data_cache = {}
        self.label_cache = {}

    def __len__(self):
        return len(
            self.data_files) * 100 if self.data_type == 'train' else len(
                self.data_files)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            random_idx = random.randint(0, len(self.data_files) - 1)

            if random_idx in self.data_cache.keys():
                data = self.data_cache[random_idx]
            else:
                if self.dataset_name == 'Potsdam':
                    data = Image.open(
                        self.data_files[random_idx]).convert('RGB')
                    data = np.array(data, dtype='float32').transpose(
                        (2, 0, 1)) / 255.0
                elif self.dataset_name == 'Vaihingen':
                    data = Image.open(
                        self.data_files[random_idx]).convert('RGB')
                    data = np.array(data, dtype='float32').transpose(
                        (2, 0, 1)) / 255.0

                self.data_cache[random_idx] = data

            if random_idx in self.label_cache.keys():
                label = self.label_cache[random_idx]
            else:
                label_img = Image.open(
                    self.label_files[random_idx]).convert('RGB')
                label_arr = np.array(label_img)
                label = np.asarray(self.convert_from_color(label_arr),
                                   dtype='int64')

                self.label_cache[random_idx] = label

            # Get a random patch
            x1, x2, y1, y2 = self.get_random_pos(data, self.window_size)
            data = data[:, x1:x2, y1:y2]
            label = label[x1:x2, y1:y2]
        else:
            data = Image.open(self.data_files[idx]).convert('RGB')
            data = np.array(data, dtype='float32').transpose((2, 0, 1)) / 255.0

            label_img = Image.open(self.label_files[idx]).convert('RGB')
            label_arr = np.array(label_img)
            label = np.asarray(self.convert_from_color(label_arr),
                               dtype='int64')

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
        W, H = img.shape[-2:]
        x1 = random.randint(0, W - w - 1)
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1)
        y2 = y1 + h
        return x1, x2, y1, y2
