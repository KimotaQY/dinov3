# 在 datasets 目录下创建 check_tiff_files.py
import os
from PIL import Image
from pathlib import Path


def check_tiff_files(root_dir="/path/to/YYYJ_dataset"):
    """检查所有 TIFF 文件是否可以正常读取"""

    # 读取 train.txt 和 test.txt
    train_txt_path = f"{root_dir}/train.txt"
    test_txt_path = f"{root_dir}/test.txt"

    error_files = []

    for txt_path in [train_txt_path, test_txt_path]:
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                filenames = [
                    line.strip() for line in f.readlines() if line.strip()
                ]

            print(
                f"\nChecking {len(filenames)} files from {os.path.basename(txt_path)}..."
            )

            for i, filename in enumerate(filenames):
                data_file = f"{root_dir}/images/{filename}"
                label_file = f"{root_dir}/label_masks/{filename}"

                try:
                    # 尝试打开数据文件
                    img = Image.open(data_file).convert('RGB')
                    img.load()  # 强制加载

                    # 尝试打开标签文件
                    label = Image.open(label_file).convert('RGB')
                    label.load()  # 强制加载

                    if (i + 1) % 100 == 0:
                        print(f"  Checked {i+1}/{len(filenames)} files")

                except Exception as e:
                    print(f"\n❌ ERROR at index {i}:")
                    print(f"   Data file: {data_file}")
                    print(f"   Label file: {label_file}")
                    print(f"   Error: {e}")
                    error_files.append({
                        'index': i,
                        'filename': filename,
                        'data_file': data_file,
                        'label_file': label_file,
                        'error': str(e)
                    })

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"Total error files: {len(error_files)}")
    if error_files:
        print(f"\nProblematic files:")
        for err in error_files:
            print(f"  [{err['index']}] {err['filename']}")
            print(f"      Data: {err['data_file']}")
            print(f"      Label: {err['label_file']}")
            print(f"      Error: {err['error']}")
    else:
        print("All files are OK! ✓")
    print(f"{'='*60}")

    return error_files


if __name__ == "__main__":
    # 修改为您的实际路径
    root_dir = "/home/yyyj/SS-datasets/YYYJ_dataset/20260316/"
    check_tiff_files(root_dir)
