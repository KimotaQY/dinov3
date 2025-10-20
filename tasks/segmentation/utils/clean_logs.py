# 清理无效的log文件

import os
import shutil


def clean_logs(log_dir):
    # 获取目录下的所有文件夹
    dirs = [
        d for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    # 判断文件夹中是否存在.pth后缀文件
    for dir_name in dirs:
        dir_path = os.path.join(log_dir, dir_name)
        pth_files = [f for f in os.listdir(dir_path) if f.endswith('.pth')]
        if not pth_files:
            # 如果不存在.pth后缀文件，则删除该文件夹
            shutil.rmtree(dir_path)
            print(f"Deleted empty directory: {dir_path}")


if __name__ == "__main__":
    log_dir = "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation/logs"
    clean_logs(log_dir)
