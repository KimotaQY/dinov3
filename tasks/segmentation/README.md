```bash
torchrun --nproc_per_node=2 ./tasks/segmentation/train_distr.py
```

# 重新挂载并指定用户和组权限
```bash
sudo mount -t cifs //10.196.21.55/20251105 /home/yyyj/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105 -o username=dell,password=guojia1995,uid=1000,gid=1000,file_mode=0777,dir_mode=0777
sudo umount  /home/yyyj/SS-datasets/YYYJ_dataset/Desktop-vvgck54/20251105
```