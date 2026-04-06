# 适用于WHDLD数据集的模型训练代码，使用resnet34模型

### 需要修改的地方
**config.yaml**文件  
**data.root**修改为WHDLD数据集所在的目录  
**data.images_dir**修改为原图相对于root所在的目录  
**data.masks_dir**修改为标签图相对于root所在的目录

**train.epochs**修改为想要训练的轮数

# 分支unet使用原生Unet

# 命令
%cd /kaggle/working  
!rm -rf /kaggle/working/whdld1  
!git clone -b data1600 https://github.com/LIKE9426334946/project.git
%cd /kaggle/working/project
!python3 utils/split.py
!python3 train.py
