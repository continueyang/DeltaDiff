import os
import random
import shutil

def create_image_pairs(low_res_folder, high_res_folder, output_folder, output1_folder,n):
    # 获取低分辨率和高分辨率图像文件夹中的文件列表
    low_res_images = os.listdir(low_res_folder)
    high_res_images = os.listdir(high_res_folder)

    # 随机选择n个图像文件名
    selected_images = random.sample(list(zip(low_res_images, high_res_images)), n)

    # 创建新文件夹
    os.makedirs(output1_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    # 复制所选图像对到新文件夹
    for low_res, high_res in selected_images:
        shutil.copy(os.path.join(low_res_folder, low_res), os.path.join(output_folder, low_res))
        shutil.copy(os.path.join(high_res_folder, high_res), os.path.join(output1_folder, high_res))

# 设置文件夹路径和需要的图像对数量
low_res_folder_path = 'D:\dataset\celeba\data\CELEBA\TRAIN_LR'
high_res_folder_path = 'D:\dataset\celeba\data\CELEBA\TRAIN_HR'
output_folder_path = r'D:\dataset\celeba\data\CELEBA\randomnew'
output1_folder_path = r'D:\dataset\celeba\data\CELEBA\randomnewlr'
number_of_image_pairs = 10000  # 修改为所需的图像对数量

# 创建图像对
create_image_pairs(low_res_folder_path, high_res_folder_path, output_folder_path,output1_folder_path, number_of_image_pairs)