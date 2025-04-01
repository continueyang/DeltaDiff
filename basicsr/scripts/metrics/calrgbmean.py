from PIL import Image
import os

def calculate_rgb_mean(image_path):
    # 打开图片
    image = Image.open(image_path)
    # 获取图片的RGB值
    rgb_values = list(image.getdata())
    # 计算每个通道的总和
    total_r = sum([rgb[0] for rgb in rgb_values])
    total_g = sum([rgb[1] for rgb in rgb_values])
    total_b = sum([rgb[2] for rgb in rgb_values])
    # 计算均值
    num_pixels = len(rgb_values)
    mean_r = total_r / num_pixels
    mean_g = total_g / num_pixels
    mean_b = total_b / num_pixels

    return mean_r, mean_g, mean_b

def calculate_folder_rgb_mean(folder_path):
    # 获取文件夹中所有图片文件的路径
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("文件夹中没有图片文件。")
        return

    # 初始化累加器
    total_mean_r = 0
    total_mean_g = 0
    total_mean_b = 0

    # 遍历每张图片并累加RGB均值
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        mean_r, mean_g, mean_b = calculate_rgb_mean(image_path)
        total_mean_r += mean_r
        total_mean_g += mean_g
        total_mean_b += mean_b

    # 计算平均RGB均值
    num_images = len(image_files)
    avg_mean_r = total_mean_r / num_images
    avg_mean_g = total_mean_g / num_images
    avg_mean_b = total_mean_b / num_images

    return avg_mean_r, avg_mean_g, avg_mean_b

# 文件夹路径
folder_path = "D:\dataset\celeba\data\CELEBA\TRAIN_HR"

# 计算文件夹下所有图片的RGB均值
avg_rgb_mean = calculate_folder_rgb_mean(folder_path)

print("文件夹下所有图片的RGB均值：", avg_rgb_mean)