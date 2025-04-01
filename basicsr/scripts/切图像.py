from PIL import Image
import os
def split_image(image_path, output_path, num_rows, num_cols):
    # 打开图像
    image = Image.open(image_path)
    width, height = image.size

    # 计算子图像的宽度和高度
    sub_width = width // num_cols
    sub_height = height // num_rows

    # 创建一个目录用于保存子图像
    os.makedirs(output_path, exist_ok=True)

    # 分割图像并保存子图像
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * sub_width
            upper = row * sub_height
            right = left + sub_width
            lower = upper + sub_height
            sub_image = image.crop((left, upper, right, lower))

            # 保存子图像
            sub_image.save(f"{output_path}/subimage_{row}_{col}.png")

if __name__ == "__main__":
    input_image_path = r"C:\Users\yang\Desktop\BasicSR-gnn\201465.png"  # 输入图像的路径
    output_dir = r"C:\Users\yang\Desktop\BasicSR-gnn\img1"  # 保存子图像的目录
    num_rows = 4  # 希望将图像分成的行数
    num_cols = 4  # 希望将图像分成的列数

    split_image(input_image_path, output_dir, num_rows, num_cols)