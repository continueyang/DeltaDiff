import os
import torch
import clip
from PIL import Image
from torchvision import transforms

# 加载 CLIP 模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 加载预训练的 MUSIQ↑ 模型
musiq_model = torch.hub.load("google-research/musiq", "musiq", pretrained=True)
musiq_model.eval()

# 定义质量提示文本
quality_texts = ["a high quality image", "a low quality image"]
text = clip.tokenize(quality_texts).to(device)

# 图像预处理（MUSIQ↑）
musiq_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 定义图片文件夹路径
image_folder = "path/to/your/image/folder"

# 初始化变量
clip_total_score = 0.0
musiq_total_score = 0.0
image_count = 0

# 遍历文件夹中的所有图片
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        image_path = os.path.join(image_folder, image_name)
        try:
            # 加载图像
            image = Image.open(image_path).convert("RGB")

            # CLIPIQA 计算
            clip_image = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_image)
                text_features = clip_model.encode_text(text)
                logits_per_image, _ = clip_model(clip_image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            clip_score = probs[0][0]  # 高质量的概率

            # MUSIQ↑ 计算
            musiq_image = musiq_preprocess(image).unsqueeze(0)
            with torch.no_grad():
                musiq_score = musiq_model(musiq_image).item()

            # 累加分数
            clip_total_score += clip_score
            musiq_total_score += musiq_score
            image_count += 1

            # 输出当前图片的质量分数
            print(f"Image: {image_name}")
            print(f"  CLIPIQA Quality Score: {clip_score:.4f}")
            print(f"  MUSIQ↑ Quality Score: {musiq_score:.4f}")
            print("-" * 40)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

# 计算并输出均值
if image_count > 0:
    clip_mean_score = clip_total_score / image_count
    musiq_mean_score = musiq_total_score / image_count
    print(f"\nMean CLIPIQA Quality Score for {image_count} images: {clip_mean_score:.4f}")
    print(f"Mean MUSIQ↑ Quality Score for {image_count} images: {musiq_mean_score:.4f}")
else:
    print("No valid images found in the folder.")