import torch
from PIL import Image

from model import ResNet
from dataset import transform, dataset

# 创建模型
model = ResNet(num_classes=6)

# 加载最佳模型参数
model.load_state_dict(
    torch.load(
        "outputs/best_resnet.pth",
        map_location="cpu"
    )
)

# 切换到评估模式
model.eval()

# 类别名称
class_names = dataset.classes

# 要预测的图片
image_path = "data/cardboard/cardboard1.jpg"

# 打开图片
image = Image.open(image_path).convert("RGB")

# 图片预处理
image = transform(image)

# 增加 batch 维度
image = image.unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(image)

    # 转换成概率
    probabilities = torch.softmax(output, dim=1)

# 找到概率最高的类别
predicted = probabilities.argmax(dim=1).item()

predicted_class = class_names[predicted]

print("图片:", image_path)
print("预测结果:", predicted_class)

print("\n各类别概率:")

for i, class_name in enumerate(class_names):
    probability = probabilities[0][i].item()

    print(
        class_name,
        f"{probability * 100:.2f}%"
    )