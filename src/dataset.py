import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
# 1. 定义图片处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# 2. 读取整个数据集
dataset = ImageFolder(
    root="data",
    transform=transform
)
# 3. 查看数据集基本信息
print("类别:", dataset.classes)
print("类别编号:", dataset.class_to_idx)
print("图片总数:", len(dataset))
# 4. 划分训练集和验证集
train_data, val_data = random_split(
    dataset,
    [2022, 505],
    generator=torch.Generator().manual_seed(42)
)
# 5. 创建训练集 DataLoader
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)
# 6. 创建验证集 DataLoader
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False
)
# 7. 查看训练集和验证集数量
print("训练集图片数:", len(train_data))
print("验证集图片数:", len(val_data))
# 8. 取出训练集的第一个 batch
train_images, train_labels = next(iter(train_loader))
print("训练 batch 图片形状:", train_images.shape)
print("训练 batch 标签形状:", train_labels.shape)
print("训练 batch 标签:", train_labels)
# 9. 取出验证集的第一个 batch
val_images, val_labels = next(iter(val_loader))
print("验证 batch 图片形状:", val_images.shape)
print("验证 batch 标签形状:", val_labels.shape)
print("验证 batch 标签:", val_labels)