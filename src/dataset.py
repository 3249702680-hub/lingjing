import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# 定义图片预处理流程：统一尺寸并转成 Tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 读取整个图片数据集
dataset = ImageFolder(
    root="data",
    transform=transform
)

# 按固定随机种子划分训练集和验证集
train_data, val_data = random_split(
    dataset,
    [2022, 505],
    generator=torch.Generator().manual_seed(42)
)

# 训练集 DataLoader：每批 32 张，并打乱顺序
train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

# 验证集 DataLoader：每批 32 张，不打乱
val_loader = DataLoader(
    val_data,
    batch_size=32,
    shuffle=False
)

# 下面这些只在直接运行 dataset.py 时执行
# 如果别的文件 import train_loader，不会执行这些测试代码
if __name__ == "__main__":
    print("类别:", dataset.classes)
    print("类别编号:", dataset.class_to_idx)
    print("图片总数:", len(dataset))
    print("训练集图片数:", len(train_data))
    print("验证集图片数:", len(val_data))

    # 取训练集第一个 batch，检查数据形状
    train_images, train_labels = next(iter(train_loader))
    print("训练 batch 图片形状:", train_images.shape)
    print("训练 batch 标签形状:", train_labels.shape)
    print("训练 batch 标签:", train_labels)

    # 取验证集第一个 batch，检查数据形状
    val_images, val_labels = next(iter(val_loader))
    print("验证 batch 图片形状:", val_images.shape)
    print("验证 batch 标签形状:", val_labels.shape)
    print("验证 batch 标签:", val_labels)