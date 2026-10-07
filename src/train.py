import torch
import torch.nn as nn

from dataset import train_loader
from model import ResNet

# 创建模型
model = ResNet(num_classes=6)

# 多分类损失函数
criterion = nn.CrossEntropyLoss()

# Adam 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 切换到训练模式
model.train()

# 只训练前 5 个 batch，检查训练流程是否正常
for batch_idx, (images, labels) in enumerate(train_loader):
    # 前向传播
    outputs = model(images)

    # 计算损失
    loss = criterion(outputs, labels)

    # 清空上一轮梯度
    optimizer.zero_grad()

    # 反向传播S
    loss.backward()

    # 更新模型参数
    optimizer.step()

    print(
        "batch:",
        batch_idx + 1,
        "loss:",
        loss.item()
    )

    # 今天只测试 5 个 batch
    if batch_idx == 4:
        break

print("训练流程测试完成")