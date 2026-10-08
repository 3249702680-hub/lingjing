import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import train_loader, val_loader
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

# 训练轮数
num_epochs = 5

# 记录每一轮的数据
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# 记录当前最好的验证准确率
best_val_acc = 0.0

for epoch in range(num_epochs):
    # =====================
    # 训练阶段
    # =====================
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累加 loss
        train_loss += loss.item()

        # 找出预测类别
        predicted = outputs.argmax(dim=1)

        # 统计总图片数和预测正确数
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # 计算训练集平均 loss 和准确率
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # =====================
    # 验证阶段
    # =====================
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # 前向传播
            outputs = model(images)

            # 计算验证 loss
            loss = criterion(outputs, labels)

            # 累加 loss
            val_loss += loss.item()

            # 找出预测类别
            predicted = outputs.argmax(dim=1)

            # 统计总图片数和预测正确数
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # 计算验证集平均 loss 和准确率
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # 保存这一轮的数据
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # 如果这一轮验证准确率最好，就保存模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            "outputs/best_resnet.pth"
        )
        print("保存新的最佳模型")

    # 输出这一轮结果
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

# =====================
# 画 Loss 曲线
# =====================
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("outputs/loss_curve.png")
plt.close()

# =====================
# 画 Accuracy 曲线
# =====================s
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accs, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("outputs/accuracy_curve.png")
plt.close()

print("训练完成")
print("最佳验证准确率:", best_val_acc)
print("最佳模型已保存到 outputs/best_resnet.pth")