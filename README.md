# Lingjing Garbage Classification

灵境竞赛组后端 AI 方向第一阶段考核项目。

本项目基于 TrashNet 垃圾分类数据集，使用 Python、NumPy、Pandas、Matplotlib 和 PyTorch 完成数据分析、数据加载、手写 ResNet、模型训练与验证、模型保存以及垃圾图片分类预测。

## 1. 项目内容

项目主要实现：

- 使用 NumPy、Pandas、Matplotlib 对垃圾数据集进行统计和可视化
- 使用 ImageFolder 和 DataLoader 完成图片读取与批量加载
- 将数据集划分为训练集和验证集
- 手动实现 ResidualBlock
- 手动组装 ResNet 图像分类网络
- 使用 CrossEntropyLoss 和 Adam 完成模型训练
- 在验证集上评估模型并保存最佳模型
- 绘制 Loss 和 Accuracy 曲线
- 加载训练好的模型进行图片分类预测
- 支持通过命令行输入新的图片进行预测

## 2. 数据集

项目使用 TrashNet 数据集，共 2527 张图片，包含 6 个类别：

| Class | Count |
| --- | ---: |
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 |

训练集与验证集划分：

```text
Train: 2022
Validation: 505
```

原始数据集未上传至 GitHub。

## 3. 项目结构

```text
lingjing/
├── data/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
├── src/
│   ├── data_analysis.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── test_images/
│   └── test.jpg
│
├── outputs/
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── .gitignore
```

文件说明：

```text
data_analysis.py  数据统计与可视化
dataset.py        图片预处理、数据集划分与 DataLoader
model.py          手写 ResidualBlock 和 ResNet
train.py          模型训练、验证、保存与训练曲线
predict.py        加载模型并预测指定图片
```

## 4. 模型

项目主体模型为手动实现的 ResNet。

ResidualBlock 使用残差连接：

```text
F(x) + shortcut(x)
```

当主分支与 shortcut 的输入输出维度不一致时，使用 1×1 卷积进行维度匹配。

模型包含四个残差阶段：

```text
64 → 128 → 256 → 512 channels
```

最后通过全局平均池化和全连接层输出 6 个垃圾类别的分类结果。

## 5. 环境配置

创建虚拟环境：

```powershell
python -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements.txt
```

`requirements-lock.txt` 保存项目开发环境中的完整依赖版本。

## 6. 运行项目

数据分析：

```powershell
python .\src\data_analysis.py
```

查看数据加载结果：

```powershell
python .\src\dataset.py
```

测试 ResNet 前向传播：

```powershell
python .\src\model.py
```

训练模型：

```powershell
python .\src\train.py
```

对新图片进行预测：

```powershell
python .\src\predict.py .\test_images\test.jpg
```

也可以替换为其他图片路径：

```powershell
python .\src\predict.py 图片路径
```

## 7. 实验结果

本次训练使用：

```text
训练集：2022 张
验证集：505 张
```

当前实验最佳验证准确率约为：

```text
44.75%
```

训练过程中会保存验证效果最好的模型：

```text
outputs/best_resnet.pth
```

并生成训练曲线：

```text
outputs/loss_curve.png
outputs/accuracy_curve.png
```

在数据集之外的新塑料瓶图片上进行测试：

```text
Prediction: plastic
```

各类别概率：

```text
cardboard 11.80%
glass      3.04%
metal      20.50%
paper      17.57%
plastic    46.87%
trash      0.21%
```

## 8. Git

项目使用 Git 和 GitHub 进行版本管理，并在开发过程中持续提交代码，保留环境搭建、数据处理、模型实现、训练和预测等阶段的开发记录。

GitHub Repository：

```text
https://github.com/3249702680-hub/lingjing
```

## 9. 说明

本项目主体使用手写 ResNet 完成图像分类，同时学习了预训练模型和迁移学习的基本思想。

迁移学习部分用于理解模型复用，不替代本项目的手写 ResNet。