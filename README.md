# 灵境 · Python AI 实验项目

实验组第一阶段考核项目，使用 NumPy、Pandas、Matplotlib 和 PyTorch。项目要求手动编写 Residual Block、组装 ResNet，并完成训练、模型保存以及新图片分类预测。

当前仅准备项目目录、文档和空的 Python 文件，尚未实现业务功能。

## 考核目标（待实现）

1. 手动编写 Residual Block（残差块）。
2. 使用手写残差块组装 ResNet 图像分类模型。
3. 完成模型训练。
4. 保存训练后的模型。
5. 加载已保存的模型，对新图片进行分类预测。

## 目录结构

```text
lingjing/
├── data/                 # 原始数据与处理后的数据
│   └── .gitkeep          # 保留空目录
├── src/                  # 以下 Python 文件当前均为空
│   ├── .gitkeep
│   ├── data_analysis.py  # 数据分析与可视化
│   ├── dataset.py        # 数据集加载与图片预处理
│   ├── model.py          # 手写 Residual Block 与 ResNet 组装
│   ├── train.py          # 模型训练与保存
│   └── predict.py        # 模型加载与新图片分类预测
├── outputs/              # 实验生成的模型、图表与评估结果
│   └── .gitkeep
├── README.md             # 项目说明与环境准备方法
├── requirements.txt      # Python 依赖
└── .gitignore            # Git 忽略规则
```

## 环境准备

建议使用 Python 3.10 或更新版本，并为本项目创建独立的虚拟环境。

在项目根目录打开 PowerShell，执行：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续使用 `.venv` 中的 Python 运行项目；也可在编辑器中将其选择为 Python 解释器。

## 依赖说明

`requirements.txt` 仅包含以下五个依赖，暂不固定版本：

- `numpy`：数组与数值计算。
- `pandas`：数据分析与表格处理。
- `matplotlib`：数据与实验结果可视化。
- `torch`：使用 PyTorch 手写 Residual Block、组装 ResNet，并完成训练、模型保存与预测。
- `torchvision`：图像数据集与图片预处理工具。

本次仅更新依赖清单，尚未安装依赖。

## 目录使用约定

- 数据文件放入 `data/`，保留原始数据，避免直接覆盖。
- 后续业务代码放入 `src/`。
- 模型文件、图表、评估报告等实验产物放入 `outputs/`。
- 虚拟环境、缓存、凭据及 `data/`、`outputs/` 中的内容默认不纳入 Git；两个目录中的 `.gitkeep` 除外。
- API 密钥等敏感信息通过环境变量提供，不写入代码或 README。

## 后续待补充

后续按照上述考核目标实现业务功能，并补充数据来源、类别定义、运行命令、评估指标和实验结果。
