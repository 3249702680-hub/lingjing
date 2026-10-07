from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 数据集目录
data_dir = Path("data")
# 允许统计的图片格式
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
# 保存类别名称和每类图片数量
classes = []
counts = []
# 遍历 data 目录下的每个内容
for class_dir in sorted(data_dir.iterdir()):
    # 如果不是文件夹，就跳过
    if not class_dir.is_dir():
        continue
    count = 0
    # 遍历当前类别文件夹中的所有文件
    for file in class_dir.iterdir():
        # 是文件，并且后缀属于图片格式
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
    # 保存类别名和图片数量
    classes.append(class_dir.name)
    counts.append(count)
# NumPy：基础统计
counts_array = np.array(counts)
print("总图片数:", counts_array.sum())
print("平均每类图片数:", counts_array.mean())
print("最多类别图片数:", counts_array.max())
print("最少类别图片数:", counts_array.min())
# Pandas：整理成表格
df = pd.DataFrame({
    "class": classes,
    "count": counts
})
print("\n类别统计:")
print(df)
# 保存 CSV
df.to_csv(
    "outputs/class_distribution.csv",
    index=False
)
# Matplotlib：画柱状图
plt.bar(df["class"], df["count"])
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Garbage Class Distribution")
# 保存图片
plt.savefig("outputs/class_distribution.png")
# 显示图片
plt.show()