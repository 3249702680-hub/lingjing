import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
counts = [403, 501, 410, 594, 482, 137]
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
df.to_csv("outputs/class_distribution.csv", index=False)
# Matplotlib：画柱状图
plt.bar(df["class"], df["count"])
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Garbage Class Distribution")
plt.savefig("outputs/class_distribution.png")
plt.show()