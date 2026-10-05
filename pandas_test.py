import pandas as pd
data = {
    "class": ["paper", "paper", "glass", "glass", "metal"],
    "count": [100, 120, 80, 90, 110]
}
df = pd.DataFrame(data)
print(df)
print(df["count"])
print(df["count"].mean())
print(df["count"].max())
print(df.sort_values("count"))
#df.to_csv("outputs/class_distribution.csv", index=False)
print(df.groupby("class")["count"].sum())
