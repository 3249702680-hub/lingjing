import matplotlib.pyplot as plt
classes = ["paper", "glass", "metal"]
counts = [220, 170, 110]
plt.bar(classes, counts)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.savefig("outputs/class_distribution.png")
plt.show()