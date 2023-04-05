import numpy as np
import matplotlib.pyplot as plt

y_loss = np.load("../../result/y_loss.npy", allow_pickle=True).item()
y_acc = np.load("../../result/y_acc.npy", allow_pickle=True).item()

x = list(range(1, 201))

plt.title("Training and Validation Loss")
plt.plot(x, y_loss["train"], label="Training Loss")
plt.plot(x, y_loss["val"], linestyle="--", label="Validation Loss")
plt.legend(loc="upper right")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./plot_loss.jpg", dpi=300)

plt.cla()


plt.title("Training and Validation Accuracy")
plt.plot(x, y_acc["train"], label="Training Accuracy")
plt.plot(x, y_acc["val"], linestyle="--", label="Validation Accuracy")
plt.legend(loc="lower right")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("./plot_acc.jpg", dpi=300)
