# type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm = np.array(
    [
        [125, 10, 4, 3, 5],
        [4, 412, 2, 19, 7],
        [1, 7, 307, 2, 2],
        [1, 8, 3, 416, 0],
        [4, 7, 6, 1, 150],
    ]
)


categories = ["Drink", "Listen", "Phone", "Trance", "Write"]
cmpic = sns.heatmap(
    cm,
    fmt="d",
    annot=True,
    cmap="Blues",
    xticklabels=categories,
    yticklabels=categories,
)

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Confusion Matrix")

plt.savefig(
    "plot_cm.png",
    dpi=300,
)
