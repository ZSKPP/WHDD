import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Dane: R | drift_type = gradual
# =========================

rows = [
    "3d | 30F",
    "3d | 60F",
    "3d | 90F",
    "5d | 30F",
    "5d | 60F",
    "5d | 90F",
    "10d | 30F",
    "10d | 60F",
    "10d | 90F",
    "15d | 30F",
    "15d | 60F",
    "15d | 90F",
]

cols = ["WHDD", "ADWIN", "EDDM", "DDM", "OCDD"]

means = np.array([
    [0.155, 0.486, 0.884, 0.410, 0.901],
    [0.050, 0.427, 0.973, 0.255, 0.766],
    [0.100, 0.459, 0.848, 0.225, 0.663],
    [0.071, 0.356, 0.785, 0.595, 0.847],
    [0.125, 0.330, 0.715, 0.820, 0.671],
    [0.225, 0.361, 0.730, 0.217, 0.593],
    [0.047, 0.125, 0.578, 0.812, 0.739],
    [0.185, 0.120, 0.555, 1.495, 0.578],
    [0.219, 0.107, 0.573, 0.437, 0.404],
    [0.019, 0.396, 0.417, 1.621, 0.666],
    [0.099, 0.802, 1.050, 1.643, 0.478],
    [0.184, 1.021, 0.600, 2.089, 0.081],
])

df = pd.DataFrame(means, index=rows, columns=cols)

# =========================
# Mała heatmapa
# =========================

fig, ax = plt.subplots(figsize=(4.8, 3.8))
im = ax.imshow(df.values, cmap="YlGnBu_r", aspect="auto")

# Ticks
ax.set_xticks(np.arange(len(cols)))
ax.set_yticks(np.arange(len(rows)))

ax.set_xticklabels(cols, fontsize=8.5, fontweight="bold")
ax.set_yticklabels(rows, fontsize=7.5, fontweight="bold")

# Wszystkie wartości w komórkach na biało
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.iloc[i, j]
        ax.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=5.8, color="white"
        )

# Niebieska ramka dla najlepszego wyniku w wierszu
for i in range(df.shape[0]):
    best_j = np.argmin(df.iloc[i].values)
    ax.add_patch(Rectangle(
        (best_j - 0.5, i - 0.5), 1, 1,
        fill=False, edgecolor="blue", linewidth=1.3
    ))

# Siatka
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
ax.tick_params(which="minor", bottom=False, left=False)

# Pasek kolorów
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.ax.tick_params(labelsize=6.5)
cbar.set_label("R (lower is better)", fontsize=7.5, fontweight="bold")

# Tytuł
ax.set_title("R metric (gradual drift)", fontsize=8.5, fontweight="bold", pad=5)

plt.tight_layout(pad=0.5)
plt.show()

fig.savefig("heatmap_R_gradual_WHDD_small_white_text.png", dpi=180, bbox_inches="tight")