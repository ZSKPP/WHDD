import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# Dane: R | drift_type = abrupt
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
    [0.170, 0.553, 0.847, 0.125, 0.925],
    [0.115, 0.573, 0.936, 0.050, 0.886],
    [0.075, 0.538, 0.792, 0.150, 0.674],
    [0.095, 0.461, 0.759, 0.175, 0.888],
    [0.095, 0.473, 1.069, 0.467, 0.833],
    [0.132, 0.470, 0.764, 0.208, 0.646],
    [0.084, 0.168, 0.587, 0.954, 0.807],
    [0.088, 0.210, 0.627, 0.563, 0.688],
    [0.238, 0.204, 0.576, 0.687, 0.549],
    [0.050, 0.186, 0.494, 1.348, 0.726],
    [0.076, 0.463, 0.343, 1.279, 0.621],
    [0.121, 0.581, 0.490, 1.507, 0.441],
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
ax.set_title("R metric (abrupt drift)", fontsize=8.5, fontweight="bold", pad=5)

plt.tight_layout(pad=0.5)
plt.show()

fig.savefig("heatmap_R_abrupt_WHDD_small_white_text.png", dpi=180, bbox_inches="tight")