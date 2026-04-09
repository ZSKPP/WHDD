import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import numpy as np
from pathlib import Path

# =========================
# DANE
# =========================

D2_gradual = {
    "WHDD": [3.39, 5.60, 13.63, 2.28, 7.88, 12.00, 1.80, 6.09, 6.42, 1.05, 3.30, 4.21],
    "ADWIN": [3.82, 6.49, 4.48, 5.92, 8.06, 5.06, 7.78, 7.83, 9.68, 7.56, 11.03, 15.57],
    "EDDM": [28.72, 41.06, 31.89, 31.26, 43.34, 35.48, 34.47, 28.60, 29.00, 40.32, 26.51, 25.91],
    "DDM": [16.04, 10.53, 14.06, 22.78, 26.62, 16.22, 20.95, 23.67, 11.13, 19.38, 19.25, 38.51],
    "OCDD": [1.33, 1.18, 1.71, 0.90, 0.74, 1.32, 0.95, 1.17, 1.10, 0.90, 1.03, 0.77],
}

D2_abrupt = {
    "WHDD": [0.30, 1.28, 8.68, 0.28, 3.78, 5.94, 1.10, 2.78, 6.09, 0.44, 2.25, 3.08],
    "ADWIN": [3.18, 2.50, 3.99, 3.30, 2.64, 3.18, 5.38, 5.71, 5.21, 6.02, 8.79, 13.29],
    "EDDM": [29.63, 46.33, 32.41, 27.38, 39.00, 28.74, 36.78, 33.01, 19.66, 36.93, 27.89, 27.45],
    "DDM": [7.24, 2.33, 7.72, 10.28, 12.10, 8.70, 19.61, 13.66, 17.36, 17.94, 19.57, 30.40],
    "OCDD": [0.51, 0.39, 0.39, 0.32, 0.12, 0.04, 0.66, 0.57, 0.53, 0.59, 0.42, 0.37],
}

R_gradual = {
    "WHDD": [0.20,0.18,0.16, 0.12,0.11,0.10, 0.09,0.08,0.08, 0.06,0.05,0.05],
    "ADWIN": [0.45,0.44,0.42, 0.43,0.42,0.41, 0.40,0.39,0.38, 0.37,0.36,0.35],
    "EDDM": [0.75,0.77,0.78, 0.74,0.75,0.76, 0.73,0.72,0.71, 0.70,0.69,0.68],
    "DDM": [0.88,0.87,0.86, 0.85,0.84,0.83, 0.82,0.81,0.80, 0.79,0.78,0.77],
    "OCDD": [0.62,0.63,0.64, 0.61,0.60,0.59, 0.58,0.57,0.56, 0.55,0.54,0.53],
}

R_abrupt = {
    "WHDD": [0.18,0.17,0.16, 0.12,0.11,0.10, 0.09,0.08,0.08, 0.06,0.05,0.05],
    "ADWIN": [0.44,0.43,0.42, 0.41,0.40,0.39, 0.38,0.37,0.36, 0.35,0.34,0.33],
    "EDDM": [0.72,0.73,0.74, 0.71,0.72,0.73, 0.70,0.69,0.68, 0.67,0.66,0.65],
    "DDM": [0.66,0.65,0.64, 0.63,0.62,0.61, 0.60,0.59,0.58, 0.57,0.56,0.55],
    "OCDD": [0.74,0.75,0.76, 0.73,0.72,0.71, 0.70,0.69,0.68, 0.67,0.66,0.65],
}

# =========================
# FUNKCJA HULL
# =========================

def draw_hull(ax, x, y, label):
    points = np.column_stack((x, y))

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # domknięcie obszaru
    hull_closed = np.vstack([hull_points, hull_points[0]])

    ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.15)
    ax.fill(hull_closed[:, 0], hull_closed[:, 1], alpha=0.20)

    ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=1.2)
    ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=1.0)

    ax.text(np.mean(x), np.mean(y), label,
            ha='center', va='center', fontsize=10, weight='bold')
# =========================
# PLOT
# =========================

fig, ax = plt.subplots(figsize=(10,7))

for m in D2_gradual:
    ax.scatter(D2_gradual[m], R_gradual[m], marker='o', s=40)
    ax.scatter(D2_abrupt[m], R_abrupt[m], marker='s', s=40)

for m in D2_gradual:
    x = D2_gradual[m] + D2_abrupt[m]
    y = R_gradual[m] + R_abrupt[m]
    draw_hull(ax, x, y, m)

ax.set_xlabel("D2", fontsize=13, fontweight='bold')
ax.set_ylabel("R", fontsize=13, fontweight='bold')

ax.set_title(
    "Trade-off between drift coverage (D2) and alarm consistency (R)",
    fontsize=14, fontweight='bold')


ax.tick_params(axis='both', labelsize=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')  # 🔥 bold tick labels

ax.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0], [0],
           marker='o',
           linestyle='None',
           markerfacecolor='none',
           markeredgecolor='black',
           markersize=9,
           label='Gradual drift'),

    Line2D([0], [0],
           marker='s',
           linestyle='None',
           markerfacecolor='none',
           markeredgecolor='black',
           markersize=9,
           label='Abrupt drift'),
]

legend = ax.legend(
    handles=legend_elements,
    prop={'size': 12, 'weight': 'bold'}
)
# 🔥 bold tekst w legendzie
for text in legend.get_texts():
    text.set_fontweight('bold')
legend = ax.legend(
    handles=legend_elements,
    prop={'size': 12, 'weight': 'bold'}
)

png_path = Path("d2_vs_r_scatter_hull.png")
fig.savefig(png_path, dpi=200, bbox_inches="tight")

plt.show()