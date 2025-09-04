import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define sharpness_n3 function
def sharpness_n3(p):
    sorted_p = np.sort(p)
    weights = np.array([(2 * j - 3 - 1) / (3 - 1) for j in range(1, 4)])
    return np.dot(weights, sorted_p)

# Define barycentric projection for 3-simplex to 2D
def barycentric_to_cartesian(p):
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([0.5, np.sqrt(3)/2])
    return p[0]*A + p[1]*B + p[2]*C

# Generate grid of points in the 2-simplex (n=3)
step = 0.01
vals = np.arange(0, 1 + step, step)
points_n3 = []
for i in vals:
    for j in vals:
        k = 1 - i - j
        if 0 <= k <= 1:
            points_n3.append([i, j, k])
points_n3 = np.array(points_n3)

# Compute sharpness for all points
sharpness_vals = np.array([sharpness_n3(p) for p in points_n3])

# Select sharpness level
sharpness_target = 0.5
tolerance = 0.005
sharpness_family = np.array([p for i, p in enumerate(points_n3) if abs(sharpness_vals[i] - sharpness_target) <= tolerance])
sharpness_proj = np.array([barycentric_to_cartesian(p) for p in sharpness_family])

# Identify permutation regions
def sort_pattern(p):
    return tuple(np.argsort(p))

patterns = np.array([sort_pattern(p) for p in points_n3])

# Map each unique permutation to a color
unique_patterns = np.unique(patterns, axis=0)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_patterns)))
pattern_to_color = {tuple(p): colors[i] for i, p in enumerate(unique_patterns)}
pattern_colors = np.array([pattern_to_color[tuple(p)] for p in patterns])

# Project all points for base coloring
proj_all = np.array([barycentric_to_cartesian(p) for p in points_n3])

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Draw simplex boundary
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1)

# Fill permutation regions
ax.scatter(proj_all[:, 0], proj_all[:, 1], color=pattern_colors, s=4, alpha=0.3)

# Overlay sharpness level set
ax.scatter(sharpness_proj[:, 0], sharpness_proj[:, 1], color='black', s=15, marker='x', label=f"Sharpness ≈ {sharpness_target}")

# Annotate corners
ax.text(-0.05, -0.05, "p1", fontsize=12)
ax.text(1.05, -0.05, "p2", fontsize=12)
ax.text(0.48, np.sqrt(3)/2 + 0.05, "p3", fontsize=12)

ax.set_title("Sharpness Set ≈ 0.5 and Permutation Regions (n = 3)", fontsize=12, pad=30, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()