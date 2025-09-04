import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Generate grid points on 3-simplex
points_4d = []
step = 0.02
vals = np.arange(0, 1 + step, step)
for p1 in vals:
    for p2 in vals:
        for p3 in vals:
            p4 = 1 - p1 - p2 - p3
            if 0 <= p4 <= 1:
                points_4d.append([p1, p2, p3, p4])
points_4d = np.array(points_4d)

def project_to_3d(p):
    v1 = np.array([1, 0, -1/np.sqrt(2)])
    v2 = np.array([-1, 0, -1/np.sqrt(2)])
    v3 = np.array([0, 1, 1/np.sqrt(2)])
    v4 = np.array([0, -1, 1/np.sqrt(2)])
    return p[0]*v1 + p[1]*v2 + p[2]*v3 + p[3]*v4

def sharpness_n4(p):
    sorted_p = np.sort(p)
    weights = np.array([(2 * j - 4 - 1) / (4 - 1) for j in range(1, 5)])
    return np.dot(weights, sorted_p)

# Compute sharpness and entropy for each point
sharpness_vals_n4 = np.array([sharpness_n4(p) for p in points_4d])
entropy_vals_n4 = np.array([entropy(p, base=np.e) for p in points_4d])

# Define sharpness levels to visualize
tolerance = 0.01
sharpness_levels_n4 = [0.2, 0.6, 0.8, 0.95]
sharpness_families_n4 = {
    level: np.array([p for i, p in enumerate(points_4d) if abs(sharpness_vals_n4[i] - level) <= tolerance])
    for level in sharpness_levels_n4
}

# Define entropy levels to visualize
entropy_levels_n4 = [0.5, 0.9, 1.2, 1.35]
entropy_families_n4 = {
    level: np.array([p for i, p in enumerate(points_4d) if abs(entropy_vals_n4[i] - level) <= tolerance])
    for level in entropy_levels_n4
}

# Project to 3D using tetrahedral projection
sharpness_projections_n4 = {
    level: np.array([project_to_3d(p) for p in sharpness_families_n4[level]])
    for level in sharpness_levels_n4
}

entropy_projections_n4 = {
    level: np.array([project_to_3d(p) for p in entropy_families_n4[level]])
    for level in entropy_levels_n4
}

# Plot level sets for n=4 in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'green', 'orange', 'red']
for i, level in enumerate(sharpness_levels_n4):
    proj = sharpness_projections_n4[level]
    if len(proj) > 0:
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], color=colors[i], s=10, marker='x', alpha=0.6, label=f"Sharpness ≈ {level}")

ax.set_title("Sharpness Sets on the 3-Simplex (n = 4)", fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, level in enumerate(entropy_levels_n4):
    proj = entropy_projections_n4[level]
    if len(proj) > 0:
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], color=colors[i], s=10, marker='x', alpha=0.6, label=f"Entropy ≈ {level}")

ax.set_title("Entropy Sets on the 3-Simplex (n = 4)", fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()