import numpy as np
import matplotlib.pyplot as plt

# Define values
x_vals = np.array([0, 1, 2, 3])
target_variance = 1.0
epsilon_variance = 0.01

delta = 0.01
grid = np.arange(0, 1 + delta, delta)
points = []

for p1 in grid:
    for p2 in grid:
        for p3 in grid:
            p4 = 1 - p1 - p2 - p3
            if 0 <= p4 <= 1:
                points.append([p1, p2, p3, p4])

points = np.array(points)

# Projection
vertices = np.array([
    [1, 0, -1/np.sqrt(2)],
    [-1, 0, -1/np.sqrt(2)],
    [0, 1, 1/np.sqrt(2)],
    [0, -1, 1/np.sqrt(2)]
])

projected = points @ vertices

# Variance and filter
means = (points * x_vals).sum(axis=1)
variances = ((points * (x_vals - means[:, np.newaxis])**2).sum(axis=1))
mask_var = np.abs(variances - target_variance) < epsilon_variance
points_var = points[mask_var]
projected_var = projected[mask_var]

# Sharpness for these
sharpness = np.dot(np.sort(points_var, axis=1), np.array([-1, -1/3, 1/3, 1]))
min_sharp = sharpness.min()
max_sharp = sharpness.max()

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(projected_var[:, 0], projected_var[:, 1], projected_var[:, 2],
                     s=10, marker='x', alpha=0.7, c=sharpness, cmap='magma',
                     vmin=min_sharp, vmax=max_sharp)
ax.set_title(f"Sharpness over Variance ≈ {target_variance}", fontsize=15, fontweight='bold')
ax.set_xlabel("X", fontsize=13)
ax.set_ylabel("Y", fontsize=13)
ax.set_zlabel("Z", fontsize=13)
cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label("Sharpness", fontsize=13)
plt.tight_layout()
plt.show()


order = np.argsort(sharpness)

print(f"\n=== Variance level set: Var(P) ≈ {target_variance} (tolerance = {epsilon_variance}) ===")

print("\nTop 10 Lowest Sharpness in variance level set:")
for idx in order[:10]:
    s = sharpness[idx]
    p = points_var[idx]
    print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")

print("\nTop 10 Highest Sharpness in variance level set:")
for idx in order[-10:]:
    s = sharpness[idx]
    p = points_var[idx]
    print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")