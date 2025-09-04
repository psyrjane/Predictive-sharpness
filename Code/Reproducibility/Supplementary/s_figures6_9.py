import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Barycentric to cartesian
def barycentric_to_cartesian(p):
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3)/2])
    return p[0]*a + p[1]*b + p[2]*c

# Generate grid points on 2-simplex
step = 0.01
vals = np.arange(0, 1 + step, step)
points_n3 = []
for i in vals:
    for j in vals:
        k = 1 - i - j
        if 0 <= k <= 1:
            points_n3.append([i, j, k])
points_n3 = np.array(points_n3)

tolerance = 0.01

# Sharpness for n=3
def sharpness_n3(p):
    sorted_p = np.sort(p)
    weights = np.array([(2 * j - 3 - 1) / (3 - 1) for j in range(1, 4)])
    return np.dot(weights, sorted_p)

sharpness_vals = np.array([sharpness_n3(p) for p in points_n3])

# Variance for n=3
x_vals_n3 = np.array([0, 1, 2])
variance_vals = np.array([
    np.dot(p, (x_vals_n3 - np.dot(p, x_vals_n3))**2) for p in points_n3
])

entropy_vals = np.array([entropy(p, base=np.e) for p in points_n3])

# Filter distributions with sharpness ≈ 0.4
sharpness_target_specific = 0.4
sharpness_set = np.array([
    p for i, p in enumerate(points_n3)
    if abs(sharpness_vals[i] - sharpness_target_specific) <= tolerance
])

# Compute entropy and variance for sharpness set
entropy_set = np.array([entropy(p, base=np.e) for p in sharpness_set])
variance_set = np.array([
    np.dot(p, (x_vals_n3 - np.dot(p, x_vals_n3))**2) for p in sharpness_set
])

# Project to 2D
proj_sharpness_set = np.array([barycentric_to_cartesian(p) for p in sharpness_set])

# Plot entropy heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(
    proj_sharpness_set[:, 0],
    proj_sharpness_set[:, 1],
    c=entropy_set,
    cmap='viridis',
    s=25,
    marker='x'
)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Entropy Variation over Sharpness ≈ 0.4", fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax, label="Entropy")
plt.tight_layout()
plt.show()

# Filter for entropy ≈ 0.9
target_entropy = 0.9
filtered_points = np.array([
    p for i, p in enumerate(points_n3)
    if abs(entropy_vals[i] - target_entropy) <= tolerance
])
filtered_sharpness = np.array([
    sharpness_n3(p) for p in filtered_points
])
proj_points = np.array([barycentric_to_cartesian(p) for p in filtered_points])

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(proj_points[:, 0], proj_points[:, 1], c=filtered_sharpness, cmap='cividis', s=25, marker='x')
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Sharpness Variation over Entropy ≈ 0.9", fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax, label="Sharpness")
plt.tight_layout()
plt.show()


# Plot variance heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sc2 = ax.scatter(
    proj_sharpness_set[:, 0],
    proj_sharpness_set[:, 1],
    c=variance_set,
    cmap='plasma',
    s=25,
    marker='x'
)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Variance Variation over Sharpness ≈ 0.4", fontsize=12, fontweight='bold')
plt.colorbar(sc2, ax=ax, label="Variance")
plt.tight_layout()
plt.show()


# Filter for variance ≈ 0.5
target_variance = 0.5
filtered_points2 = np.array([
    p for i, p in enumerate(points_n3)
    if abs(variance_vals[i] - target_variance) <= tolerance
])
filtered_sharpness = np.array([
    sharpness_n3(p) for p in filtered_points2
])
proj_points = np.array([barycentric_to_cartesian(p) for p in filtered_points2])

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(proj_points[:, 0], proj_points[:, 1], c=filtered_sharpness, cmap='cividis', s=25, marker='x')
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("Sharpness Variation over Variance ≈ 0.5", fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax, label="Sharpness")
plt.tight_layout()
plt.show()



print("\n=== Sharpness ≈ 0.4 ===")
sorted_entropy = sorted(zip(sharpness_set, entropy_set), key=lambda x: x[1])
print("\nTop 10 Lowest Entropy:")
for p, v in sorted_entropy[:10]:
    print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")
print("\nTop 10 Highest Entropy:")
for p, v in sorted_entropy[-10:]:
    print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")

sorted_variance = sorted(zip(sharpness_set, variance_set), key=lambda x: x[1])
print("\nTop 10 Lowest Variance:")
for p, v in sorted_variance[:10]:
    print(f"Variance: {v:.4f}, Dist: {np.round(p, 4)}")
print("\nTop 10 Highest Variance:")
for p, v in sorted_variance[-10:]:
    print(f"Variance: {v:.4f}, Dist: {np.round(p, 4)}")

# Print top/bottom 10 for entropy ≈ 0.9
print("\n=== Entropy ≈ 0.9 ===")
filtered_sharpness = np.array([sharpness_n3(p) for p in filtered_points])
filtered_variance = np.array([
    np.dot(p, (x_vals_n3 - np.dot(p, x_vals_n3))**2) for p in filtered_points
])

sorted_sharp = sorted(zip(filtered_points, filtered_sharpness), key=lambda x: x[1])
print("\nTop 10 Lowest Sharpness:")
for p, v in sorted_sharp[:10]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")
print("\nTop 10 Highest Sharpness:")
for p, v in sorted_sharp[-10:]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")

# Print top/bottom 10 for variance ≈ 0.5
print("\n=== Variance ≈ 0.5 ===")
filtered_entropy = np.array([entropy(p, base=np.e) for p in filtered_points2])
filtered_sharpness = np.array([sharpness_n3(p) for p in filtered_points2])

sorted_sharp = sorted(zip(filtered_points2, filtered_sharpness), key=lambda x: x[1])
print("\nTop 10 Lowest Sharpness:")
for p, v in sorted_sharp[:10]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")
print("\nTop 10 Highest Sharpness:")
for p, v in sorted_sharp[-10:]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")