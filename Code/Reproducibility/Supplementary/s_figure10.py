import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sharpness_n4(p):
    sorted_p = np.sort(p)
    weights = np.array([(2 * j - 4 - 1) / (4 - 1) for j in range(1, 5)])
    return np.dot(weights, sorted_p)

def project_to_3d(p):
    v1 = np.array([1, 0, -1/np.sqrt(2)])
    v2 = np.array([-1, 0, -1/np.sqrt(2)])
    v3 = np.array([0, 1, 1/np.sqrt(2)])
    v4 = np.array([0, -1, 1/np.sqrt(2)])
    return p[0]*v1 + p[1]*v2 + p[2]*v3 + p[3]*v4

# Generate 4D points
step = 0.02
vals = np.arange(0, 1 + step, step)
points_4d = []
for p1 in vals:
    for p2 in vals:
        for p3 in vals:
            p4 = 1 - p1 - p2 - p3
            if 0 <= p4 <= 1:
                points_4d.append([p1, p2, p3, p4])
points_4d = np.array(points_4d)

# Compute sharpness
sharpness_vals_n4 = np.array([sharpness_n4(p) for p in points_4d])
x_vals_n4 = np.array([0, 1, 2, 3])

# Tolerance & target
tolerance = 0.02
target_S_n4_mid = 0.7
sharpness_set_n4_mid = np.array([
    p for i, p in enumerate(points_4d)
    if abs(sharpness_vals_n4[i] - target_S_n4_mid) <= tolerance
])

# Compute variance
variance_set_n4_mid = np.array([
    np.dot(p, (x_vals_n4 - np.dot(p, x_vals_n4))**2) for p in sharpness_set_n4_mid
])

# Project to 3D
proj_sharpness_set_n4_mid = np.array([project_to_3d(p) for p in sharpness_set_n4_mid])

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    proj_sharpness_set_n4_mid[:, 0],
    proj_sharpness_set_n4_mid[:, 1],
    proj_sharpness_set_n4_mid[:, 2],
    c=variance_set_n4_mid,
    cmap='plasma',
    s=15,
    marker='x',
    alpha=0.7
)

ax.set_title("Variance over Sharpness ≈ 0.7", fontsize=12, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
fig.colorbar(sc, ax=ax, label="Variance")
plt.tight_layout()
plt.show()


order = np.argsort(variance_set_n4_mid)

print(f"\n=== Sharpness level set: S(P) ≈ {target_S_n4_mid} (tolerance = {tolerance}) ===")

print("\nTop 10 Lowest Variance in level set:")
for idx in order[:10]:
    v = variance_set_n4_mid[idx]
    p = sharpness_set_n4_mid[idx]
    print(f"Variance: {v:.4f}, Dist: {np.round(p, 4)}")

print("\nTop 10 Highest Variance in level set:")
for idx in order[-10:]:
    v = variance_set_n4_mid[idx]
    p = sharpness_set_n4_mid[idx]
    print(f"Variance: {v:.4f}, Dist: {np.round(p, 4)}")