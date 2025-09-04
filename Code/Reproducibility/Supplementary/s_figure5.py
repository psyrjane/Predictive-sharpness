import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 3-simplex grid (n=4)
delta = 0.01
points_hr = []
for p1 in np.arange(0, 1 + delta, delta):
    for p2 in np.arange(0, 1 - p1 + delta, delta):
        for p3 in np.arange(0, 1 - p1 - p2 + delta, delta):
            p4 = 1 - p1 - p2 - p3
            if p4 >= 0:
                points_hr.append((p1, p2, p3, p4))
points_hr = np.array(points_hr)

# Step 2: Compute variance
def variance_fn_n4(p):
    x_vals = np.array([0, 1, 2, 3])
    mu = np.dot(p, x_vals)
    return np.dot(p, (x_vals - mu) ** 2)
variances = np.array([variance_fn_n4(p) for p in points_hr])

# Step 3: Define variance level sets
levels = [0.5, 1.0, 1.5, 2.0]
epsilon = 0.01
level_sets_hr = {
    level: points_hr[np.abs(variances - level) < epsilon]
    for level in levels
}

# Step 4: Define 3D projection from 4D simplex
v1 = np.array([1, 0, -1/np.sqrt(2)])
v2 = np.array([-1, 0, -1/np.sqrt(2)])
v3 = np.array([0, 1, 1/np.sqrt(2)])
v4 = np.array([0, -1, 1/np.sqrt(2)])
def project_to_3d(p):
    return p[:,0:1]*v1 + p[:,1:2]*v2 + p[:,2:3]*v3 + p[:,3:4]*v4

# Step 5: Define custom colors
custom_colors = ['gold', 'darkorange', 'red', 'purple']

# Step 6: Create the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, level in enumerate(levels):
    pts = level_sets_hr[level]
    proj = project_to_3d(pts)
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=4, marker='x', color=custom_colors[i], label=f'Variance ≈ {level}')

ax.set_title("Variance Sets on the 3-Simplex (n=4)", fontsize=14, fontweight='bold')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()