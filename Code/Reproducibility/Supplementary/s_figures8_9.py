import numpy as np
import matplotlib.pyplot as plt

# Parameters
delta_entropy = 0.02
epsilon_entropy = 0.02
target_entropy = 1.05
delta_sharpness = 0.01
tolerance_sharpness = 0.005
sharpness_targets = [0.78, 0.70]

# Weights for sharpness
weights = np.array([-1, -1/3, 1/3, 1])

# Projection vertices of the tetrahedron
vertices = np.array([
    [1, 0, -1 / np.sqrt(2)],
    [-1, 0, -1 / np.sqrt(2)],
    [0, 1, 1 / np.sqrt(2)],
    [0, -1, 1 / np.sqrt(2)]
])

# Shannon entropy
def shannon_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# === Generate grid for the 3-simplex ===
N_entropy = int(round(1 / delta_entropy))
points_entropy = np.array([
    [i / N_entropy,
     j / N_entropy,
     k / N_entropy,
     (N_entropy - i - j - k) / N_entropy]
    for i in range(N_entropy + 1)
    for j in range(N_entropy + 1 - i)
    for k in range(N_entropy + 1 - i - j)
])

# Compute entropies
entropies = np.array([shannon_entropy(p) for p in points_entropy])

# Filter points where entropy ≈ target
mask = np.abs(entropies - target_entropy) < epsilon_entropy
points_coarse = points_entropy[mask]
entropies_coarse = entropies[mask]

# Project to 3D and compute sharpness
projected_coarse = np.dot(points_coarse, vertices)
sharpness_coarse = np.dot(np.sort(points_coarse, axis=1), weights)

# === Plot 1: Sharpness over Entropy ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(projected_coarse[:, 0], projected_coarse[:, 1], projected_coarse[:, 2],
                c=sharpness_coarse, cmap='plasma', s=15, marker='x', alpha=0.7)
ax.set_title(f"Sharpness over Entropy ≈ {target_entropy}",
             fontsize=16, fontweight='bold')
ax.set_xlabel("X", fontsize=13)
ax.set_ylabel("Y", fontsize=13)
ax.set_zlabel("Z", fontsize=13)
fig.colorbar(sc, ax=ax, shrink=0.6).set_label("Sharpness", fontsize=13)
plt.tight_layout()
plt.show()


# Sharpness grid
grid_sharp = np.arange(0, 1 + delta_sharpness, delta_sharpness)

# Generate probability distributions
N_sharp = int(round(1 / delta_sharpness))
probs_list = np.array([
    [i / N_sharp,
     j / N_sharp,
     k / N_sharp,
     (N_sharp - i - j - k) / N_sharp]
    for i in range(N_sharp + 1)
    for j in range(N_sharp + 1 - i)
    for k in range(N_sharp + 1 - i - j)
])

# === Plots 2 & 3: Entropy over Sharpness ===
for target_sharpness in sharpness_targets:
    sharpness_vals = np.array([np.dot(weights, np.sort(p)) for p in probs_list])
    mask = np.abs(sharpness_vals - target_sharpness) < tolerance_sharpness
    selected_probs = probs_list[mask]
    selected_entropy = np.array([shannon_entropy(p) for p in selected_probs])

    proj_points = np.dot(selected_probs, vertices)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(proj_points[:, 0], proj_points[:, 1], proj_points[:, 2],
                    c=selected_entropy, cmap='viridis', s=5, marker='x')
    ax.set_title(f"Entropy Over Sharpness ≈ {target_sharpness}",
                 fontsize=16, fontweight='bold')
    ax.set_xlabel("X", fontsize=13)
    ax.set_ylabel("Y", fontsize=13)
    ax.set_zlabel("Z", fontsize=13)
    fig.colorbar(sc, ax=ax, shrink=0.6).set_label("Entropy", fontsize=13)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()


# === Min/max sharpness for entropy target ===
print(f"\n=== Entropy ≈ {target_entropy:.2f} ===")
sorted_sharp = sorted(zip(points_coarse, sharpness_coarse), key=lambda x: x[1])

print("\nTop 10 Lowest Sharpness:")
for p, v in sorted_sharp[:10]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")

print("\nTop 10 Highest Sharpness:")
for p, v in sorted_sharp[-10:]:
    print(f"Sharpness: {v:.4f}, Dist: {np.round(p, 4)}")


# === Min/max entropy for sharpness targets ===

# finer grid
delta_sharpness_levels = 0.005 
N_sharpness_levels = int(round(1 / delta_sharpness_levels))

probs_list_fine_sharp = np.array([
    [i / N_sharpness_levels,
     j / N_sharpness_levels,
     k / N_sharpness_levels,
     (N_sharpness_levels - i - j - k) / N_sharpness_levels]
    for i in range(N_sharpness_levels + 1)
    for j in range(N_sharpness_levels + 1 - i)
    for k in range(N_sharpness_levels + 1 - i - j)
], dtype=float)

sharpness_vals_fine = np.array([np.dot(weights, np.sort(p)) for p in probs_list_fine_sharp])
entropy_vals_fine = np.array([shannon_entropy(p) for p in probs_list_fine_sharp])

for target_sharpness in sharpness_targets:
    mask = np.abs(sharpness_vals_fine - target_sharpness) < tolerance_sharpness
    selected_probs = probs_list_fine_sharp[mask]
    selected_entropy = entropy_vals_fine[mask]

    sorted_entropy = sorted(zip(selected_probs, selected_entropy), key=lambda x: x[1])

    print(f"\n=== Sharpness ≈ {target_sharpness:.2f} ===")
    print("\nTop 10 Lowest Entropy:")
    for p, v in sorted_entropy[:10]:
        print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")

    print("\nTop 10 Highest Entropy:")
    for p, v in sorted_entropy[-10:]:
        print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")


# === Sharpness within specific entropy level sets ===
entropy_targets = [0.63, 0.64]
entropy_tolerance = 0.005

# finer grid
delta_entropy_levels = 0.005
N_entropy_levels = int(round(1 / delta_entropy_levels))

probs_list_fine = np.array([
    [i / N_entropy_levels,
     j / N_entropy_levels,
     k / N_entropy_levels,
     (N_entropy_levels - i - j - k) / N_entropy_levels]
    for i in range(N_entropy_levels + 1)
    for j in range(N_entropy_levels + 1 - i)
    for k in range(N_entropy_levels + 1 - i - j)
], dtype=float)

entropies_vals_fine = np.array([shannon_entropy(p) for p in probs_list_fine])
sharpness_vals_fine = np.array([np.dot(weights, np.sort(p)) for p in probs_list_fine])

print("\n=== Sharpness within Entropy Level Sets ===")

for h_target in entropy_targets:
    mask = np.abs(entropies_vals_fine - h_target) < entropy_tolerance
    selected_probs = probs_list_fine[mask]
    selected_sharpness = sharpness_vals_fine[mask]

    sorted_sharp = sorted(zip(selected_probs, selected_sharpness), key=lambda x: x[1])

    print(f"\n--- Entropy ≈ {h_target:.3f} ---")

    print("\nTop 10 Lowest Sharpness:")
    for p, s in sorted_sharp[:10]:
        print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")

    print("\nTop 10 Highest Sharpness:")
    for p, s in sorted_sharp[-10:]:
        print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")
