import numpy as np
import matplotlib.pyplot as plt

# Parameters
delta_entropy = 0.02
epsilon_entropy = 0.02
target_entropy = 1.05
delta_sharpness = 0.01
tolerance_sharpness = 0.005
sharpness_targets = [0.79, 0.70]

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
grid_entropy = np.arange(0, 1 + delta_entropy, delta_entropy)
points_entropy = []
for p1 in grid_entropy:
    for p2 in grid_entropy:
        for p3 in grid_entropy:
            if p1 + p2 + p3 <= 1:
                p4 = 1 - p1 - p2 - p3
                points_entropy.append([p1, p2, p3, p4])
points_entropy = np.array(points_entropy)

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
             fontsize=12, fontweight='bold')
ax.set_xlabel("X", fontsize=13)
ax.set_ylabel("Y", fontsize=13)
ax.set_zlabel("Z", fontsize=13)
fig.colorbar(sc, ax=ax, shrink=0.6).set_label("Sharpness", fontsize=13)
plt.tight_layout()
plt.show()


# Sharpness grid
grid_sharp = np.arange(0, 1 + delta_sharpness, delta_sharpness)

# Generate probability distributions
probs_list = []
for p1 in grid_sharp:
    for p2 in grid_sharp:
        for p3 in grid_sharp:
            p4 = 1 - p1 - p2 - p3
            if 0 <= p4 <= 1:
                probs_list.append([p1, p2, p3, p4])
probs_list = np.array(probs_list)

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
                 fontsize=12, fontweight='bold')
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
for target_sharpness in sharpness_targets:
    sharpness_vals = np.array([np.dot(weights, np.sort(p)) for p in probs_list])
    mask = np.abs(sharpness_vals - target_sharpness) < tolerance_sharpness
    selected_probs = probs_list[mask]
    selected_entropy = np.array([shannon_entropy(p) for p in selected_probs])

    sorted_entropy = sorted(zip(selected_probs, selected_entropy), key=lambda x: x[1])

    print(f"\n=== Sharpness ≈ {target_sharpness:.2f} ===")
    print("\nTop 10 Lowest Entropy:")
    for p, v in sorted_entropy[:10]:
        print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")

    print("\nTop 10 Highest Entropy:")
    for p, v in sorted_entropy[-10:]:
        print(f"Entropy: {v:.4f}, Dist: {np.round(p, 4)}")


# === Sharpness within specific entropy level sets ===
entropy_targets = [0.56, 0.57]
entropy_tolerance = 0.005

print("\n=== Sharpness within Entropy Level Sets ===")

for h_target in entropy_targets:
    entropies_vals = np.array([shannon_entropy(p) for p in probs_list])
    mask = np.abs(entropies_vals - h_target) < entropy_tolerance
    selected_probs = probs_list[mask]
    selected_sharpness = np.array([np.dot(weights, np.sort(p)) for p in selected_probs])

    sorted_sharp = sorted(zip(selected_probs, selected_sharpness), key=lambda x: x[1])

    print(f"\n--- Entropy ≈ {h_target:.3f} ---")
    
    print("\nTop 10 Lowest Sharpness:")
    for p, s in sorted_sharp[:10]:
        print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")

    print("\nTop 10 Highest Sharpness:")
    for p, s in sorted_sharp[-10:]:
        print(f"Sharpness: {s:.4f}, Dist: {np.round(p, 4)}")