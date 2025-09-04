# Step 1: Define grid on the 2-simplex with fine delta
delta = 0.01
points_n3 = []
for i in np.arange(0, 1 + delta, delta):
    for j in np.arange(0, 1 - i + delta, delta):
        k = 1 - i - j
        if 0 <= k <= 1:
            points_n3.append((i, j, k))
points_n3 = np.array(points_n3)

# Step 2: Define functions
def barycentric_to_cartesian(p):
    p1, p2, p3 = p
    x = p2 + 0.5 * p3
    y = (np.sqrt(3) / 2) * p3
    return (x, y)

def entropy_fn(p):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.sum([pi * np.log(pi) if pi > 0 else 0 for pi in p])

def variance_fn(p):
    x_vals = np.array([0, 1, 2])
    mu = np.dot(p, x_vals)
    return np.dot(p, (x_vals - mu) ** 2)

def sharpness_fn(p):
    p_sorted = np.sort(p)
    weights = np.array([-1, 0, 1])
    return np.dot(weights, p_sorted)

# Step 3: Compute statistics
entropy_n3 = np.array([entropy_fn(p) for p in points_n3])
variance_n3 = np.array([variance_fn(p) for p in points_n3])
sharpness_n3 = np.array([sharpness_fn(p) for p in points_n3])

# Step 4: Define level sets and project them
epsilon_entropy_variance = 0.01
epsilon_sharpness = 0.005
colors = ['red', 'orange', 'green', 'blue']

# Entropy
entropy_levels = [0.2, 0.5, 0.8, 1.0]
entropy_families = {
    level: np.array([p for i, p in enumerate(points_n3) if abs(entropy_n3[i] - level) <= epsilon_entropy_variance])
    for level in entropy_levels
}
entropy_projections = {
    level: np.array([barycentric_to_cartesian(p) for p in entropy_families[level]])
    for level in entropy_levels
}

# Variance
variance_levels = [0.2, 0.5, 0.8, 1.0]
variance_families = {
    level: np.array([p for i, p in enumerate(points_n3) if abs(variance_n3[i] - level) <= epsilon_entropy_variance])
    for level in variance_levels
}
variance_projections = {
    level: np.array([barycentric_to_cartesian(p) for p in variance_families[level]])
    for level in variance_levels
}

# Sharpness
sharpness_levels = [0.2, 0.4, 0.6, 0.8]
sharpness_families = {
    level: np.array([p for i, p in enumerate(points_n3) if abs(sharpness_n3[i] - level) <= epsilon_sharpness])
    for level in sharpness_levels
}
sharpness_projections = {
    level: np.array([barycentric_to_cartesian(p) for p in sharpness_families[level]])
    for level in sharpness_levels
}

# Step 5: Plotting
def plot_level_sets(projections, levels, title, label_prefix):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    # Draw simplex boundary
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=1)
    for i, level in enumerate(levels):
        proj = projections[level]
        if len(proj) > 0:
            ax.scatter(proj[:, 0], proj[:, 1], s=12, marker='x', color=colors[i], alpha=0.6, label=f"{label_prefix} ≈ {level}")
    ax.text(-0.05, -0.05, "p1", fontsize=12)
    ax.text(1.05, -0.05, "p2", fontsize=12)
    ax.text(0.48, np.sqrt(3)/2 + 0.05, "p3", fontsize=12)
    ax.set_title(title, fontsize=12, pad=30, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Step 6: Generate plots
plot_level_sets(entropy_projections, entropy_levels, "Entropy Sets on the 2-Simplex", "Entropy")
plot_level_sets(variance_projections, variance_levels, "Variance Sets on the 2-Simplex", "Variance")
plot_level_sets(sharpness_projections, sharpness_levels, "Sharpness Sets on the 2-Simplex", "Sharpness")