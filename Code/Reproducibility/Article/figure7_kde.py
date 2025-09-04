import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# === 2. Multidimensional sharpness calculator ===
def sharpness_multi(dvals, mode="simplified", plot_data=False):

    dvals = np.asarray(dvals, float).ravel()
    N = dvals.size
    L = 1.0 / dvals.mean()
    v = L / N
    d_sorted = np.sort(dvals)

    if mode == "simplified":
        weights = np.arange(N, dtype=float) + 0.5
        t = weights * v
        integral = v * np.dot(d_sorted, t)
        score = (2.0 / L) * integral - 1.0
        if plot_data:
            return score, t, t * d_sorted
        return score

    elif mode == "ml":
        idx = np.arange(N, dtype=float)
        t = idx * v
        m = np.cumsum(d_sorted[::-1])[::-1] * v
        dL = d_sorted * (L - t)
        score = (m[:-1] - dL[:-1]).sum() / N
        if plot_data:
            return score, t, m, dL
        return score

    elif mode == "gini":
        cum_mass = np.concatenate([[0], np.cumsum(d_sorted) * v])
        lorenz_area = np.sum((cum_mass[:-1] + cum_mass[1:]) / 2) * (1 / N)
        score = 1.0 - 2.0 * lorenz_area
        if plot_data:
            u = np.linspace(0, 1, N+1)
            return score, u, cum_mass
        return score

    else:
        raise ValueError("mode must be 'simplified', 'ml', or 'gini'")

# Parameters
grid_size = (6, 6)
ensemble_members = 30
domain_min, domain_max = 0, 10
evaluation_points = np.linspace(domain_min, domain_max, 10000)

# Seed for reproducibility
np.random.seed(42)

# Generate ensemble forecasts
grid_forecasts = []
for i in range(grid_size[0]):
    row = []
    for j in range(grid_size[1]):
        mean = np.random.choice([0.2, 0.5, 1.0, 2.0, 3.0], p=[0.4, 0.3, 0.15, 0.1, 0.05])
        std = np.random.uniform(0.1, 1.0)
        samples = np.random.normal(loc=mean, scale=std, size=ensemble_members)
        samples = np.clip(samples, domain_min, domain_max)
        row.append(samples)
    grid_forecasts.append(row)

# Compute sharpness with KDE and reflection near boundary
sharpness_scores = np.zeros(grid_size)
display_labels = np.empty(grid_size, dtype=object)

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        samples = grid_forecasts[i][j]

        # Reflect near-boundary if concentration at left or right
        reflect_left = np.count_nonzero(samples <= domain_min + 0.1) > 3
        reflect_right = np.count_nonzero(samples >= domain_max - 0.1) > 3
        mirrored_samples = samples.copy()
        if reflect_left:
            mirrored_samples = np.concatenate([mirrored_samples, -samples[samples <= domain_min + 0.1]])
        if reflect_right:
            mirrored_samples = np.concatenate([mirrored_samples, 2 * domain_max - samples[samples >= domain_max - 0.1]])

        # KDE
        kde = gaussian_kde(mirrored_samples, bw_method='scott')
        pdf = kde(evaluation_points)
        pdf /= np.trapezoid(pdf, evaluation_points)

        # Compute S(d_*)
        score = sharpness_multi(pdf, mode="simplified")
        sharpness_scores[i, j] = score

        # 90% CI
        lower, upper = np.percentile(samples, [5, 95])
        display_labels[i, j] = f"[{lower:.1f}, {upper:.1f}]"

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(sharpness_scores, cmap="viridis", origin="upper", 
                vmin=np.min(sharpness_scores), vmax=1.0)
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label("$S(d_*)$", fontsize=14)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        ax.text(j, i, display_labels[i, j], va='center', ha='center', color='white', fontsize=10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()
plt.show()