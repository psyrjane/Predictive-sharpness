import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === 1a. Midpoint grid sampler ===
def midpoint_discretize(pdf, bounds, bins, normalize=False, return_pts=False):

    bounds = np.array(bounds, dtype=float)
    dims = bounds.shape[0]
    if np.isscalar(bins):
        bins = [int(bins)] * dims
    bins = np.array(bins, dtype=int)

    # Step sizes for each dimension
    widths = (bounds[:, 1] - bounds[:, 0]) / bins

    # Midpoint coordinates for each axis
    coords = [
        bounds[i, 0] + (np.arange(bins[i]) + 0.5) * widths[i]
        for i in range(dims)
    ]

    # Cartesian product of midpoints
    grids = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=-1)

    # Evaluate PDF
    try:
        dvals = pdf(pts) if dims > 1 else pdf(pts.ravel())
    except Exception:
        dvals = np.zeros(pts.shape[0], dtype=float) # return zeros if failure

    dvals = np.asarray(dvals, float)

    # Handle NaN → 0, -inf → 0, +inf → large finite number
    nan_mask = np.isnan(dvals)
    neg_inf_mask = dvals == -np.inf
    pos_inf_mask = dvals == np.inf

    if np.any(pos_inf_mask):
        finite_mask = np.isfinite(dvals)
        if np.any(finite_mask):
            max_finite = np.max(dvals[finite_mask])
            replacement_value = max_finite * 1e6
        else:
            replacement_value = 1e6
        dvals[pos_inf_mask] = replacement_value

    dvals[nan_mask | neg_inf_mask] = 0.0

    # Clip negatives
    dvals = np.clip(dvals, 0, None)

    if normalize:
        # --- Normalization ---
        cell_volume = np.prod(widths)
        total_mass = np.sum(dvals) * cell_volume
        if total_mass <= 0:
            raise ValueError("PDF has zero total mass over the given bounds.")
        dvals /= total_mass

    if return_pts:
        return dvals.ravel(), pts
    return dvals.ravel()

# === 2. Multidimensional sharpness calculator ===
def sharpness_multi(dvals, mode="simplified", plot_data=False):

    dvals = np.asarray(dvals, float).ravel()
    N = dvals.size
    L = 1.0 / dvals.mean()  # infer total domain volume
    v = L / N               # volume per cell
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

# === Domain & PDF ===
bounds = [(0.0, 5.0)]
bins = 10_000
pdf_norm, pts = midpoint_discretize(lambda y: norm.pdf(y, 3.4, 0.8), bounds, bins, normalize=True, return_pts=True)
x = pts if pts.ndim == 1 else pts[:, 0]

# === Observations & RL ===
y_obs = [2.0, 3.5]
colors = ['darkblue', 'red']
d_obs = np.interp(y_obs, x, pdf_norm)
RLs = d_obs / np.max(pdf_norm)

# === Sharpness ===
S_d_star = sharpness_multi(pdf_norm, mode="simplified")

# === Plot ===
plt.figure(figsize=(8, 4))
plt.plot(x, pdf_norm, color="orange", linewidth=2,
         label=r"$d(y) = \frac{1}{Z} \cdot \varphi(y; \mu=3.4, \sigma=0.8)$")
for y, d, RL, c in zip(y_obs, d_obs, RLs, colors):
    plt.plot(y, d, 'o', color=c,
             label=fr"$y_{{\mathrm{{obs}}}} = {y}$ (RL ≈ {RL:.3f})")
plt.xlabel(r"$y$")
plt.ylabel(r"$d(y)$")
plt.title(f"Sharpness Score ≈ {S_d_star:.3f}", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()