import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# Midpoint grid sampler
# ============================================================
def midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=False):
    bounds = np.array(bounds, dtype=float)
    dims = bounds.shape[0]
    if np.isscalar(bins):
        bins = [int(bins)] * dims
    bins = np.array(bins, dtype=int)

    widths = (bounds[:, 1] - bounds[:, 0]) / bins
    coords = [
        bounds[i, 0] + (np.arange(bins[i]) + 0.5) * widths[i]
        for i in range(dims)
    ]

    grids = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=-1)

    if callable(pdf):
        pdfs = [pdf]
        single = True
    elif isinstance(pdf, (list, tuple)):
        pdfs = pdf
        single = False
    else:
        raise TypeError("pdf must be a callable or list of callables")

    results = []
    for f in pdfs:
        try:
            dvals = f(pts) if dims > 1 else f(pts.ravel())
        except Exception:
            dvals = np.zeros(pts.shape[0], dtype=float)

        dvals = np.asarray(dvals, float)

        nan_mask = np.isnan(dvals)
        neg_inf_mask = dvals == -np.inf
        pos_inf_mask = dvals == np.inf

        if np.any(pos_inf_mask):
            finite_mask = np.isfinite(dvals)
            if np.any(finite_mask):
                max_finite = np.max(dvals[finite_mask])
                replacement_value = max(max_finite, 1.0) * 1e6
            else:
                replacement_value = 1e6
            dvals[pos_inf_mask] = replacement_value

        dvals[nan_mask | neg_inf_mask] = 0.0
        dvals = np.clip(dvals, 0, None)

        if normalize:
            cell_volume = np.prod(widths)
            total_mass = np.sum(dvals) * cell_volume
            if total_mass <= 0:
                raise ValueError("PDF has zero total mass over the given bounds.")
            dvals /= total_mass

        results.append(dvals.ravel())

    if single:
        return (results[0], coords) if return_coords else results[0]
    return (results, coords) if return_coords else results

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

# ============================================================
# 3D PDFs on [0,2]^3, so |Omega| = 8
# ============================================================
bounds8 = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
bins = 60
L = 8.0

def local_coords(pts):
    pts = np.atleast_2d(pts)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    q1 = (x < 1.0) & (y < 1.0)
    q2 = (x >= 1.0) & (y < 1.0)
    q3 = (x < 1.0) & (y >= 1.0)
    q4 = (x >= 1.0) & (y >= 1.0)

    u = np.where(x < 1.0, x, x - 1.0)
    v = np.where(y < 1.0, y, y - 1.0)
    w = z / 2.0
    s = (u + v + w) / 3.0
    return s, q1, q2, q3, q4

def exp_shape_mean1(s, alpha):
    if abs(alpha) < 1e-12:
        return np.ones_like(s)
    mean = (3.0 * (np.exp(alpha / 3.0) - 1.0) / alpha) ** 3
    return np.exp(alpha * s) / mean

def pdf_uniform_3d(pts):
    pts = np.atleast_2d(pts)
    return np.full(pts.shape[0], 1.0 / L, dtype=float)

def pdf_smooth_quadrants_2(pts):
    pts = np.atleast_2d(pts)
    s, q1, q2, q3, q4 = local_coords(pts)
    out = np.zeros(pts.shape[0], dtype=float)
    out[q1] = (0.04 / 2.0) * exp_shape_mean1(s[q1], alpha=0.9)
    out[q2] = (0.18 / 2.0) * exp_shape_mean1(s[q2], alpha=3.5)
    out[q3] = (0.28 / 2.0) * exp_shape_mean1(s[q3], alpha=1.1)
    out[q4] = (0.50 / 2.0)
    return out

def pdf_smooth_quadrants_3(pts):
    pts = np.atleast_2d(pts)
    s, q1, q2, q3, q4 = local_coords(pts)
    out = np.zeros(pts.shape[0], dtype=float)
    out[q1] = 0.0
    out[q2] = (0.08 / 2.0)
    out[q3] = (0.22 / 2.0) * exp_shape_mean1(s[q3], alpha=1.1)
    out[q4] = (0.70 / 2.0) * exp_shape_mean1(s[q4], alpha=4.0)
    return out

def _gaussian_box_norm(mu, sigma, a=0.0, b=2.0):
    total = 1.0
    for m, s in zip(mu, sigma):
        z1 = (a - m) / (math.sqrt(2.0) * s)
        z2 = (b - m) / (math.sqrt(2.0) * s)
        total *= s * math.sqrt(math.pi / 2.0) * (math.erf(z2) - math.erf(z1))
    return total

def pdf_spiked_gaussian(pts, mu=(1.65, 1.35, 1.55), sigma=(0.30, 0.30, 0.30)):
    pts = np.atleast_2d(pts)
    mu = np.asarray(mu, dtype=float)[None, :]
    sigma = np.asarray(sigma, dtype=float)[None, :]
    z = np.exp(-0.5 * np.sum(((pts - mu) / sigma) ** 2, axis=1))
    Z = _gaussian_box_norm(mu.ravel(), sigma.ravel(), a=0.0, b=2.0)
    return z / Z


# ============================================================
# 3D PDFS
# ============================================================
pdf_specs = [
    ("Uniform", pdf_uniform_3d),
    ("Slow + fast + fast + uniform", pdf_smooth_quadrants_2),
    ("Excluded + uniform + slow + fast", pdf_smooth_quadrants_3),
    ("Spiked Gaussian", pdf_spiked_gaussian),
]

color_map = {
    "Uniform": "black",
    "Slow + fast + fast + uniform": "red",
    "Excluded + uniform + slow + fast": "blue",
    "Spiked Gaussian": "#6FB06A",
}

# Discretize the 3D PDFs on the midpoint grid
pdf_vals_list = midpoint_discretize(
    [pdf for _, pdf in pdf_specs],
    bounds8,
    bins,
    normalize=True
)

pdfs = list(zip(
    [name for name, _ in pdf_specs],
    pdf_vals_list
))

# ============================================================
# Plot Gini-style curves
# ============================================================
plt.figure(figsize=(10, 6))

for name, pdf_vals in pdfs:
    sh = sharpness_multi(pdf_vals, mode="simplified")
    print(f"{name}: Sharpness S(d_*) = {sh:.6f}")
    sh2, u, cum_mass = sharpness_multi(pdf_vals, mode="gini", plot_data=True)
    plt.plot(u, cum_mass, label=f"{name}", color=color_map.get(name, None))

plt.title("Lorenz Curves for Select Distributions", fontsize=14)
plt.xlabel("Fraction of Domain (u)")
plt.ylabel("Cumulative Probability Mass")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
