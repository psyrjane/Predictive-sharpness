import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


# ============================================================
# PDFs on [0,2]^3, so |Omega| = 8
# ============================================================
bounds8 = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
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
# Build flipped ML-slope vase data
# width = 1 - Δ(t), where Δ(t)=m(t)-q(t)(L-t)
# ============================================================
def flipped_ml_vase_data(pdf, bounds, bins=60):
    dvals = midpoint_discretize(pdf, bounds, bins, normalize=True)
    N = dvals.size
    v = L / N

    q = np.sort(dvals)
    t_left = np.arange(N) * v
    t_mid = t_left + 0.5 * v
    m = np.cumsum(q[::-1])[::-1] * v
    a = q * (L - t_left)
    delta = m - a

    # Equal 25% mass bands in ranked space
    cum_mass_used = np.cumsum(q) * v

    def t_at_mass(p):
        if p <= 0:
            return 0.0
        if p >= 1:
            return L
        i = np.searchsorted(cum_mass_used, p, side="left")
        prev = 0.0 if i == 0 else cum_mass_used[i - 1]
        qi = q[i]
        if qi <= 0:
            return t_left[i]
        return t_left[i] + (p - prev) / qi

    mass_edges = np.linspace(0.0, 1.0, 4)
    t_edges = np.array([t_at_mass(p) for p in mass_edges])

    # Sharpness
    sharpness = (2.0 / L) * (v * np.dot(q, t_mid)) - 1.0

    # Flipped width: intrinsic fractional version
    width = 1.0 - delta

    # Display downsample
    n_plot = min(4500, N)
    idx = np.linspace(0, N - 1, n_plot).astype(int)

    return {
        "t": t_mid[idx],
        "width": width[idx],
        "t_edges": t_edges,
        "sharpness": sharpness,
    }


pdf_specs = [
    ("Slow + fast + fast + uniform", pdf_smooth_quadrants_2),
    ("Excluded + uniform + slow + fast", pdf_smooth_quadrants_3),
    ("Spiked Gaussian", pdf_spiked_gaussian),
]

data_list = []
for title, pdf in pdf_specs:
    d = flipped_ml_vase_data(pdf, bounds8, bins=60)
    d["title"] = title
    data_list.append(d)

mass_bins = 3
cmap = plt.get_cmap("tab10", mass_bins)

legend_handles = [
    Patch(
        facecolor=cmap(i),
        edgecolor="none",
        alpha=0.72,
        label=f"{int(round(100*i/mass_bins))}–{int(round(100*(i+1)/mass_bins))}% mass",
    )
    for i in range(mass_bins)
]

# ============================================================
# Plot concentration profiles
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14.4, 6), sharey=True, constrained_layout=True)

for ax, data in zip(axes, data_list):
    t = data["t"]
    w = data["width"]
    t_edges = data["t_edges"]
    y = t / L
    y_edges = t_edges / L

    ax.plot(w, y, color="black", lw=1.8)
    ax.plot(-w, y, color="black", lw=1.8)

    for i in range(mass_bins):
        mask = (y >= y_edges[i]) & (y <= y_edges[i + 1])
        ax.fill_betweenx(y[mask], -w[mask], w[mask], color=cmap(i), alpha=0.72)

    for ye in y_edges[1:-1]:
        w_ye = np.interp(ye, y, w)
        ax.hlines(ye, xmin=-0.99*w_ye, xmax=0.99*w_ye, color="white", lw=1.3, alpha=0.95)

    ax.set_ylim(0, 1.0)

    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"{data['title']}\nS(f) = {data['sharpness']:.3f}", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.grid(False)
    ax.set_xlabel(r"$1-\Delta(t)$", fontsize=9)

    # Only show ±0.5, labeled as 0.5
    ax.set_xticks([-0.5, 0.5])
    ax.set_xticklabels(["0.5", "0.5"], fontsize=8)

for ax in axes[1:]:
    ax.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].spines["left"].set_visible(True)
axes[0].spines["right"].set_visible(False)
axes[0].yaxis.tick_left()
axes[0].yaxis.set_label_position("left")
axes[0].set_ylabel("fractions of the rearranged domain")

axes[0].legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=8,
    frameon=True,
    title="Mass bins",
    title_fontsize=8,
)

plt.show()
