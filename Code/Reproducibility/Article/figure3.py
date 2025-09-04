import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === 1a. Midpoint grid sampler ===
def midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=False):

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

    # Check list of PDFs or single PDF
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
            dvals = np.zeros(pts.shape[0], dtype=float)  # return zeros if failure

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

        results.append(dvals.ravel())

    if single:
        if return_coords:
            return results[0], coords
        return results[0]
    else:
        if return_coords:
            return results, coords
        return results

# === Define PDF (mixture + gaps) ===
def pdf_func(x):
    vals = (
        0.45 * norm.pdf(x, loc=1.0, scale=0.25) +
        0.18 * norm.pdf(x, loc=3.8, scale=0.3) +
        0.15 * norm.pdf(x, loc=2.4, scale=0.4) +
        0.02 * norm.pdf(x, loc=7.0, scale=0.1)
    )
    vals[(x > 5) & (x < 6.5)] = 0
    vals[(x > 7.5)] = 0
    vals /= np.sum(vals) * ((8 - 0) / len(x))
    return vals

# === Discretize using midpoint rule ===
bins = 10000
bounds = [(0.0, 8.0)]
pdf_vals = midpoint_discretize(pdf_func, bounds, bins)

# === ML-calculations ===
L = 8
v = L / bins
idx = np.arange(bins, dtype=float)
t_vals = idx * v
d_star = np.sort(pdf_vals)
m_t = np.cumsum(d_star[::-1])[::-1] * v
L_t = L - t_vals

# === Find the minimum non-zero value ===
nonzero_index = np.where(pdf_vals > 0)[0][0]
d_min = pdf_vals[nonzero_index]

# === 1000-bin region mapping for [5, 8] ===
x_region_500 = np.linspace(5, 8, 1000)
pdf_region_500 = np.interp(x_region_500,
                           np.linspace(bounds[0][0], bounds[0][1], bins),
                           pdf_vals)
valid_mask = pdf_region_500 > d_min
pdf_region_valid = pdf_region_500[valid_mask]
t_region_filtered = [t_vals[np.argmin(np.abs(d_star - d))] for d in pdf_region_valid]

# === Position of least non-zero value ===
t_val_min = t_vals[np.argmin(np.abs(d_star - d_min))]

# === Plotting ===
fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Original PDF
x_mid = np.linspace(bounds[0][0], bounds[0][1], bins)
axs[0].plot(x_mid, pdf_vals, color="orange", lw=2)
axs[0].set_title("Original pdf")
axs[0].set_xlabel("y")
axs[0].set_ylabel("f(y)")
axs[0].grid(True)

# Mass-length integral components
axs[1].plot(t_vals, m_t, label=r"$m(t)$", color="blue", lw=2)
axs[1].plot(t_vals, d_star * L_t, label=r"$d_*(t) \cdot L(t)$", color="red", lw=2)
for t_val in t_region_filtered:
    m_val = np.interp(t_val, t_vals, m_t)
    dL_val = np.interp(t_val, t_vals, d_star * L_t)
    axs[1].vlines(t_val, ymin=min(m_val, dL_val), ymax=max(m_val, dL_val),
                  color='steelblue', alpha=0.03)
axs[1].axvline(t_val_min, color="black", linestyle="--", lw=2, label=r"Least non-zero $y$")
axs[1].set_title(f"Mass-length integral components")
axs[1].set_xlabel("t")
axs[1].set_ylabel("Integrand Value")
axs[1].grid(True)
axs[1].legend()

plt.show()