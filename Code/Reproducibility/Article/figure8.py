# Continuous 3D simulation case study:
# - 4 forecasters issue monthly joint predictive densities on [-3,3]^3
# - realized outcome drawn once at year end from a "true" macro density
# - metrics: continuous sharpness, differential entropy, energy score

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

# ============================================================
# Midpoint discretization + continuous sharpness & entropy
# ============================================================

def midpoint_discretize_short(pdf, bounds, bins, normalize=True):
    bounds = np.asarray(bounds, float)
    dims = bounds.shape[0]
    if np.isscalar(bins):
        bins = [int(bins)] * dims
    bins = np.asarray(bins, int)

    widths = (bounds[:, 1] - bounds[:, 0]) / bins
    coords = [bounds[i, 0] + (np.arange(bins[i]) + 0.5) * widths[i] for i in range(dims)]
    grids = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=-1)

    dvals = np.asarray(pdf(pts), float).ravel()
    dvals = np.nan_to_num(dvals, nan=0.0, posinf=0.0, neginf=0.0)
    dvals = np.clip(dvals, 0.0, None)

    cell_volume = np.prod(widths)
    if normalize:
        total_mass = np.sum(dvals) * cell_volume
        if total_mass <= 0:
            raise ValueError("Density has zero total mass over the given bounds.")
        dvals = dvals / total_mass

    return dvals, coords, cell_volume

def sharpness_multi_short(dvals):
    """
    Simplified formula only.
    """
    dvals = np.asarray(dvals, float).ravel()
    N = dvals.size
    L = 1.0 / dvals.mean()   # total domain volume
    v = L / N                # volume per cell
    d_sorted = np.sort(dvals)

    weights = np.arange(N, dtype=float) + 0.5
    t = weights * v
    integral = v * np.dot(d_sorted, t)
    score = (2.0 / L) * integral - 1.0
    return float(score)

def differential_entropy_bits(dvals, cell_volume):
    dvals = np.asarray(dvals, float).ravel()
    dvals = dvals[dvals > 0]
    return float(-np.sum(dvals * np.log2(dvals)) * cell_volume)

# ============================================================
# Distribution helpers
# ============================================================

bounds = np.array([[-3.0, 3.0],   # "GDP growth"
                   [-3.0, 3.0],   # "Inflation"
                   [-3.0, 3.0]])  # "Unemployment change"
grid_n = 31
mc_samples = 3000
months = np.arange(1, 13)

def cov_from_sd_corr(sd, corr):
    sd = np.asarray(sd, float)
    corr = np.asarray(corr, float)
    return np.outer(sd, sd) * corr

def stabilize_cov(cov, min_eig=1e-8):
    cov = np.asarray(cov, float)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, min_eig)
    return vecs @ np.diag(vals) @ vecs.T

def mvn_pdf(x, mean, cov):
    x = np.atleast_2d(x).astype(float)
    mean = np.asarray(mean, float)
    cov = stabilize_cov(cov)
    d = mean.size
    inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    diffs = x - mean
    q = np.einsum("ni,ij,nj->n", diffs, inv, diffs)
    log_norm = -0.5 * (d * np.log(2.0 * np.pi) + logdet)
    return np.exp(log_norm - 0.5 * q)

def mixture_pdf(x, weights, means, covs):
    x = np.atleast_2d(x)
    dens = np.zeros(x.shape[0], dtype=float)
    for w, m, c in zip(weights, means, covs):
        dens += float(w) * mvn_pdf(x, m, c)
    return dens

def sample_mixture_truncated(weights, means, covs, n, bounds, rng):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    d = len(means[0])
    out = []
    while len(out) < n:
        k = max(600, 4 * (n - len(out)))
        idx = rng.choice(len(weights), size=k, p=weights)
        draws = np.zeros((k, d), dtype=float)
        for j in range(len(weights)):
            mask = idx == j
            if np.any(mask):
                draws[mask] = rng.multivariate_normal(means[j], covs[j], size=mask.sum())
        ok = np.all((draws >= bounds[:, 0]) & (draws <= bounds[:, 1]), axis=1)
        out.extend(draws[ok].tolist())
    return np.asarray(out[:n])

def sample_from_density_grid(dvals, bounds, bins, n, rng):
    bounds = np.asarray(bounds, float)
    dims = bounds.shape[0]
    if np.isscalar(bins):
        bins = [int(bins)] * dims
    bins = np.asarray(bins, int)

    widths = (bounds[:, 1] - bounds[:, 0]) / bins
    cell_volume = np.prod(widths)

    masses = np.asarray(dvals, float).ravel() * cell_volume
    masses = masses / masses.sum()

    idx = rng.choice(len(masses), size=n, p=masses)
    multi = np.array(np.unravel_index(idx, bins)).T

    pts = np.zeros((n, dims), dtype=float)
    for j in range(dims):
        lo = bounds[j, 0] + multi[:, j] * widths[j]
        hi = lo + widths[j]
        pts[:, j] = rng.uniform(lo, hi)

    return pts

def energy_score(samples, y):
    samples = np.asarray(samples, float)
    y = np.asarray(y, float)
    term1 = np.mean(np.linalg.norm(samples - y[None, :], axis=1))
    term2 = 0.5 * np.mean(np.linalg.norm(samples[::2] - samples[1::2], axis=1))
    return float(term1 - term2)

def clip_mean(m):
    m = np.asarray(m, float)
    return np.clip(m, bounds[:, 0] + 0.15, bounds[:, 1] - 0.15)

# ============================================================
# "True" outcome macro density
# Variables:
#   x1 = GDP growth
#   x2 = Inflation
#   x3 = Unemployment change
#
# Regime 1: moderate growth
# Regime 2: recession risk
# ============================================================

corr_soft = np.array([
    [ 1.00,  0.25, -0.75],
    [ 0.25,  1.00,  0.10],
    [-0.75,  0.10,  1.00]
])
corr_stag = np.array([
    [ 1.00, -0.35, -0.55],
    [-0.35,  1.00,  0.45],
    [-0.55,  0.45,  1.00]
])

true_weights = np.array([0.72, 0.28])
true_means = [
    np.array([ 1.20,  1.00, -0.75]),
    np.array([-1.45,  2.30,  1.15]),
]
true_covs = [
    cov_from_sd_corr([0.55, 0.45, 0.40], corr_soft),
    cov_from_sd_corr([0.70, 0.55, 0.50], corr_stag),
]

# Draw one outcome realization from the true density
y_real = sample_mixture_truncated(true_weights, true_means, true_covs, 1, bounds, rng)[0]
true_mean = true_weights[0] * true_means[0] + true_weights[1] * true_means[1]

# Monthly public information signals moving toward the realized outcome
signal_cov_base = cov_from_sd_corr(
    [0.95, 0.80, 0.75],
    np.array([
        [ 1.00,  0.15, -0.50],
        [ 0.15,  1.00,  0.05],
        [-0.50,  0.05,  1.00]
    ])
)
signal_scales = np.linspace(1.30, 0.18, 12)
signals = []
for s in signal_scales:
    sig = y_real + rng.multivariate_normal(np.zeros(3), (s ** 2) * signal_cov_base)
    signals.append(np.clip(sig, bounds[:, 0] + 0.1, bounds[:, 1] - 0.1))
signals = np.asarray(signals)

# ============================================================
# Forecasters
# A: disciplined, steadily sharpens
# B: central spike sharpens with broader tails/background
# C: regime-switcher, initially wrong regime then pivots
# D: narrowing core region with tail risk
# ============================================================

def forecast_A(t, signal):
    mean = clip_mean((1 - t) * np.array([0.35, 1.15, 0.05]) + t * (0.90 * signal + 0.10 * true_mean))
    sd = np.array([1.08 - 0.72 * t, 0.88 - 0.56 * t, 0.84 - 0.52 * t])
    corr = np.array([
        [ 1.00,  0.20, -0.70],
        [ 0.20,  1.00,  0.05],
        [-0.70,  0.05,  1.00]
    ])
    return {
        "kind": "gaussian_mixture",
        "weights": np.array([1.0]),
        "means": [mean],
        "covs": [cov_from_sd_corr(sd, corr)],
    }

def forecast_B(t, signal):
    center = clip_mean(0.86 * signal + 0.14 * true_mean)

    spike_sd = np.maximum(np.array([0.70 - 0.62 * t,
                                    0.58 - 0.50 * t,
                                    0.56 - 0.46 * t]),
                          np.array([0.10, 0.10, 0.10]))
    corr_spike = np.array([
        [ 1.00,  0.20, -0.72],
        [ 0.20,  1.00,  0.08],
        [-0.72,  0.08,  1.00]
    ])

    halo_sd = np.array([0.85 + 0.55 * t,
                        0.72 + 0.48 * t,
                        0.68 + 0.46 * t])
    corr_halo = np.array([
        [ 1.00,  0.10, -0.30],
        [ 0.10,  1.00,  0.05],
        [-0.30,  0.05,  1.00]
    ])

    stag_tail = np.array([-1.95,  2.55,  1.45])
    boom_tail = np.array([ 2.25,  2.20, -1.25])

    cov_stag = cov_from_sd_corr([0.72, 0.58, 0.50], corr_stag)
    cov_boom = cov_from_sd_corr(
        [0.68, 0.55, 0.45],
        np.array([
            [ 1.00,  0.35, -0.60],
            [ 0.35,  1.00, -0.15],
            [-0.60, -0.15,  1.00]
        ])
    )

    w_spike = 0.66 - 0.04 * t
    w_halo  = 0.16 + 0.11 * t
    w_stag  = 0.10 + 0.03 * t
    w_boom  = 0.08 + 0.02 * t
    weights = np.array([w_spike, w_halo, w_stag, w_boom], float)
    weights /= weights.sum()

    return {
        "kind": "gaussian_mixture",
        "weights": weights,
        "means": [center, center, stag_tail, boom_tail],
        "covs": [
            cov_from_sd_corr(spike_sd, corr_spike),
            cov_from_sd_corr(halo_sd, corr_halo),
            cov_stag,
            cov_boom,
        ],
    }

def forecast_C(t, signal):
    wrong_mean = clip_mean(np.array([
        signal[0] - (1.35 - 0.55 * t),
        signal[1] + (1.10 - 0.45 * t),
        signal[2] + (0.95 - 0.35 * t),
    ]))
    right_mean = clip_mean(0.88 * signal + 0.12 * true_mean)

    w_wrong = 0.75 * (1 - t) + 0.12
    w_right = 1.0 - w_wrong

    cov_wrong = cov_from_sd_corr(
        [0.85 - 0.30 * t, 0.70 - 0.20 * t, 0.62 - 0.15 * t],
        np.array([
            [ 1.00, -0.25, -0.45],
            [-0.25,  1.00,  0.35],
            [-0.45,  0.35,  1.00]
        ])
    )
    cov_right = cov_from_sd_corr(
        [0.70 - 0.35 * t, 0.58 - 0.28 * t, 0.50 - 0.22 * t],
        np.array([
            [ 1.00,  0.18, -0.65],
            [ 0.18,  1.00,  0.05],
            [-0.65,  0.05,  1.00]
        ])
    )
    return {
        "kind": "gaussian_mixture",
        "weights": np.array([w_wrong, w_right]),
        "means": [wrong_mean, right_mean],
        "covs": [cov_wrong, cov_right],
    }

def forecast_D(t, signal):
    center = clip_mean(0.84 * signal + 0.16 * true_mean)

    core_sd = np.array([
        0.98 - 0.32 * t,
        0.85 - 0.26 * t,
        0.81 - 0.22 * t,
    ])

    tail_sd = np.array([
        1.08 + 0.72 * t,
        0.94 + 0.62 * t,
        0.90 + 0.56 * t,
    ])

    corr = np.array([
        [ 1.00,  0.10, -0.35],
        [ 0.10,  1.00,  0.05],
        [-0.35,  0.05,  1.00]
    ])

    w_tail = 0.08 + 0.05 * t
    w_core = 1.0 - w_tail

    return {
        "kind": "gaussian_mixture",
        "weights": np.array([w_core, w_tail]),
        "means": [center, center],
        "covs": [
            cov_from_sd_corr(core_sd, corr),
            cov_from_sd_corr(tail_sd, corr),
        ],
    }

forecasters = {
    "Forecaster A": forecast_A,
    "Forecaster B": forecast_B,
    "Forecaster C": forecast_C,
    "Forecaster D": forecast_D,
}

# ============================================================
# Evaluate metrics
# ============================================================

metrics = {
    name: {"sharpness": [], "entropy": [], "energy": []}
    for name in forecasters
}

for m_idx, month in enumerate(months):
    t = m_idx / (len(months) - 1)
    signal = signals[m_idx]

    for name, make_forecast in forecasters.items():
        spec = make_forecast(t, signal)

        if spec["kind"] == "gaussian_mixture":
            weights, means, covs = spec["weights"], spec["means"], spec["covs"]
            pdf = lambda pts, w=weights, m=means, c=covs: mixture_pdf(pts, w, m, c)
            samples = sample_mixture_truncated(weights, means, covs, mc_samples, bounds, rng)

        dvals, coords, cell_volume = midpoint_discretize_short(pdf, bounds, [grid_n, grid_n, grid_n], normalize=True)

        S = sharpness_multi_short(dvals)
        H = differential_entropy_bits(dvals, cell_volume)
        ES = energy_score(samples, y_real)

        metrics[name]["sharpness"].append(S)
        metrics[name]["entropy"].append(H)
        metrics[name]["energy"].append(ES)

# ============================================================
# Plot
# ============================================================

palette = {
    "Forecaster A": "#6FB06A",  # green
    "Forecaster B": "#C9A227",  # mustard
    "Forecaster C": "#B05AA9",  # magenta
    "Forecaster D": "#63C7C7",  # cyan
}

linestyles = {
    "Forecaster A": "-",
    "Forecaster B": "-",
    "Forecaster C": "-",
    "Forecaster D": "-",
}

markers = {
    "Forecaster A": "o",
    "Forecaster B": "s",
    "Forecaster C": "^",
    "Forecaster D": "D",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
plt.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.14)

for name in forecasters:
    color = palette[name]
    ls = linestyles[name]
    mk = markers[name]

    axes[0].plot(
        months, metrics[name]["sharpness"],
        color=color, linestyle=ls, marker=mk,
        linewidth=2, markersize=5.5, markerfacecolor=color,
        markeredgecolor="black", markeredgewidth=0.6,
        label=name
    )
    axes[1].plot(
        months, metrics[name]["entropy"],
        color=color, linestyle=ls, marker=mk,
        linewidth=2, markersize=5.5, markerfacecolor=color,
        markeredgecolor="black", markeredgewidth=0.6,
        label=name
    )
    axes[2].plot(
        months, metrics[name]["energy"],
        color=color, linestyle=ls, marker=mk,
        linewidth=2, markersize=5.5, markerfacecolor=color,
        markeredgecolor="black", markeredgewidth=0.6,
        label=name
    )

axes[0].set_title("Sharpness")
axes[0].set_xlabel("Issue month")
axes[0].set_ylabel("Score")
axes[0].grid(True, alpha=0.25)

axes[1].set_title("Entropy")
axes[1].set_xlabel("Issue month")
axes[1].set_ylabel("Bits")
axes[1].grid(True, alpha=0.25)

axes[2].set_title("Energy score")
axes[2].set_xlabel("Issue month")
axes[2].set_ylabel("Score")
axes[2].grid(True, alpha=0.25)

axes[0].legend(loc="best", frameon=True)

fig.suptitle(
    "Simulated monthly 3D macro forecasts over one year\n"
    f"Realized outcome = GDP {y_real[0]:+.2f}, Inflation {y_real[1]:+.2f}, Unemployment change {y_real[2]:+.2f}",
    y=0.96
)

plt.show()
