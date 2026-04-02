import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            u = np.linspace(0, 1, N + 1)
            return score, u, cum_mass
        return score
    else:
        raise ValueError("mode must be 'simplified', 'ml', or 'gini'")

def single_gaussian_case(alpha, scale=0.30):
    side = math.sqrt(alpha)
    sx = scale * side
    sy = scale * side
    return [(1.0, 0.5, 0.5, sx, sy, 0.0)]

def two_gaussian_case(alpha, w, s, scale=0.30):
    h = alpha / (2 * w)
    cx1 = 0.5 - s / 2
    cx2 = 0.5 + s / 2
    sx = scale * w
    sy = scale * h
    return [
        (0.5, cx1, 0.5, sx, sy, 0.0),
        (0.5, cx2, 0.5, sx, sy, 0.0),
    ]

def s_for_ds(alpha, w, ds_target):
    h = alpha / (2 * w)
    val = 4.0 * (ds_target ** 4 * 12.0 / (h * h) - w * w / 12.0)
    return math.sqrt(val)

def rot(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def cov_from_params(sx, sy, theta=0.0):
    R = rot(theta)
    D = np.diag([sx**2, sy**2])
    return R @ D @ R.T

def gaussian_pdf_grid(X, Y, mean, cov):
    mu = np.array(mean, dtype=float)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    dx = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    expo = np.einsum("...i,ij,...j->...", dx, inv, dx)
    return np.exp(-0.5 * expo) / (2 * np.pi * np.sqrt(det))

def mixture_pdf_grid(components, bins=240):
    xs = (np.arange(bins) + 0.5) / bins
    ys = (np.arange(bins) + 0.5) / bins
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pdf = np.zeros_like(X)
    for w, mx, my, sx, sy, theta in components:
        pdf += w * gaussian_pdf_grid(X, Y, (mx, my), cov_from_params(sx, sy, theta))
    cell = (1.0 / bins) ** 2
    pdf = np.maximum(pdf, 0.0)
    pdf /= pdf.sum() * cell
    return X, Y, pdf

def determinant_sharpness_from_grid(pdf):
    bins = pdf.shape[0]
    cell = (1.0 / bins) ** 2
    xs = (np.arange(bins) + 0.5) / bins
    ys = (np.arange(bins) + 0.5) / bins
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    w = pdf * cell
    mx = np.sum(w * X)
    my = np.sum(w * Y)
    dx = X - mx
    dy = Y - my
    sxx = np.sum(w * dx * dx)
    syy = np.sum(w * dy * dy)
    sxy = np.sum(w * dx * dy)
    Sigma = np.array([[sxx, sxy], [sxy, syy]])
    return float(np.linalg.det(Sigma) ** 0.25)

def energy_dispersion_from_grid(pdf, n=1500, seed=0):
    bins = pdf.shape[0]
    cell = (1.0 / bins) ** 2
    xs = (np.arange(bins) + 0.5) / bins
    ys = (np.arange(bins) + 0.5) / bins
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    probs = (pdf * cell).ravel()
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(probs), size=n, replace=True, p=probs)
    samp = pts[idx]
    D = np.sqrt(((samp[:, None, :] - samp[None, :, :]) ** 2).sum(axis=2))
    return float(0.5 * D.mean())

ds_row1 = 0.15746798441989338

scale = 0.30
gaussian_components = {
    "S1": single_gaussian_case(0.25, scale=scale),
    "S2": two_gaussian_case(0.25, 0.235, s_for_ds(0.25, 0.235, ds_row1), scale=scale),
    "S3": two_gaussian_case(0.25, 0.445, 0.555, scale=scale),
    "E1": two_gaussian_case(0.20, 0.2655178275862069, 0.5646361655130392, scale=scale),
    "E2": two_gaussian_case(0.16, 0.4131036551724138, 0.5847107055794103, scale=scale),
    "E3": two_gaussian_case(0.16, 0.10896644827586208, 0.4131891696374098, scale=scale),
    "D1": two_gaussian_case(0.28, 0.19, 0.4148664267121577, scale=scale),
    "D2": two_gaussian_case(0.16, 0.15294751175757484, 0.5980344854237843, scale=scale),
    "D3": two_gaussian_case(0.45, 0.4104655781781457, 0.52590672191585, scale=scale),
}

row_order = ["Fixed sharpness", "Fixed energy dispersion", "Fixed det. sharpness"]
col_order = ["Col 1", "Col 2", "Col 3"]
case_rows = {
    "S1": "Fixed sharpness", "S2": "Fixed sharpness", "S3": "Fixed sharpness",
    "E1": "Fixed energy dispersion", "E2": "Fixed energy dispersion", "E3": "Fixed energy dispersion",
    "D1": "Fixed det. sharpness", "D2": "Fixed det. sharpness", "D3": "Fixed det. sharpness",
}
case_cols = {
    "S1": "Col 1", "S2": "Col 2", "S3": "Col 3",
    "E1": "Col 1", "E2": "Col 2", "E3": "Col 3",
    "D1": "Col 1", "D2": "Col 2", "D3": "Col 3",
}

bins = 240
plot_matrix = [[None] * 3 for _ in range(3)]
rows = []

for case in ["S1", "S2", "S3", "E1", "E2", "E3", "D1", "D2", "D3"]:
    comps = gaussian_components[case]
    X, Y, pdf = mixture_pdf_grid(comps, bins=bins)
    S = float(sharpness_multi(pdf.ravel(), mode="simplified"))
    ED = energy_dispersion_from_grid(pdf, n=1500, seed=0)
    DS = determinant_sharpness_from_grid(pdf)
    rows.append({
        "Case": case,
        "Sharpness S": S,
        "Energy dispersion 0.5 E||X-X'||": ED,
        "Determinant sharpness DS": DS,
    })
    i = row_order.index(case_rows[case])
    j = col_order.index(case_cols[case])
    plot_matrix[i][j] = (case, X, Y, pdf, S, ED, DS)

df = pd.DataFrame(rows)
print(df.round(6).to_string(index=False))

fig, axes = plt.subplots(
    3, 3,
    figsize=(10.2, 10.1),
    gridspec_kw={"wspace": 0.04, "hspace": 0.40}
)

for i in range(3):
    for j in range(3):
        case, X, Y, pdf, S, ED, DS = plot_matrix[i][j]
        ax = axes[i, j]
        levels = np.linspace(float(pdf.min()), float(pdf.max()), 24)
        ax.imshow(
            pdf,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="viridis",
            interpolation="bilinear",
            aspect="equal"
        )
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"{case}\nS={S:.3f}, ED={ED:.3f}, DS={DS:.3f}",
            fontsize=9,
            pad=5
        )

        if j == 0:
            ax.set_ylabel(row_order[i], fontsize=10, labelpad=18)

fig.suptitle("Gaussian densities on a 2D domain", fontsize=13, y=0.99)
fig.subplots_adjust(left=0.02, right=0.98, top=0.89, bottom=0.08)
plt.show()
