import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

# === DOMAIN SETUP ===
a, b = 0.0, 4.0
bins = 10_000
L = b - a
w = L / bins
x = (np.arange(bins) + 0.5) * w

# === PDF DEFINITIONS ===
def normalize_pdf(pdf_vals):
    return pdf_vals / (np.sum(pdf_vals) * w)

pdf_uniform = np.ones_like(x) / L

pdf_gauss_1 = norm.pdf(x, loc=2.8, scale=1.0)
pdf_gauss_1 = normalize_pdf(pdf_gauss_1)

pdf_mix_unequal = (
    0.5 * norm.pdf(x, loc=1.2, scale=0.3) +
    0.5 * norm.pdf(x, loc=3.0, scale=0.4)
)
pdf_mix_unequal = normalize_pdf(pdf_mix_unequal)

pdf_gauss_01 = norm.pdf(x, loc=2.8, scale=0.1)
pdf_gauss_01 = normalize_pdf(pdf_gauss_01)

pdfs = [
    ("Uniform", pdf_uniform),
    ("Gaussian σ=1", pdf_gauss_1),
    ("Mixture (1.2,0.3)+(3.0,0.4)", pdf_mix_unequal),
    ("Gaussian σ=0.1", pdf_gauss_01)
]

color_map = {
    "Uniform": "#ffaf00",
    "Gaussian σ=1": "#f46920",
    "Mixture (1.2,0.3)+(3.0,0.4)": "red",
    "Gaussian σ=0.1": "#F75DC3"
}

# === PLOT ===
plt.figure(figsize=(10, 6))

for name, pdf_vals in pdfs:
    sh = sharpness_multi(pdf_vals, mode="simplified")
    print(f"{name}: Sharpness S(d_*) = {sh:.6f}")
    sh2, u, cum_mass = sharpness_multi(pdf_vals, mode="gini", plot_data=True)
    plt.plot(u, cum_mass, label=f"{name}", color=color_map.get(name, None))

plt.title("Gini-style Curves for Select Distributions", fontsize=14)
plt.xlabel("Fraction of Domain (u)")
plt.ylabel("Cumulative Probability Mass")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
