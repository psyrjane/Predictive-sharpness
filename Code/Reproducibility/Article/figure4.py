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

pdfs = {
    "Uniform": np.ones_like(x) / L,
    "Mixture (unequal)": normalize_pdf(
        0.5 * norm.pdf(x, loc=1.2, scale=0.3) +
        0.5 * norm.pdf(x, loc=3.0, scale=0.4)
    ),
    "Gaussian σ=0.1": normalize_pdf(norm.pdf(x, loc=2.8, scale=0.1))
}

color_map = {
    "Uniform": "orange",
    "Mixture (unequal)": "red",
    "Gaussian σ=0.1": "blue"
}

label_map = {
    "Uniform": r"$f(y) = \frac{1}{|\Omega|}$",
    "Mixture (unequal)": r"$f(y) = \frac{1}{Z} [0.5\, \varphi(y; 1.2, 0.09) + 0.5\, \varphi(y; 3.0, 0.16)]$",
    "Gaussian σ=0.1": r"$f(y) = \frac{1}{Z}\, \varphi(y; 2.8, 0.1)$"
}

# === PLOT ===
plt.figure(figsize=(8, 5))

for name, pdf_vals in pdfs.items():
    score, t, t_times_d = sharpness_multi(pdf_vals, mode="simplified", plot_data=True)
    print(f"{name}: Sharpness S(d_*) = {score:.6f}")
    eq_symbol = "=" if name == "Uniform" else r"\approx"
    
    plt.plot(
        t, t_times_d,
        label=f"{label_map[name]},  $S {eq_symbol} {score:.3f}$",
        color=color_map[name]
    )

plt.xlabel("t")
plt.ylabel(r"$t \cdot d_*(t)$")
plt.title("Integrands for select pdfs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()