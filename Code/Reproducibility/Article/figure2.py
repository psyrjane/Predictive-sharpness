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
x = (np.arange(bins) + 0.5) * w  # midpoints

def normalize_pdf(pdf_vals):
    return pdf_vals / (np.sum(pdf_vals) * w)

# === PDF DEFINITIONS ===
pdfs = [
    np.ones_like(x) / L,
    normalize_pdf(0.6 * norm.pdf(x, loc=1.2, scale=0.3) +
                  0.4 * norm.pdf(x, loc=3.0, scale=0.4)),
    normalize_pdf(norm.pdf(x, loc=2.8, scale=0.5)),
    normalize_pdf(norm.pdf(x, loc=2.8, scale=0.01))
]

titles = [
    r"$f(y) = 1 / |\Omega|$",
    r"$0.6 \cdot \mathcal{N}(1.2, 0.3^2) + 0.4 \cdot \mathcal{N}(3.0, 0.4^2)$",
    r"$\mathcal{N}(2.8, 0.5^2)$",
    r"$\mathcal{N}(2.8, 0.01^2)$"
]

# === PLOTTING ===
plt.figure(figsize=(14, 16))

grid_style = dict(linewidth=0.7, alpha=0.5)

for name, pdf_vals in zip(titles, pdfs):
    sh = sharpness_multi(pdf_vals, mode="simplified")
    print(f"{name}: Sharpness S(d_*) = {sh:.6f}")

for i, (pdf, title) in enumerate(zip(pdfs, titles), 1):
    score, t_vals, m, dL = sharpness_multi(pdf, mode="ml", plot_data=True)

    # Left: PDF
    plt.subplot(4, 2, 2*i - 1)
    plt.plot(x, pdf, color="orange", linewidth=2)
    plt.title(title, fontsize=12)
    plt.xlabel("y")
    plt.ylabel("f(y)")
    plt.grid(True, **grid_style)

    # Right: Integrand components
    plt.subplot(4, 2, 2*i)
    plt.plot(t_vals, m, label="m(t)", color="blue", linewidth=2)
    plt.plot(t_vals, dL, label=r"$d_*(t)\cdot L(t)$", color="red", linewidth=2)
    if i == 1:
        plt.title(f"Integral Components (S = {score:.3f})", fontsize=12)
    else:
        plt.title(f"Integral Components (S ≈ {score:.3f})", fontsize=12)
    plt.xlabel("t")
    plt.ylabel("Integrand Value")
    plt.legend()
    plt.grid(True, **grid_style)

plt.tight_layout()
plt.show()