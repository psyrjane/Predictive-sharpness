import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import norm

# === DOMAIN SETUP ===
a, b = 0.0, 4.0
bins = 10_000
L = b - a
w = L / bins
x = (np.arange(bins) + 0.5) * w

# === FUNCTIONS ===

def normalize_pdf(pdf_vals):
    return pdf_vals / (np.sum(pdf_vals) * w)

def compute_sharpness(pdf_vals):
    d_sorted = np.sort(np.asarray(pdf_vals, float))
    integral = w * np.dot(d_sorted, x)
    return (2.0 / L) * integral - 1.0

def compute_entropy(pdf_vals):
    safe_pdf = np.where(pdf_vals > 0, pdf_vals, 1.0)
    return -np.sum(pdf_vals * np.log(safe_pdf)) * w

def compute_kl_divergence(pdf_vals):
    u = 1 / L
    safe_pdf = np.where(pdf_vals > 0, pdf_vals, 1.0)
    return np.sum(pdf_vals * np.log(safe_pdf / u)) * w

# Define distributions over [0, 4]
distributions = []

# Uniform
pdf_uniform = np.ones_like(x) / L
distributions.append(("Uniform", pdf_uniform))

# Gaussian mu=2.8, sigma=1
pdf_gauss1 = norm.pdf(x, loc=2.8, scale=1)
pdf_gauss1 = normalize_pdf(pdf_gauss1)
distributions.append(("Gaussian μ=2.8, σ=1", pdf_gauss1))

# Mixture (0.5, 0.5)
pdf_mix1 = 0.5 * norm.pdf(x, 1.2, 0.3) + 0.5 * norm.pdf(x, 3.0, 0.4)
pdf_mix1 = normalize_pdf(pdf_mix1)
distributions.append(("Mixture 0.5 1.2/0.3, 0.5 3.0/0.4", pdf_mix1))

# Mixture (0.6, 0.4)
pdf_mix2 = 0.6 * norm.pdf(x, 1.2, 0.3) + 0.4 * norm.pdf(x, 3.0, 0.4)
pdf_mix2 = normalize_pdf(pdf_mix2)
distributions.append(("Mixture 0.6 1.2/0.3, 0.4 3.0/0.4", pdf_mix2))

# Piecewise: 0 on [0,2); 0.5 on [2,4]
pdf_piece1 = np.zeros_like(x)
pdf_piece1[x >= 2] = 0.5
pdf_piece1 = normalize_pdf(pdf_piece1)
distributions.append(("Piecewise: 0 on [0,2), 0.5 on [2,4]", pdf_piece1))

# Gaussian mu=2.8, sigma=0.5
pdf_gauss2 = norm.pdf(x, loc=2.8, scale=0.5)
pdf_gauss2 = normalize_pdf(pdf_gauss2)
distributions.append(("Gaussian μ=2.8, σ=0.5", pdf_gauss2))

# Piecewise: 0 on [0,2); 0.15 on [2,3); 0.85 on [3,4]
pdf_piece2 = np.zeros_like(x)
pdf_piece2[(x >= 2) & (x < 3)] = 0.15
pdf_piece2[(x >= 3)] = 0.85
pdf_piece2 = normalize_pdf(pdf_piece2)
distributions.append(("Piecewise: 0 on [0,2); 0.15 on [2,3); 0.85 on [3,4]", pdf_piece2))

# Gaussian mu=2.8, sigma=0.1
pdf_gauss3 = norm.pdf(x, loc=2.8, scale=0.1)
pdf_gauss3 = normalize_pdf(pdf_gauss3)
distributions.append(("Gaussian μ=2.8, σ=0.1", pdf_gauss3))

# Gaussian mu=2.8, sigma=0.01
pdf_gauss4 = norm.pdf(x, loc=2.8, scale=0.01)
pdf_gauss4 = normalize_pdf(pdf_gauss4)
distributions.append(("Gaussian μ=2.8, σ=0.01", pdf_gauss4))

# Symbolic sharpness calculation for Dirac delta approximation
delta, L_sym = sp.symbols('delta L', positive=True)

# Expression for d_*(t) = 1/δ over [L - δ, L]
# ∫ (t * (1/δ)) dt from t = L - δ to L
t = sp.symbols('t')
integrand = t * (1 / delta)
int_result = sp.integrate(integrand, (t, L_sym - delta, L_sym))

# Plug into sharpness formula:
S_expr = (2 / L_sym) * int_result - 1
S_limit = sp.limit(S_expr, delta, 0).subs(L_sym, L)

# Compute metrics for all
rows = []
for label, pdf in distributions:
    sharpness = compute_sharpness(pdf)
    entropy = compute_entropy(pdf)
    kl = compute_kl_divergence(pdf)
    rows.append({
        "Distribution": label,
        "S(d*)": round(sharpness, 4),
        "Entropy (nats)": round(entropy, 4),
        "KL Divergence (nats)": round(kl, 4)
    })

rows.append({
    "Distribution": "Dirac delta (symbolic)",
    "S(d*)": float(S_limit),
    "Entropy (nats)": "−",
    "KL Divergence (nats)": "-"
})

# Output results
df = pd.DataFrame(rows)
print("\nContinuous Distributions on [0, 4]:\n")
print(df.to_string(index=False))