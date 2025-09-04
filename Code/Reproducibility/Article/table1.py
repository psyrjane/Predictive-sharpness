import numpy as np
import pandas as pd

def compute_sharpness(P):
    """
    Compute S(P), the sharpness score for a discrete distribution.
    Formula:
    S(P) = sum_{j=1}^n ((2j - n - 1) / (n - 1)) * p_{(j)},
    where p_{(j)} are sorted ascending.
    """
    p_sorted = np.sort(np.asarray(P, float))
    n = P.size
    j = np.arange(1, n + 1, dtype=float)
    w = (2.0 * j - n - 1.0) / (n - 1.0)
    return np.dot(w, p_sorted)

def compute_entropy(P):
    """
    Compute Shannon entropy in bits:
    H(P) = -sum p_i * log2(p_i)
    Skip p_i = 0 to avoid log(0).
    """
    return -np.sum([p * np.log2(p) for p in P if p > 0])

def compute_kl_divergence(P):
    """
    Compute KL divergence from uniform distribution in bits:
    KL(P || U) = sum p_i * log2(p_i / u_i)
    where u_i = 1/n.
    """
    n = len(P)
    u = 1 / n
    return np.sum([p * np.log2(p / u) for p in P if p > 0])

def compute_variance(P):
    """
    Compute variance over values [0, 1, ..., n-1]
    """
    x = np.arange(len(P))
    mean = np.sum(P * x)
    return np.sum(P * (x - mean) ** 2)

# Define distributions
distributions = [
    [0.25, 0.25, 0.25, 0.25],
    [0.24, 0.24, 0.28, 0.24],
    [0.0, 1/3, 1/3, 1/3],
    [0.0, 0.25, 0.25, 0.5],
    [0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.3, 0.7],
    [0.16, 0.0, 0.0, 0.84],
    [0.0, 0.0, 0.1, 0.9],
    [0.0, 0.0, 0.01, 0.99],
    [0.0, 0.0, 0.0, 1.0]
]

# Compute and collect metrics
rows = []
for P in distributions:
    P = np.array(P, dtype=np.float64)
    sharpness = compute_sharpness(P)
    entropy_bits = compute_entropy(P)
    kl_bits = compute_kl_divergence(P)
    var = compute_variance(P)
    rows.append({
        "Distribution": P.tolist(),
        "S(P)": round(sharpness, 4),
        "Entropy (bits)": round(entropy_bits, 4),
        "KL Divergence (bits)": round(kl_bits, 4),
        "Variance": round(var, 4)
    })

# Show as DataFrame
df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", None)
print(df.to_string(index=False))

print("\n---\nAdditional Calculations:\n")

# 7-element distribution
dist_7 = np.array([0, 0, 0.2, 0.2, 0.2, 0.2, 0.2])
sharpness_7 = compute_sharpness(dist_7)
entropy_7 = compute_entropy(dist_7)
kl_7 = compute_kl_divergence(dist_7)
var_7 = compute_variance(dist_7)

print("Distribution (n=7): [0, 0, 0.2, 0.2, 0.2, 0.2, 0.2]")
print(f"S(P)               = {round(sharpness_7, 4)}")
print(f"Entropy (bits)     = {round(entropy_7, 4)}")
print(f"KL divergence      = {round(kl_7, 4)}")
print(f"Variance           = {round(var_7, 4)}\n")

# Low sharpness, approx Var = 1
low_sharp = np.array([0.19, 0.32, 0.3, 0.19])
sharp_low = compute_sharpness(low_sharp)
var_low = compute_variance(low_sharp)

print("Low-sharpness distribution: [0.19, 0.32, 0.3, 0.19]")
print(f"S(P)               = {round(sharp_low, 4)}")
print(f"Variance           = {round(var_low, 4)}\n")

# High sharpness, approx Var = 1
high_sharp = np.array([0.12, 0.02, 0.0, 0.86])
sharp_high = compute_sharpness(high_sharp)
var_high = compute_variance(high_sharp)

print("High-sharpness distribution: [0.12, 0.02, 0.0, 0.86]")
print(f"S(P)               = {round(sharp_high, 4)}")
print(f"Variance           = {round(var_high, 4)}")