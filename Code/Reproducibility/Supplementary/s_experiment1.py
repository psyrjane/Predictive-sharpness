import numpy as np

# === Entropy & Sharpness ===
def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def sp(p) -> float:
    if isinstance(p, dict):
        p = list(p.values())
    p = np.asarray(p, float).ravel()
    p_sorted = np.sort(p)
    n = p.size
    j = np.arange(1, n + 1, dtype=float)
    w = (2.0 * j - n - 1.0) / (n - 1.0)
    return np.dot(w, p_sorted)

# === Sampling Distributions ===
def sample_distributions(n: int, num_samples: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n), size=num_samples)

# === Analysis Helpers ===
def analyze_level_set(distributions, entropies, sharpness_vals,
                      target_val, tol, mode="sharpness", top_k=10):
    """
    Print top_k lowest and highest values within a level set.
    mode = "sharpness": look for entropy extremes at given sharpness
    mode = "entropy": look for sharpness extremes at given entropy
    """
    if mode == "sharpness":
        mask = np.abs(sharpness_vals - target_val) < tol
        vals, other = entropies[mask], sharpness_vals[mask]
        val_label, other_label = "Entropy", "Sharpness"
    elif mode == "entropy":
        mask = np.abs(entropies - target_val) < tol
        vals, other = sharpness_vals[mask], entropies[mask]
        val_label, other_label = "Sharpness", "Entropy"
    else:
        raise ValueError("mode must be 'sharpness' or 'entropy'")

    if not np.any(mask):
        print(f"\nNo distributions found for {other_label} ≈ {target_val}")
        return

    selected_probs = distributions[mask]
    pairs = sorted(zip(vals, selected_probs), key=lambda x: x[0])

    print(f"\n=== {other_label} ≈ {target_val} ===")

    # lowest values
    print(f"\nTop {top_k} Lowest {val_label}:")
    for v, p in pairs[:top_k]:
        print(f"{val_label}: {v:.4f}, Dist: {np.round(p, 3)}")

    # highest values
    print(f"\nTop {top_k} Highest {val_label}:")
    for v, p in pairs[-top_k:]:
        print(f"{val_label}: {v:.4f}, Dist: {np.round(p, 3)}")


# === Experiment ===
if __name__ == "__main__":
    n = 10
    num_samples = 5_000_000
    seed = 42

    # --- Sample distributions ---
    distributions = sample_distributions(n, num_samples, seed)
    entropies = np.array([shannon_entropy(p) for p in distributions])
    sharpness_vals = np.array([sp(p) for p in distributions])

    # --- Sharpness level set analysis ---
    for target_s in [0.4, 0.6]:
        analyze_level_set(distributions, entropies, sharpness_vals,
                          target_val=target_s, tol=0.01, mode="sharpness", top_k=10)

    # --- Entropy level set analysis ---
    for target_H in [1.0, 2.0]:
        analyze_level_set(distributions, entropies, sharpness_vals,
                          target_val=target_H, tol=0.01, mode="entropy", top_k=10)