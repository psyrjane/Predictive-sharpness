"""
Predictive Sharpness Core Formulas (discrete + continuous 1D)

This script implements sharpness measures from the paper:
"A Measure of Predictive Sharpness for Probabilistic Models".

Only the core formulas are included here (in simple, computationally efficient form). The formulas can be
further customized for specific applications. For generally applicable continuous sharpness formulas, use
sharpness_multi.py.

WHAT THIS GIVES YOU
-------------------
1) Discrete sharpness:
   - sp(p):     Simplified discrete sharpness S(P)
   - sp_ml(p):  Mass–length version of discrete sharpness (analytically equivalent to sp)

2) Continuous sharpness on a bounded interval [a, b]:
   - sd(pdf, a, b, bins):        Simplified continuous sharpness S(d_*)
   - sd_ml(pdf, a, b, bins):     Mass–length continuous sharpness (equiv. to sd)
   - sd_gini(pdf, a, b, bins):   Gini-style continuous sharpness (equiv. to sd)

3) Relative sharpness, domain transformations:
   - s_rel(S1, S2):              Relative sharpness gain (ΔS_REL)
   - sp_f(Sm, m, n, strict=True) / sp_i(Sn, n, m, strict=True):    Discrete-domain scaling (forward/inverse)
   - sd_f(S_ell, ell, L, strict=True) / sd_i(S_L, L, ell, strict=True):    Continuous-domain scaling (forward/inverse)

INTERPRETATION
--------------
All sharpness scores are normalized to [0, 1]:
- 0   → maximally diffuse (uniform over the domain)
- 1   → maximally sharp (degenerate point prediction / Dirac-like)

ASSUMPTIONS / REQUIREMENTS
--------------------------
- For discrete functions (sp, sp_ml), 'p' is a valid pmf: p_i ≥ 0 and sum(p) = 1.
- For continuous functions (sd, sd_ml, sd_gini):
    * 'pdf(x)' returns density values on [a, b].
    * The density integrates to ~1 over [a, b]. (Small numerical errors are fine.)
    * The domain [a, b] is bounded with a < b.

PRACTICAL TIPS
----
- If your pdf isn't exactly normalized on [a, b], renormalize before sharpness calculation.
- Increase 'bins' for higher-precision numerical integration. Bin size 10_000 is default and achieves good balance for accuracy and efficiency.
"""


import numpy as np

def sp(p):
    """
    Discrete sharpness S(P): simplified (fast) form — works for 1D or multidimensional PMFs.

    Definition (paper, Eq. 2.4):
        Let p_(j) be the probabilities sorted in ascending order, j = 1..n.
        Then
            S(P) = sum_{j=1}^n [ (2j - n - 1) / (n - 1) ] * p_(j)

    Intuition:
        - Measures how much the mass deviates from the uniform and concentrates
          over fewer outcomes.
        - For multidimensional arrays, the PMF is flattened so the calculation
          is done over all outcomes as a single list.

    Args:
        p (array-like or dict): probabilities over a finite outcome space, sum to 1.
           - If a dict is given, its values are extracted as the PMF.
           - For multidimensional arrays, the PMF is flattened so the calculation
             is done over all outcomes as a single list (pass the array of probabilities
             evaluated at each grid point).

    Returns:
        float in [0, 1]:
            0 for uniform, 1 when all mass is on a single outcome.
    """

    if isinstance(p, dict):
        p = list(p.values())
    p = np.asarray(p, float).ravel()
    p_sorted = np.sort(p)
    n = p.size
    j = np.arange(1, n + 1, dtype=float)
    w = (2.0 * j - n - 1.0) / (n - 1.0)
    return np.dot(w, p_sorted)


def sp_ml(p):
    """
    Discrete sharpness S(P): mass–length (expanded) form — works for 1D or multidimensional PMFs.

    Construction (paper, Eq. 2.3):
        - Let p_(j) be the probabilities sorted in ascending order, j = 1..n.
        - At each step j, define:
              m_(j) = sum_{k=j}^n p_(k)        (remaining mass)
              L_(j) = n - j + 1                (remaining outcomes)
          The local deviation from uniform is:
              m_(j) - p_(j) * L_(j)
        - Average these contributions over j = 1..(n-1):
              S(P) = (1/(n-1)) * sum_{j=1}^{n-1} [ m_(j) - p_(j) * L_(j) ]

    Intuition:
        - Adds up normalized local "under-uniform" deficits as mass concentrates
          on fewer outcomes.

    Args:
        p (array-like or dict): probabilities over a finite outcome space, sum to 1.
           - If a dict is given, its values are extracted as the PMF.
           - For multidimensional arrays, the PMF is flattened so the calculation
             is done over all outcomes as a single list (pass the array of probabilities
             evaluated at each grid point).

    Returns:
        float in [0, 1], equivalent to sp(p).
    """
    if isinstance(p, dict):
        p = list(p.values())
    p = np.asarray(p, float).ravel()
    p_sorted = np.sort(p)
    n = p.size
    m = np.cumsum(p_sorted[::-1])[::-1]
    num = m[:-1].sum() - np.dot(p_sorted[:-1], np.arange(n, 1, -1, dtype=float))
    return num / (n - 1.0)


def sd(pdf, a, b, bins=10_000):

    """
    Continuous sharpness S(d_*): simplified (fast) form.

    Definition (paper, Eq. 2.6):
        S(d_*) = (2 / L) * ∫_0^L t * d_*(t) dt - 1,
        where:
            - L = b - a
            - d_*(t) is the non-decreasing rearrangement of the density values.

    Numerical scheme:
        - Discretize [a, b] with 'bins'.
        - Evaluate pdf at midpoints and sort ascending to approximate d_*(t).
        - Approximate ∫ t d_*(t) dt by midpoint Riemann sum.

    Intuition:
        - Computes the mean position of mass in rearranged space: more mass
          pushed to the high-density end ⇒ larger integral ⇒ higher sharpness.

    Args:
        pdf (callable): function mapping array x → density values on [a, b].
        a (float): left bound.
        b (float): right bound (must be > a).
        bins (int): number of bins (resolution).

    Returns:
        float in [0, 1].
    """

    L = b - a
    w = L / bins
    weights = np.arange(bins) + 0.5
    x = a + weights * w
    d_sorted = np.sort(np.asarray(pdf(x), float))
    t = weights * w
    integral = w * np.dot(d_sorted, t) 
    return (2.0 / L) * integral - 1.0

def sd_ml(pdf, a, b, bins=10_000):

    """
    Continuous sharpness S(d_*): mass–length (expanded) form.

    Construction (paper, Eq. 2.5):
        Define in rearranged space:
          m(t) = ∫_t^L d_*(s) ds          (remaining mass)
          L(t) = L - t                    (remaining length)
        Then
          S(d_*) = (1 / L) ∫_0^L [ m(t) - d_*(t) * L(t) ] dt

    Numerical scheme:
        - Discretize [a, b] at midpoints x_i; evaluate and sort pdf(x_i) to get d_*(t_i).
        - Approximate m(t_i) via reverse cumulative sums times bin width.
        - Compute d_*(t_i)*(L - t_i)] for each bin.

    Intuition:
        - Accumulates the local deficits from uniformity across the rearranged domain.

    Args:
        pdf (callable): function mapping array x → density values on [a, b].
        a (float): left bound.
        b (float): right bound.
        bins (int): number of bins (resolution).

    Returns:
        float in [0, 1], equivalent to sd(pdf, a, b).
    """

    L = b - a
    w = L / bins
    idx = np.arange(bins, dtype=float)
    t = idx * w
    x = a + (idx + 0.5) * w
    d_sorted = np.sort(np.asarray(pdf(x), float))
    m = np.cumsum(d_sorted[::-1])[::-1] * w
    dL = d_sorted * (L - t)
    return ((m - dL)[:-1].sum()) / bins

def sd_gini(pdf, a, b, bins=10_000):

    """
    Continuous sharpness S(d_*): Gini-style form.

    Construction (paper, Eq. 4.2):
        S(d_*) = 1 - 2 ∫_0^1 L(u) du,
        where L(u) is a Lorenz-type curve built from the cumulative mass of d_*(t).

    Numerical scheme:
        - Build d_*(t) by sorting pdf values on midpoints.
        - cum_mass[i] = ∑_{k ≤ i} d_*(t_k) * w, from 0 to 1.
        - Approximate ∫_0^1 L(u) du by trapezoidal sum over u = t / L.

    Intuition:
        - Same score as sd/sd_ml, but via the Lorenz-curve perspective:
          more "inequality" of mass across the domain ⇒ higher sharpness.

    Args:
        pdf (callable): function mapping array x → density values on [a, b].
        a (float): left bound.
        b (float): right bound.
        bins (int): number of bins (resolution).

    Returns:
        float in [0, 1], equivalent to sd and sd_ml.
    """

    L = b - a
    w = L / bins
    x = a + (np.arange(bins) + 0.5) * w
    d_sorted = np.sort(np.asarray(pdf(x), float))
    cum_mass = np.concatenate([[0], np.cumsum(d_sorted) * w])
    lorenz = np.sum((cum_mass[:-1] + cum_mass[1:]) / 2) / bins
    return 1.0 - 2.0 * lorenz

def s_rel(S1, S2, strict=True):

    """
    Compute the relative sharpness gain from S1 to S2.

    Formula:
        ΔS_REL = (S2 - S1) / (1 - S1)

    Assumptions / requirements:
        S1 and S2 are the sharpness scores of distributions evaluated over the same domain.

    Parameters:
        S1 (float): Baseline sharpness score (0 ≤ S1 < 1)
        S2 (float): New sharpness score (S2 > S1, S2 ≤ 1)
        strict (bool): If True, enforces valid range checks

    Returns:
                   float in [0, 1].

         Raises:
                   ValueError if inputs violate the valid ranges (when strict=True).
         """

    S1, S2 = float(S1), float(S2)
    if strict and not (0 <= S1 < 1 and S1 < S2 <= 1):
        raise ValueError("Require 0 ≤ S1 < S2 ≤ 1")
    return (S2 - S1) / (1 - S1)


def sp_f(Sm, m, n, strict=True):

    """
    Transform sharpness score from discrete domain of size m to larger domain of size n.

    Formula:
        S_n = 1 + ((m - 1)/(n - 1)) * (S_m - 1)

    Parameters:
        Sm (float): Sharpness on original discrete domain of size m
        m (int): Original domain size (e.g., 4 outcomes)
        n (int): Expanded domain size (n > m)
        strict (bool): If True, enforces valid range checks

    Returns:
        float: Sharpness scaled to the larger domain

    Raises:
                   ValueError if inputs violate the valid ranges (when strict=True).
         """

    Sm = float(Sm)
    if strict and (not (0 <= Sm <= 1) or m >= n):
        raise ValueError("Require 0 ≤ Sm ≤ 1 and m < n")
    return 1 + ((m - 1) / (n - 1)) * (Sm - 1)


def sp_i(Sn, n, m, strict=True):

    """
    Transform sharpness score from discrete domain of size n to smaller domain of size m on the assumption that the n - m additional outcomes are assigned zero probability.

    Formula:
        S_m = 1 + ((n - 1)/(m - 1)) * (S_n - 1)

    Parameters:
        Sn (float): Sharpness on original discrete domain of size n
        n (int): Original domain size (e.g., 7 outcomes)
        m (int): Smaller domain size (m < n)
        strict (bool): If True, enforces valid range checks

    Returns:
        float: Sharpness scaled to the smaller domain

    Raises:
                   ValueError if inputs violate the valid ranges (when strict=True).
         """

    Sn = float(Sn)
    if strict and (not (0 <= Sn <= 1) or n <= m):
        raise ValueError("Require 0 ≤ Sn ≤ 1 and n > m")
    return 1 + ((n - 1) / (m - 1)) * (Sn - 1)


def sd_f(S_ell, ell, L, strict=True):

    """
    Transform sharpness score from a restricted continuous domain of measure ell to extended domain of measure L.

    Formula:
        S_L = 1 + (ell / L) * (S_ell - 1)

    Parameters:
        S_ell (float): Sharpness over original domain of measure ell
        ell (float): Original domain Lebesgue measure (e.g., 4)
        L (float): Target domain Lebesgue measure (L > ell)
        strict (bool): If True, enforces valid range checks

    Returns:
        float: Sharpness score scaled to the larger domain

    Raises:
                   ValueError if inputs violate the valid ranges (when strict=True).
         """

    S_ell = float(S_ell)
    if strict and (not (0 <= S_ell <= 1) or ell >= L):
        raise ValueError("Require 0 ≤ S_ell ≤ 1 and ell < L")
    return 1 + (ell / L) * (S_ell - 1)


def sd_i(S_L, L, ell, strict=True):

    """
    Transform sharpness score from an extended continuous domain of measure L to restricted domain of measure ell, on the assumption that the original probability density function d(y) assigned zero probability to the extended region from ell to L.

    Formula:
        S_ell = 1 + (L / ell) * (S_L - 1)

    Parameters:
        S_L (float): Sharpness over original domain of measure L
        L (float): Original domain Lebesgue measure (e.g., 10)
        ell (float): Target domain Lebesgue measure (ell < L)
        strict (bool): If True, enforces valid range checks

    Returns:
        float: Sharpness score scaled to the restricted domain

    Raises:
                   ValueError if inputs violate the valid ranges (when strict=True).
         """

    S_L = float(S_L)
    if strict and (not (0 <= S_L <= 1) or L <= ell):
        raise ValueError("Require 0 ≤ S_L ≤ 1 and L > ell")
    return 1 + (L / ell) * (S_L - 1)

# === EXAMPLE INPUTS ===

distributions = [
    [0.25, 0.25, 0.25, 0.25],
    [0.5, 0.25, 0.25, 0],
    [0.1, 0.7, 0.1, 0.1],
    [0.6, 0, 0.4, 0],
    [1.0, 0, 0, 0]
]

a, b = 0, 4.0  # domain for pdfs
bins = 10_000  # bins for pdfs

pdfs = [
    ("Uniform",
     lambda x: np.ones_like(x) / (np.sum(np.ones_like(x)) * (b - a) / bins)),

    ("Gaussian μ=2.8, σ=1",
     lambda x: (np.exp(-0.5 * ((x - 2.8) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))) /
               (np.sum(np.exp(-0.5 * ((x - 2.8) / 1.0) ** 2) / (1.0 * np.sqrt(2 * np.pi))) * (b - a) / bins)),

    ("Mixture 0.5·N(1.2,0.3²)+0.5·N(3.0,0.4²)",
     lambda x: (0.5 * np.exp(-0.5 * ((x - 1.2) / 0.3) ** 2) / (0.3 * np.sqrt(2 * np.pi)) +
                0.5 * np.exp(-0.5 * ((x - 3.0) / 0.4) ** 2) / (0.4 * np.sqrt(2 * np.pi))) /
               (np.sum(0.5 * np.exp(-0.5 * ((x - 1.2) / 0.3) ** 2) / (0.3 * np.sqrt(2 * np.pi)) +
                       0.5 * np.exp(-0.5 * ((x - 3.0) / 0.4) ** 2) / (0.4 * np.sqrt(2 * np.pi))) * (b - a) / bins))
]

# === USER INPUT FROM CSV ===
# You can manually define distributions, or load them from external files (e.g., xlsx, csv).
# A discrete example, with each excel/csv row containing one probability vector:
#
#     import pandas as pd
#     df = pd.read_csv(r"you path\your_file.csv", header=None, sep=";")
#     distributions_csv = df.values.tolist()
#
# To enable sp(p) to read files with distributions defined over varying n, add the following line to the definition of sp(p), p = p[~np.isnan(p)], below p = np.asarray(p, float)

# === OUTPUT ===

sharpness_scores = []

for i, p in enumerate(distributions, start=1):
    s1 = sp(p)
    s2 = sp_ml(p)
    sharpness_scores.append(s1)
    print(f"Distribution {i}: {p}")
    print(f"  Sharpness (simplified):   {s1:.3f}")
    print(f"  Sharpness (mass-length): {s2:.3f}")

print("\nContinuous Distributions on [0, 4]:\n")
for label, pdf in pdfs:
    S_simpl = sd(pdf, a, b)
    S_ml_val = sd_ml(pdf, a, b)
    S_gini_val = sd_gini(pdf, a, b)
    print(f"{label:55s}  S(sd)={S_simpl:.4f} | S(sd_ml)={S_ml_val:.4f} | S(sd_gini)={S_gini_val:.4f}")

# === OTHER FORMULAS ===

print("\n=== Discrete Relative Gain & Domain Transformation Example ===")
m, n = 4, 7
S2 = sp(distributions[1])  # Dist 2
S4 = sp(distributions[3])  # Dist 4
rel_gain_discrete = s_rel(S2, S4)
print(f"ΔS_REL (Dist 2 → Dist 4): {rel_gain_discrete:.4f}")

S_val = S2
S_forward = sp_f(S_val, m, n)
S_inverse = sp_i(S_forward, n, m)
print(f"\nDomain transformation for Dist 2: {distributions[1]}")
print(f"  S(P) = {S_val:.4f}")
print(f"  sp_f (m={m} → n={n}): {S_forward:.4f}")
print(f"  sp_i (n={n} → m={m}): {S_inverse:.4f}")

print("\n=== Continuous Relative Gain & Domain Transformation Example ===")
ell, L = 4.0, 7.0
S_gauss = sd(pdfs[1][1], a, b)   # Gaussian μ=2.8, σ=1
S_mix   = sd(pdfs[2][1], a, b)   # Mixture
rel_gain_cont = s_rel(S_gauss, S_mix)
print(f"ΔS_REL (Gaussian → Mixture): {rel_gain_cont:.4f}")

S_val = S_gauss
S_forward = sd_f(S_val, ell, L)
S_inverse = sd_i(S_forward, L, ell)
print(f"\nDomain transformation for Gaussian μ=2.8, σ=1")
print(f"  S(d*) = {S_val:.4f}")
print(f"  sd_f (ell={ell} → L={L}): {S_forward:.4f}")
print(f"  sd_i (L={L} → ell={ell}): {S_inverse:.4f}")

# From csv
# print("\n=== Sharpness Calculations (sp) ===")
# for i, p in enumerate(distributions_csv, start=1):
#     clean_p = [val for val in p if not (isinstance(val, float) and np.isnan(val))]
#     sharp = sp(clean_p)
#     print(f"Distribution {i}: {clean_p}")
#     print(f"  Sharpness (simplified): {sharp:.4f}")