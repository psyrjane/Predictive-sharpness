""""
Entropy–Sharpness Extremizers for Discrete and Continuous Cases
---------------------------------------------------------------

This script implements the extremal distribution families discussed in the paper
and the supplementary material for the predictive sharpness measure S(P) and S(f):

    1) Maximum entropy at fixed sharpness
    2) Maximum sharpness at fixed entropy
    3) Minimum entropy at fixed sharpness (discrete)
    4) Minimum sharpness at fixed entropy (discrete)

The script covers both the discrete ranked-PMF case and the continuous rearranged
density case on a bounded domain.

WHAT THIS GIVES YOU
-------------------
1) DISCRETE maximum-entropy ranked PMFs at fixed sharpness:
   - max_entropy_discrete(n, S_target, ...)
     → Returns the Gibbs / exponential ranked PMF
       p_(j) ∝ exp(beta * w_j)
       with beta chosen so that S(P) = S_target

2) DISCRETE maximum-sharpness ranked PMFs at fixed entropy:
   - max_sharpness_discrete(n, H_target, ...)
     → Returns the same Gibbs family
       p_(j) ∝ exp(beta * w_j)
       with beta chosen so that H(P) = H_target

3) CONTINUOUS maximum-entropy rearranged densities at fixed sharpness:
   - max_entropy_continuous(Omega, S_target, ...)
     → Returns the truncated exponential family on [0, Omega]
       f_lambda(t) ∝ exp(lambda * t)
       with lambda chosen so that S(f) = S_target

4) CONTINUOUS maximum-sharpness rearranged densities at fixed entropy:
   - max_sharpness_continuous(Omega, h_target, ...)
     → Returns the same truncated exponential family
       f_lambda(t) ∝ exp(lambda * t)
       with lambda chosen so that h(f) = h_target

5) DISCRETE minimum-entropy ranked PMFs at fixed sharpness:
   - min_entropy_discrete(n, S)
     → Searches the boundary / extreme-point candidate families described in the
       supplementary material, including uniform-on-support and two-level ranked PMFs

6) DISCRETE minimum-sharpness ranked PMFs at fixed entropy:
   - min_sharpness_discrete(n, target_entropy, ...)
     → Searches the two boundary families described in the
       supplementary material and returns the minimizing ranked PMF

INTERPRETATION
--------------
This script does NOT compute sharpness from arbitrary PDFs directly.
Instead, it computes the ranked PMFs or rearranged densities that are extremal
under entropy/sharpness constraints.

In other words:
- the "max" functions return the smooth exponential / Gibbs extremizers
- the discrete "min" functions return the boundary-type extremizers

ASSUMPTIONS / REQUIREMENTS
--------------------------
- Discrete PMFs are treated in ranked form
- Continuous results are for the rearranged density on [0, Omega)
- The target values must be feasible:
    * S_target in [0, 1) for maximum-entropy / continuous maximum cases
    * H_target in (0, log(n)] for discrete max-sharpness
    * h_target <= log(Omega) for continuous max-sharpness

OUTPUTS
-------
Depending on the function, outputs include:
- beta or lambda
- ranked PMF or extremizing rearranged density parameter
- achieved sharpness
- achieved entropy

EXAMPLE
-------
A typical discrete comparison at n = 4:

    max_entropy_discrete(4, 0.7)
    min_entropy_discrete(4, 0.7)

    max_sharpness_discrete(4, 0.9, entropy_unit="nats/bits")
    min_sharpness_discrete(4, 0.9, entropy_unit="nats/bits")

"""

import numpy as np
from scipy.optimize import brentq

def max_entropy_discrete(n, S_target, tol=1e-12, max_iter=200, verbose=True):
    """
    DISCRETE (any n): Given target sharpness S_target in [0,1), return the max-entropy ranked PMF.

    Uses the weights for sharpness:
        w_j = (2j - n - 1)/(n - 1), j=1..n
    Max-entropy at fixed sharpness is exponential:
        p_(j) ∝ exp(beta * w_j)

    Returns:
        beta (float),
        p_ranked (np.ndarray, shape (n,), ascending by rank),
        H_nats (float),
        H_bits (float),
        S_achieved (float)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    S_target = float(S_target)
    if not (0.0 <= S_target < 1.0):
        raise ValueError("S_target must be in [0,1)")

    # weights
    j = np.arange(1, n + 1, dtype=float)
    w = (2.0 * j - n - 1.0) / (n - 1.0)

    # function S(beta) - S_target, with stable softmax
    def f(beta):
        x = beta * w
        m = np.max(x)
        ex = np.exp(x - m)
        p = ex / np.sum(ex)
        S = float(np.dot(w, p))
        return S - S_target

    # bracket beta in [0, hi]
    lo, hi = 0.0, 1.0
    flo = f(lo)  # = -S_target < 0
    fhi = f(hi)
    while fhi < 0.0:
        hi *= 2.0
        fhi = f(hi)
        if hi > 1e6:
            raise ValueError("Could not bracket beta; S_target may be too close to 1.")

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= tol or (hi - lo) <= tol:
            beta = mid
            break
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    else:
        beta = 0.5 * (lo + hi)

    # build solution
    x = beta * w
    m = np.max(x)
    ex = np.exp(x - m)
    p = ex / np.sum(ex)
    H_nats = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    H_bits = H_nats / np.log(2.0)
    S = float(np.dot(w, p))

    if verbose:
        print(f"Beta: {beta:.4f}")
        print(f"PMF (Ranked Probabilities): {p}")
        print(f"Entropy (H_nats): {H_nats:.4f}")
        print(f"Entropy (H_bits): {H_bits:.4f}")
        print(f"Achieved Sharpness (S): {S:.4f}")

    return beta, p, H_nats, H_bits, S

def max_sharpness_discrete(n, H_target, entropy_unit="nats", tol=1e-12, max_iter=200, verbose=True):
    """
    DISCRETE (any n): Given target Shannon entropy H_target in (0, log(n)),
    return the max-sharpness ranked PMF.

    Max-sharpness at fixed entropy uses the same Gibbs family:
        p_(j) ∝ exp(beta * w_j), with beta >= 0 chosen so H(beta) = H_target

    Returns:
        beta (float),
        p_ranked (np.ndarray),
        S_achieved (float),
        H_achieved_nats (float),
        H_achieved_bits (float)
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    unit = entropy_unit.lower().strip()
    if unit in ("nat", "nats", "e", "ln"):
        H_target_nats = float(H_target)
    elif unit in ("bit", "bits", "log2", "base2"):
        H_target_nats = float(H_target) * np.log(2.0)
    else:
        raise ValueError("entropy_unit must be 'nats' or 'bits'")

    H_max_nats = np.log(n)
    H_max_bits = H_max_nats / np.log(2.0)

    if not (0.0 < H_target_nats <= H_max_nats + 1e-12):
        if unit in ("bit", "bits", "log2", "base2"):
            raise ValueError(f"H_target must be in (0, log2(n)] = (0, {H_max_bits:.12f}] bits")
        else:
            raise ValueError(f"H_target must be in (0, log(n)] = (0, {H_max_nats:.12f}] nats")

    # weights
    j = np.arange(1, n + 1, dtype=float)
    w = (2.0 * j - n - 1.0) / (n - 1.0)

    # entropy at beta, in nats
    def H_of_beta(beta):
        x = beta * w
        m = np.max(x)
        ex = np.exp(x - m)
        p = ex / np.sum(ex)
        return float(-np.sum(p[p > 0] * np.log(p[p > 0])))

    # bracket beta >= 0 where H decreases from log n to 0
    def g(beta):
        return H_of_beta(beta) - H_target_nats

    lo, hi = 0.0, 1.0
    glo = g(lo)
    ghi = g(hi)
    while ghi > 0.0:
        hi *= 2.0
        ghi = g(hi)
        if hi > 1e6:
            raise ValueError("Could not bracket beta; H_target may be too close to 0.")

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if abs(gmid) <= tol or (hi - lo) <= tol:
            beta = mid
            break
        if glo * gmid <= 0:
            hi, ghi = mid, gmid
        else:
            lo, glo = mid, gmid
    else:
        beta = 0.5 * (lo + hi)

    # build solution
    x = beta * w
    m = np.max(x)
    ex = np.exp(x - m)
    p = ex / np.sum(ex)

    H_nats = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    H_bits = H_nats / np.log(2.0)
    S = float(np.dot(w, p))

    if verbose:
        print(f"Beta: {beta:.4f}")
        print(f"PMF (Ranked Probabilities): {p}")
        print(f"Achieved Sharpness (S): {S:.4f}")
        print(f"Entropy (H_achieved_nats): {H_nats:.4f}")
        print(f"Entropy (H_achieved_bits): {H_bits:.4f}")

    return beta, p, S, H_nats, H_bits

def max_entropy_continuous(Omega, S_target, tol=1e-10, max_iter=200, verbose=True):
    """
    CONTINUOUS: Domain length/measure Omega (>0). We model the maximizing family as a truncated exponential
    on [0, Omega]: f_lambda(t) ∝ exp(lambda * t)

    We choose lambda >= 0 so that S(lambda)=S_target, then compute
    differential entropy in closed form.

    Returns:
        lam (float),
        S_achieved (float),
        h_nats (float),
        h_bits (float)
    """
    L = float(Omega)
    if L <= 0:
        raise ValueError("Omega must be > 0")
    S_target = float(S_target)
    if not (0.0 <= S_target < 1.0):
        raise ValueError("S_target must be in [0,1)")

    def _ET_logZ(lam):
        lam = float(lam)
        if abs(lam) < 1e-15:
            return 0.5 * L, np.log(L)
        u = lam * L
        em1 = np.expm1(-u)  # exp(-u) - 1  (<= 0)
        # E[T]
        E = L * (u + em1) / (u * (-em1))
        # logZ
        if u < 50.0:
            logZ = np.log(np.expm1(u)) - np.log(lam)
        else:
            logZ = (u + np.log(-em1)) - np.log(lam)

        return E, logZ

    def S_of_lam(lam):
        E, _ = _ET_logZ(lam)
        return 2.0 * E / L - 1.0

    def h_of_lam(lam):
        E, logZ = _ET_logZ(lam)
        return logZ - lam * E

    # function for bisection: S(lam) - target
    def f(lam):
        return S_of_lam(lam) - S_target

    # bracket lam >= 0
    lo, hi = 0.0, 1.0 / L
    flo = f(lo)  # = -S_target <= 0
    fhi = f(hi)
    while fhi < 0.0:
        hi *= 2.0
        fhi = f(hi)
        if hi > 1e12:
            raise ValueError("Could not bracket lambda; S_target may be too close to 1.")

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) <= tol or (hi - lo) <= tol:
            lam = mid
            break
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    else:
        lam = 0.5 * (lo + hi)

    S = S_of_lam(lam)
    h = h_of_lam(lam)
    h_bits = h / np.log(2.0)

    if verbose:
        print(f"Lambda: {lam:.4f}")
        print(f"Achieved Sharpness (S): {S:.4f}")
        print(f"Entropy (h_nats): {h:.4f}")
        print(f"Entropy (h_bits): {h_bits:.4f}")

    return lam, S, h, h_bits

def max_sharpness_continuous(Omega, h_target, entropy_unit="nats", tol=1e-10, max_iter=200, verbose=True):
    """
    CONTINUOUS: Domain measure Omega (>0). We use truncated exponential f_lambda on [0,Omega]:
        f_lambda(t) ∝ exp(lambda * t)

    For maximum sharpness at fixed entropy, take lambda >= 0 and solve
        h(lambda) = h_target
    where h is computed in closed form.

    Returns:
        lam (float),
        h_achieved_nats (float),
        h_achieved_bits (float),
        S_achieved (float)
    """
    L = float(Omega)
    if L <= 0:
        raise ValueError("Omega must be > 0")

    unit = entropy_unit.lower().strip()
    if unit in ("nat", "nats", "e", "ln"):
        h_target_nats = float(h_target)
    elif unit in ("bit", "bits", "log2", "base2"):
        h_target_nats = float(h_target) * np.log(2.0)
    else:
        raise ValueError("entropy_unit must be 'nats' or 'bits'")

    def _ET_logZ(lam):
        lam = float(lam)
        if abs(lam) < 1e-15:
            return 0.5 * L, np.log(L)
        u = lam * L
        em1 = np.expm1(-u)  # exp(-u) - 1  (<= 0)

        # E[T]
        E = L * (u + em1) / (u * (-em1))

        # logZ
        if u < 50.0:
            logZ = np.log(np.expm1(u)) - np.log(lam)
        else:
            logZ = (u + np.log(-em1)) - np.log(lam)

        return E, logZ

    def S_of_lam(lam):
        E, _ = _ET_logZ(lam)
        return 2.0 * E / L - 1.0

    def h_of_lam(lam):
        E, logZ = _ET_logZ(lam)
        return logZ - lam * E

    h_max_nats = np.log(L)
    h_max_bits = h_max_nats / np.log(2.0)

    if h_target_nats > h_max_nats + 1e-10:
        if unit in ("bit", "bits", "log2", "base2"):
            raise ValueError(f"h_target must be <= log2(Omega) = {h_max_bits}")
        else:
            raise ValueError(f"h_target must be <= log(Omega) = {h_max_nats}")

    # function for bisection: h(lam) - target, lam>=0, h decreases with lam
    def g(lam):
        return h_of_lam(lam) - h_target_nats

    lo, hi = 0.0, 1.0 / L
    glo = g(lo)
    ghi = g(hi)
    while ghi > 0.0:
        hi *= 2.0
        ghi = g(hi)
        if hi > 1e12:
            raise ValueError("Could not bracket lambda; h_target may be extremely negative.")

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if abs(gmid) <= tol or (hi - lo) <= tol:
            lam = mid
            break
        if glo * gmid <= 0:
            hi, ghi = mid, gmid
        else:
            lo, glo = mid, gmid
    else:
        lam = 0.5 * (lo + hi)

    h_nats = h_of_lam(lam)
    h_bits = h_nats / np.log(2.0)
    S = S_of_lam(lam)

    if verbose:
        print(f"Lambda: {lam:.4f}")
        print(f"Achieved entropy (h_nats): {h_nats:.4f}")
        print(f"Achieved entropy (h_bits): {h_bits:.4f}")
        print(f"Sharpness (S): {S:.4f}")

    return lam, h_nats, h_bits, S


def min_entropy_discrete(n, S):
    """
    DISCRETE: Minimum entropy at fixed sharpness.
    
    Given:
        - n outcomes
        - target sharpness S in [0, 1]
    
    this function searches for the ranked PMF that achieves the given sharpness
    while minimizing Shannon entropy.
    
    METHOD
    ------
    The function evaluates the boundary / extreme-point candidate families
    described in the supplementary material:
    
    1) Uniform-on-support candidates:
           [0, ..., 0, 1/k, ..., 1/k]
    
    2) Two-level ranked extreme points:
           [0, ..., 0, a, ..., a, b, ..., b], with a <= b
    
    Among all feasible candidates matching the target sharpness, it returns
    the one with the lowest entropy. Ties are broken in favor of smaller support.
    
    RETURNS
    -------
        best_p       : minimizing ranked PMF
        best_H_nats  : minimum entropy in nats
        best_H_bits  : minimum entropy in bits
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not (0.0 <= S <= 1.0):
        raise ValueError("S must be in [0, 1]")

    w = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n - 1.0)
    tol = 1e-12

    best_p = None
    best_H_nats = None
    best_support = None

    # Uniform-on-support candidates
    for zeros in range(n):
        k = n - zeros
        c = 1.0 / k
        p = np.concatenate([np.zeros(zeros), np.full(k, c)])
        sharp = float(np.dot(w, p))
        if abs(sharp - S) <= 1e-10:
            H_nats = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
            support = int(np.count_nonzero(p > 1e-14))
            if (
                best_H_nats is None
                or H_nats < best_H_nats - 1e-12
                or (abs(H_nats - best_H_nats) <= 1e-12 and support < best_support)
            ):
                best_p = p.copy()
                best_H_nats = H_nats
                best_support = support

    # Two-level extreme points: [0,...,0, a,...,a, b,...,b] with a <= b
    for zeros in range(n):
        positive_count = n - zeros
        for r in range(1, positive_count):
            m = positive_count - r
            if m < 1:
                continue

            A = np.sum(w[zeros:zeros + r])
            B = np.sum(w[zeros + r:zeros + r + m])

            M = np.array([[r, m], [A, B]], dtype=float)
            det = np.linalg.det(M)
            if abs(det) <= tol:
                continue

            a, b = np.linalg.solve(M, np.array([1.0, S], dtype=float))

            if a < -tol or b < -tol or a > b + tol:
                continue

            a = max(a, 0.0)
            b = max(b, 0.0)

            p = np.concatenate([
                np.zeros(zeros),
                np.full(r, a),
                np.full(m, b)
            ])
            p = np.sort(p)

            if abs(np.sum(p) - 1.0) > 1e-9:
                continue

            sharp = float(np.dot(w, p))
            if abs(sharp - S) > 1e-9:
                continue

            H_nats = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
            support = int(np.count_nonzero(p > 1e-14))

            if (
                best_H_nats is None
                or H_nats < best_H_nats - 1e-12
                or (abs(H_nats - best_H_nats) <= 1e-12 and support < best_support)
            ):
                best_p = p.copy()
                best_H_nats = H_nats
                best_support = support

    if best_p is None:
        raise ValueError("No feasible pmf found for the given (n, S).")

    best_H_bits = best_H_nats / np.log(2.0)

    print(f"Minimum entropy (bits): {best_H_bits:.12f}")
    print(f"Minimum entropy (nats): {best_H_nats:.12f}")
    print(
        "Minimizing distribution: "
        + np.array2string(np.round(best_p, 12), precision=12, separator=", ")
    )

    return best_p, best_H_nats, best_H_bits


def min_sharpness_discrete(n, target_entropy, entropy_unit="nats"):
    """
    DISCRETE: Minimum sharpness at fixed entropy.

    Given:
        - n outcomes
        - target Shannon entropy (in nats or bits)

    this function searches for the ranked PMF that achieves the given entropy
    while minimizing sharpness.

    METHOD
    ------
    The function searches the full two-level ranked family

        [0, ..., 0, a, ..., a, b, ..., b],   with a <= b,

    where the positive part has:
        - r copies of a
        - m copies of b
        - support size k = r + m

    For each support size k and split (r, m), the parameter a is chosen so that
    the entropy matches the target exactly, with

        r a + m b = 1,   so   b = (1 - r a)/m.

    Among all feasible candidates, the function returns the one with the lowest
    sharpness. Ties are broken in favor of smaller support.

    RETURNS
    -------
        best_p : minimizing ranked PMF
        best_S : minimum sharpness
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    unit = entropy_unit.lower().strip()
    if unit in ("nat", "nats", "e", "ln"):
        H_target_nats = float(target_entropy)
    elif unit in ("bit", "bits", "log2", "base2"):
        H_target_nats = float(target_entropy) * np.log(2.0)
    else:
        raise ValueError("entropy_unit must be 'nats' or 'bits'")

    tol = 1e-12
    max_iter = 200
    H_max = np.log(n)

    if not (0.0 <= H_target_nats <= H_max + tol):
        raise ValueError(f"target_entropy must be in [0, log(n)] = [0, {H_max:.12f}] nats")

    H_target_nats = float(np.clip(H_target_nats, 0.0, H_max))
    w = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n - 1.0)

    # H = 0 case: point mass
    if H_target_nats <= tol:
        p = np.zeros(n)
        p[-1] = 1.0
        print(f"Minimum sharpness: {1.000000000000:.12f}")
        print(
            "Minimizing distribution: "
            + np.array2string(np.round(p, 12), precision=12, separator=", ")
        )
        return p, 1.0

    best_p = None
    best_S = None
    best_support = None

    def entropy_two_level(a, r, m):
        """Entropy of [a repeated r times, b repeated m times], with b from normalization."""
        b = (1.0 - r * a) / m
        H_a = -(r * a * np.log(a) if a > 0 else 0.0)
        H_b = -(m * b * np.log(b) if b > 0 else 0.0)
        return H_a + H_b

    for zeros in range(n):
        k = n - zeros
        if k < 2:
            continue

        # positive part split into r copies of a and m copies of b
        for r in range(1, k):
            m = k - r

            # Feasible interval for a under a <= b:
            #   r a + m b = 1,  a <= b  =>  a <= 1/k
            a_lo = 0.0
            a_hi = 1.0 / k

            # Entropy range on this family:
            # at a = 0      -> support size m, entropy = log(m)
            # at a = 1/k    -> uniform on k outcomes, entropy = log(k)
            h_lo = np.log(m)
            h_hi = np.log(k)

            if not (h_lo - tol <= H_target_nats <= h_hi + tol):
                continue

            if H_target_nats <= h_lo + tol:
                a = 0.0
            elif H_target_nats >= h_hi - tol:
                a = a_hi
            else:
                a = brentq(
                    lambda x: entropy_two_level(x, r, m) - H_target_nats,
                    1e-15,
                    a_hi - 1e-15,
                    xtol=tol,
                    maxiter=max_iter,
                )

            b = (1.0 - r * a) / m

            if a < -tol or b < -tol or a > b + tol:
                continue

            a = max(a, 0.0)
            b = max(b, 0.0)

            p = np.concatenate([
                np.zeros(zeros),
                np.full(r, a),
                np.full(m, b)
            ])
            p = np.sort(p)

            if abs(np.sum(p) - 1.0) > 1e-9:
                continue

            H_nats = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
            if abs(H_nats - H_target_nats) > 1e-9:
                continue

            S_val = float(np.dot(w, p))
            support = int(np.count_nonzero(p > 1e-14))

            if (
                best_S is None
                or S_val < best_S - 1e-12
                or (abs(S_val - best_S) <= 1e-12 and support < best_support)
            ):
                best_p = p.copy()
                best_S = S_val
                best_support = support

    if best_p is None:
        raise ValueError("No feasible pmf found for the given entropy target.")

    print(f"Minimum sharpness: {best_S:.12f}")
    print(
        "Minimizing distribution: "
        + np.array2string(np.round(best_p, 12), precision=12, separator=", ")
    )

    return best_p, best_S
