"""
Predictive Sharpness for Multidimensional Cases — Midpoint Discretization + Visualizations
--------------------------------------------------------------------------

This script implements the sharpness measure S(d*) from the paper:
"A Measure of Predictive Sharpness for Probabilistic Models" with support for multidimensional use, midpoint discretization, direct array inputs,
and visualizations, using the three equivalent sharpness formulas:

    1) Simplified
    2) Mass–Length (ML)
    3) Gini-style

WHAT THIS GIVES YOU
-------------------
1) Midpoint-based discretization of callable PDFs:
   - midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=False)
     → Returns flattened array of density values at midpoint grid.

2) Preparation of existing array-based PDFs for sharpness calculation:
   - prepare_array_pdf(array_pdf, coords, bounds=None, normalize=True, return_coords=False)
     → Cleans and pads/crops an existing PDF array to match the target bounds.
       Assumes the array is from a **uniform** grid in physical coordinates, starting at the lower
       bound of 'bounds'. Optional normalization to integrate to 1.

3) Sharpness calculation for discretized PDF values:
   - sharpness_multi(dvals, mode="simplified" | "ml" | "gini")
     → Returns S(d*) in [0, 1]

4) Visualizations of the three formulations:
   - visualize_sharpness(pdfs, titles, mode="simplified" | "ml" | "gini")
     → Simplified: plots t·d*(t)
     → ML: plots m(t) and d*(t)·L(t) separately
     → Gini: plots Lorenz-style cumulative mass curves

INTERPRETATION
--------------
All sharpness scores are normalized to [0, 1]:
- 0   → maximally diffuse (uniform distribution over the domain)
- 1   → maximally sharp (degenerate / Dirac-like prediction)

ASSUMPTIONS / REQUIREMENTS
--------------------------
- Input density arrays for sharpness calculation must:
    * Be non-negative and finite
    * Be from a uniform grid
    * Integrate to ~1 over the domain (small numerical error OK)
    * Use prepare_array_pdf() if needed for flattening, domain changes, normalization, or cleaning

- If using midpoint_discretize():
    * 'pdf' must be a callable
    * 'bounds' is a list of (a, b) tuples for each dimension
    * 'bins' is an int or list of ints for number of bins per dimension

- If you already have a uniformly discretized, normalized PDFs defined over the full target domain range in 1D array form (e.g., from histogram or KDE):
    * Skip midpoint_discretize() and prepare_array_pdf() — pass the normalized arrays directly to
      sharpness_multi() or visualize_sharpness().
    * IMPORTANT: To use a specific target domain, your array must span the full
      domain (including explicit zeros for any region where the PDF is zero) so that
      sharpness_multi() can correctly account for all areas of the domain. Otherwise,
      use prepare_array_pdf() first.

PRACTICAL TIPS
--------------
- Higher bin counts improve accuracy; in 1D, bins = 10_000
  is a good balance for most cases.
- Use "simplified" mode for fastest computation; "ml" and "gini" are
  mathematically equivalent but yield additional interpretable curves.
- For dimensions >4, use sampling methods to evaluate the pdf (e.g., Monte Carlo)

TYPICAL WORKFLOWS
-----------------
1) Callable PDFs → Discretization → Sharpness:
    bounds = [(0.0, 4.0)]
    bins = 10000
    pdfs = midpoint_discretize(my_pdfs, bounds, bins)
    for i, pdf in enumerate(pdfs, 1):
        score = sharpness_multi(pdf, mode="simplified")
        print(f"PDF {i}: {score:.3f}")

2) Already discretized PDF → Sharpness:
    score = sharpness_multi(my_pdf_array, mode="simplified")

3) Discretized PDFs → Visualization:
    visualize_sharpness(my_pdfs, titles, mode="gini")

4) ----------------------------
   EXAMPLE: array-based PDF
   ----------------------------
   Suppose you already have a discretized, normalized PDF on a uniform grid over known, specified bounds in 1D
   array form, e.g. from histogram/KDE:

       # Load file, e.g. np.loadtxt(...), xarray, netCDF4, etc.
       example_pdf_vals = np.loadtxt("my_pdf_values.txt")  # shape = (N,), uniform grid, already normalized

       # Compute sharpness directly (no discretizer needed for arrays)
        for i, pdf in enumerate(example_pdf_vals, 1):
            score = sharpness_multi(pdf, mode="simplified")
            print(f"PDF {i}: {score:.3f}")

       # Or visualize (pass a list of arrays and titles)
       visualize_sharpness(example_pdf_vals, [list of titles], mode="gini")

EXAMPLE
-------
An example using 1D callable PDFs (mixture, Gaussians) is included
at the bottom of this file. It is commented out — uncomment the
'if __name__ == "__main__":' block to run it.
"""

import numpy as np
import matplotlib.pyplot as plt

# === 1a. Midpoint grid sampler ===
def midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=False):
    """
    Discretize a multidimensional PDF or a list of PDFs over a hyper-rectangular domain using the midpoint rule.
    Automatically:
      - Returns a 1D array of density values for each PDF, regardless of underlying dimensions
      - Zero-pads where the PDF can't be evaluated (e.g., larger given bounds than where the pdf is defined)
      - NaN and -inf → 0
      - +inf → large finite spike (scaled from max finite value)
      - Clips negatives to zero
      - If given bounds are smaller than the area where the pdf is defined, truncates the pdf over the given bounds
      - Optional renormalization to ensure the PDF integrates to 1
      - Raises ValueError if PDF has zero mass over the given bounds

    Parameters
    ----------
    pdf : callable or list of callables
        Function(s) returning density values.
    bounds : list of (float, float)
        Domain bounds [(a1, b1), (a2, b2), ..., (ad, bd)].
    bins : int or list of ints
        Number of bins per dimension.
    normalize : bool, default=False
        If True, normalize so the PDF integrates to 1.
    return_coords : bool, default=False
        If True, also return the coordinate arrays of the midpoints.

    Returns
    -------
    dvals : 1D numpy.ndarray or list of 1D numpy.ndarray
        Density values at midpoints. Returns a single array if one PDF was given,
        or a list of arrays if multiple PDFs were given.
    coords : list of numpy.ndarray, optional
        Midpoint coordinate arrays for each axis.
        Returned only if 'return_coords=True'.

    Raises
    ------
    ValueError
        If the cleaned PDF has zero total mass over the given bounds.
    """
    bounds = np.array(bounds, dtype=float)
    dims = bounds.shape[0]
    if np.isscalar(bins):
        bins = [int(bins)] * dims
    bins = np.array(bins, dtype=int)

    # Step sizes for each dimension
    widths = (bounds[:, 1] - bounds[:, 0]) / bins

    # Midpoint coordinates for each axis
    coords = [
        bounds[i, 0] + (np.arange(bins[i]) + 0.5) * widths[i]
        for i in range(dims)
    ]

    # Cartesian product of midpoints
    grids = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=-1)

    # Check list of PDFs or single PDF
    if callable(pdf):
        pdfs = [pdf]
        single = True
    elif isinstance(pdf, (list, tuple)):
        pdfs = pdf
        single = False
    else:
        raise TypeError("pdf must be a callable or list of callables")

    results = []
    for f in pdfs:
        try:
            dvals = f(pts) if dims > 1 else f(pts.ravel())
        except Exception:
            dvals = np.zeros(pts.shape[0], dtype=float)  # return zeros if failure

        dvals = np.asarray(dvals, float)

        # Handle NaN → 0, -inf → 0, +inf → large finite number
        nan_mask = np.isnan(dvals)
        neg_inf_mask = dvals == -np.inf
        pos_inf_mask = dvals == np.inf

        if np.any(pos_inf_mask):
            finite_mask = np.isfinite(dvals)
            if np.any(finite_mask):
                max_finite = np.max(dvals[finite_mask])
                replacement_value = max_finite * 1e6
            else:
                replacement_value = 1e6
            dvals[pos_inf_mask] = replacement_value

        dvals[nan_mask | neg_inf_mask] = 0.0

        # Clip negatives
        dvals = np.clip(dvals, 0, None)

        if normalize:
            # --- Normalization ---
            cell_volume = np.prod(widths)
            total_mass = np.sum(dvals) * cell_volume
            if total_mass <= 0:
                raise ValueError("PDF has zero total mass over the given bounds.")
            dvals /= total_mass

        results.append(dvals.ravel())

    if single:
        if return_coords:
            return results[0], coords
        return results[0]
    else:
        if return_coords:
            return results, coords
        return results

# === 1b. Prepare array for sharpness calculation ===
def prepare_array_pdf(array_pdf, coords, bounds=None, normalize=True, return_coords=False):
    """
    Prepare one or multiple array-form PDFs for sharpness calculation by
    padding/cropping to the target domain, cleaning infinities/NaNs,
    and optionally normalizing (default setting = True).

    Automatically:
      - Returns a 1D array(s) of density values, regardless of underlying dimensions
      - Zero-pads if the input array is smaller than the target grid implied by 'bounds'
      - Crops if the input array is larger than the target grid implied by 'bounds'
      - NaN and -inf → 0
      - +inf → large finite spike (scaled from max finite value)
      - Clips negatives to zero
      - Optional renormalization to ensure the PDF integrates to 1
      - Raises ValueError if PDF has zero mass over the given bounds

    Parameters
    ----------
    array_pdf : array_like or list of array_like
        PDF values (densities).
    coords : list of numpy.ndarray
        List of coordinate arrays for each dimension.
        Must correspond in shape to array_pdf.
    bounds : list of (float, float), optional
        Full domain bounds. If None, inferred from coords.
    normalize : bool, default=True
        Normalize so PDF integrates to 1.
    return_coords : bool, default=False
        Return the coordinate arrays.

    Returns
    -------
    dvals : 1D numpy.ndarray or list of 1D numpy.ndarray
        Flattened PDF values (optionally normalized).
    coords : list of numpy.ndarray, optional
        Coordinate arrays (possibly cropped/padded).
        Returned only if 'return_coords=True'.

    Raises
    ------
    ValueError
        If the cleaned PDF has zero total mass over the given bounds.
    """

    # Check list of arrays or single array
    if isinstance(array_pdf, (list, tuple)):
        multiple = True
        pdfs = array_pdf
        if isinstance(coords[0], np.ndarray):
            coords_list = [coords] * len(pdfs)
        else:
            coords_list = coords
    else:
        multiple = False
        pdfs = [array_pdf]
        coords_list = [coords]

    results = []
    coords_results = []

    for arr, coord_set in zip(pdfs, coords_list):
        arr = np.asarray(arr, float)
        if arr.ndim == 1 and len(coord_set) > 1:
            expected_shape = tuple(len(c) for c in coord_set)
            if arr.size == np.prod(expected_shape):
                arr = arr.reshape(expected_shape)

        dims = arr.ndim
        if len(coord_set) != dims:
            raise ValueError(f"coords must have {dims} arrays, got {len(coord_set)}")
        if any(len(c) != arr.shape[i] for i, c in enumerate(coord_set)):
            raise ValueError("Coordinate lengths must match array dimensions.")

        widths = np.array([(c[-1] - c[0]) / (len(c) - 1) if len(c) > 1 else 1.0 for c in coord_set])
        coord_starts = np.array([c[0] for c in coord_set], float)
        coord_ends = np.array([c[-1] for c in coord_set], float)
        coord_bins = np.array([len(c) for c in coord_set], int)

        if bounds is None:
            bounds_arr = np.stack([coord_starts, coord_ends], axis=1)
            target_bins = coord_bins.copy()
            same_grid = True
        else:
            bounds_arr = np.array(bounds, float)
            same_grid = np.allclose(bounds_arr[:, 0], coord_starts) and np.allclose(bounds_arr[:, 1], coord_ends)
            target_bins = coord_bins.copy() if same_grid else np.maximum(
                1, np.rint((bounds_arr[:, 1] - bounds_arr[:, 0]) / widths).astype(int)
            )

        if not same_grid:
            target_coords = [
                bounds_arr[i, 0] + (np.arange(target_bins[i]) + 0.5) * widths[i] for i in range(dims)
            ]
        else:
            target_coords = coords

        dvals = np.zeros(target_bins, float)
        slices_orig, slices_tgt = [], []
        for i in range(dims):
            o_start = int(np.ceil((target_coords[i][0] - coord_set[i][0]) / widths[i]))
            o_end = o_start + target_bins[i]
            t_start, t_end = 0, target_bins[i]

            if o_start < 0:
                t_start, o_start = -o_start, 0
            if o_end > coord_bins[i]:
                t_end -= o_end - coord_bins[i]
                o_end = coord_bins[i]

            if t_end <= t_start or o_end <= o_start:
                slices_orig.append(slice(0, 0))
                slices_tgt.append(slice(0, 0))
            else:
                slices_orig.append(slice(o_start, o_end))
                slices_tgt.append(slice(t_start, t_end))

        dvals[tuple(slices_tgt)] = arr[tuple(slices_orig)]

        # --- CLEANUP: identical to midpoint_discretize ---
        nan_mask = np.isnan(dvals)
        neg_inf_mask = dvals == -np.inf
        pos_inf_mask = dvals == np.inf

        if np.any(pos_inf_mask):
            finite_mask = np.isfinite(dvals)
            if np.any(finite_mask):
                max_finite = np.max(dvals[finite_mask])
                replacement_value = max_finite * 1e6
            else:
                replacement_value = 1e6
            dvals[pos_inf_mask] = replacement_value

        dvals[nan_mask | neg_inf_mask] = 0.0
        dvals = np.clip(dvals, 0, None)

        if normalize:
            cell_volume = np.prod(widths)
            total_mass = np.sum(dvals) * cell_volume
            if total_mass <= 0:
                raise ValueError("PDF has zero mass over the given bounds.")
            dvals /= total_mass

        results.append(dvals.ravel())
        if return_coords:
            coords_results.append(target_coords)

    # Return single or multiple depending on input
    if multiple:
        return (results, coords_results) if return_coords else results
    else:
        return (results[0], coords_results[0]) if return_coords else results[0]

# === 2. Multidimensional sharpness calculator ===
def sharpness_multi(dvals, mode="simplified", plot_data=False):
    """
    Calculate sharpness from precomputed density values.

    Parameters
    ----------
    dvals : array_like
        1D array of density values
    mode : str
        Sharpness mode: 'simplified', 'ml', or 'gini'.
    plot_data : bool, default=False
        If True, also return arrays needed for plotting in each mode.

    Returns
    -------
    score : float
        Sharpness score in [0, 1].
    plot_arrays : tuple, optional
        Additional arrays for plotting (only if plot_data=True):
        - simplified: t, t*d_sorted
        - ml: t, m, dL
        - gini: u, cum_mass
    """
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

# === 3. Visualization function ===
def visualize_sharpness(pdfs, titles=None, mode="gini"):
    """
    Visualize sharpness for select PDFs.

    Parameters
    ----------
    pdfs : array_like or list of array_like
        1D arrays of density values.
    titles : list of str, optional
        Optional titles/labels for each PDF. If not provided, will use
        "PDF 1", "PDF 2", etc. If fewer titles are provided than PDFs,
        remaining PDFs will be labeled automatically.
    mode : str
        Sharpness mode to visualize: 'simplified', 'ml', or 'gini'.
    """

    if not isinstance(pdfs, (list, tuple)):
        pdfs = [pdfs]

    n = len(pdfs)

    # Handle titles: fill missing with defaults
    if titles is None:
        titles = [f"PDF {i+1}" for i in range(n)]
    else:
        titles = list(titles) + [f"PDF {i+1}" for i in range(len(titles), n)]

    if mode == "simplified":
        plt.figure(figsize=(8, 5))
        for pdf, title in zip(pdfs, titles):
            score, t, t_times_d = sharpness_multi(pdf, mode="simplified", plot_data=True)
            plt.plot(t, t_times_d, label=f"{title}, $S = {score:.3f}$")
        plt.xlabel("t")
        plt.ylabel(r"$t \cdot d_*(t)$")
        plt.title("Integrands for pdfs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif mode == "ml":
        plt.figure(figsize=(7, 4 * len(pdfs)))
        for i, (pdf, title) in enumerate(zip(pdfs, titles), 1):
            score, t_vals, m, dL = sharpness_multi(pdf, mode="ml", plot_data=True)
            # Plot m(t) and d*(t)L(t) in their own subplot
            plt.subplot(len(pdfs), 1, i)
            plt.plot(t_vals, m, label="m(t)", color="#ffaf00", linewidth=2)
            plt.plot(t_vals, dL, label=r"$d_*(t)\cdot L(t)$", color="#f46920", linewidth=2)
            plt.title(f"{title} (S = {score:.3f})", fontsize=12)
            plt.xlabel("t")
            plt.ylabel("Integrand Value")
            plt.legend()
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif mode == "gini":
        plt.figure(figsize=(8, 5))
        plt.plot([0, 1], [0, 1], "k-", label="Uniform baseline")
        for pdf, title in zip(pdfs, titles):
            score, u, cum_mass = sharpness_multi(pdf, mode="gini", plot_data=True)
            plt.plot(u, cum_mass, label=f"{title}, $S = {score:.3f}$")
        plt.title("Gini-style curves for probability density functions")
        plt.xlabel("Fraction of Domain (u)")
        plt.ylabel("Cumulative Probability Mass")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("mode must be 'simplified', 'ml', or 'gini'")

# === 4. EXAMPLE (UNCOMMENT TO RUN) ===
#if __name__ == "__main__":
#    from scipy.stats import norm
#
#    # Domain
#    a, b = 0.0, 4.0
#    bounds = [(a, b)]
#    bins = 10_000
#
#    # Define PDFs
#    pdf_funcs = [
#        lambda x: 0.6 * norm.pdf(x, loc=1.2, scale=0.3) +
#                  0.4 * norm.pdf(x, loc=3.0, scale=0.4),
#        lambda x: norm.pdf(x, loc=2.8, scale=0.5),
#        lambda x: norm.pdf(x, loc=2.8, scale=0.1)
#    ]
#
#    # Discretize PDFs
#    pdfs = midpoint_discretize(pdf_funcs, bounds, bins, normalize=True)
#
#    titles = [
#        r"$0.6 \cdot \mathcal{N}(1.2, 0.3^2) + 0.4 \cdot \mathcal{N}(3.0, 0.4^2)$",
#        r"$\mathcal{N}(2.8, 0.5^2)$",
#        r"$\mathcal{N}(2.8, 0.1^2)$"
#    ]
#
#    print("\n--- Sharpness scores ---")
#    for i, pdf in enumerate(pdfs, 1):
#        score = sharpness_multi(pdf, mode="simplified")
#        print(f"PDF {i}: {score:.3f}")
#
#    # Run visualizations
#    print("\n--- Simplified mode ---")
#    visualize_sharpness(pdfs, titles, mode="simplified")
#
#    print("\n--- ML mode ---")
#    visualize_sharpness(pdfs, titles, mode="ml")
#
#    print("\n--- Gini mode ---")
#    visualize_sharpness(pdfs, titles, mode="gini")