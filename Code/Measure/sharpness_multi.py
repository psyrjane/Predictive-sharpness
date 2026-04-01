"""
Predictive Sharpness for Multidimensional Cases — Midpoint Discretization + Visualizations
--------------------------------------------------------------------------

This script implements the continuous sharpness measure S(f) from the paper:
"A Measure of Predictive Sharpness for Probabilistic Models" with support for multidimensional use, 
midpoint discretization, direct array inputs, and visualizations, using the three equivalent sharpness formulas:

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
     → Returns S(f) in [0, 1]

4) Visualizations of the three formulations:
   - visualize_sharpness(pdfs, titles, mode="simplified" | "ml" | "gini" | "cplot")
     → Simplified: plots t·f^uparrow(t)
     → ML: plots m(t) and f^uparrow(t)·L(t) separately
     → Gini: plots Lorenz-style cumulative mass curves
     → Cplot: concentration plot version of the ml-plot (inverse vase-view of ml)

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
- For dimensions >4, use sampling methods to evaluate the pdf instead (e.g., Monte Carlo)

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
def visualize_sharpness(pdfs, titles=None, mode="gini", show_fractional=True, mass_bins=4, zoom_y=0.0):
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
        Sharpness mode to visualize: 'simplified', 'ml', 'gini', or 'cplot'.
    show_fractional : bool, default=True
        Used only for mode='cplot'. If True, show the y-axis as fractions
        of the rearranged domain (0 to 1). If False, show the full rearranged
        domain scale (0 to |Omega|).
    mass_bins : int, default=4
        Used only for mode='cplot'. Number of mass-bins show.
        Must be an integer between 1-10.
    zoom_y : float, default=0.0
        Used only for mode='cplot'. Fraction of the rearranged domain from which
        to begin visualization. If 0.0, no zoom is applied. If nonzero, must be
        between 0.000001 and 0.999999. The visible portion is stretched to the 
        full plot height.
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

    elif mode == "cplot":
        if not isinstance(mass_bins, int) or not (1 <= mass_bins <= 10):
            raise ValueError("mass_bins must be an integer between 1 and 10.")

        if zoom_y != 0.0:
            if not (0.000001 <= zoom_y <= 0.999999):
                raise ValueError(
                    "zoom must be 0.0 (no zoom) or between 0.000001 and 0.999999."
                )

        def _concentration_plot_data(dvals):
            dvals = np.asarray(dvals, float).ravel()
            N = dvals.size
            L = 1.0 / dvals.mean()
            v = L / N

            score, t_left, m, dL = sharpness_multi(dvals, mode="ml", plot_data=True)

            # Recover sorted densities q from reverse cumulative mass m
            q = np.empty_like(m)
            if N > 1:
                q[:-1] = (m[:-1] - m[1:]) / v
            q[-1] = m[-1] / v

            t_mid = t_left + 0.5 * v
            delta = m - dL
            width = 1.0 - delta

            # Equal-mass bin edges in ranked space
            cum_mass = np.cumsum(q) * v
            mass_edges = np.linspace(0.0, 1.0, mass_bins + 1)

            # Interpolate on right-edge positions: 0, v, 2v, ..., L
            t_edges = np.interp(
                mass_edges,
                np.concatenate(([0.0], cum_mass)),
                np.linspace(0.0, L, N + 1)
            )

            # Downsample for plotting only
            n_plot = min(4500, N)  ### INCREASE FOR LARGE (HIGH-D) DOMAINS WITH VERY SMALL SUPPORT
            idx = np.linspace(0, N - 1, n_plot).astype(int)

            return {
                "t": t_mid[idx],
                "width": width[idx],
                "t_edges": t_edges,
                "sharpness": float(score),
                "L": float(L),
            }

        data_list = [_concentration_plot_data(pdf) for pdf in pdfs]

        ncols = min(3, n)
        nrows = int(np.ceil(n / 3))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.8 * ncols, 5.8 * nrows),
            sharey=True,
            constrained_layout=True
        )

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = axes[:, None]

        cmap = plt.get_cmap("tab10", mass_bins)

        for k, (ax, data, title) in enumerate(zip(axes.ravel(), data_list, titles)):
            t = data["t"]
            width = data["width"]
            t_edges = data["t_edges"]
            L = data["L"]

            # Fractional positions in original (unzoomed) rearranged domain
            frac = t / L
            frac_edges = t_edges / L

            local_handles = []

            if zoom_y == 0.0:
                if show_fractional:
                    y = frac
                    y_edges = frac_edges
                    y_max = 1.0
                else:
                    y = t
                    y_edges = t_edges
                    y_max = L

                ax.plot(width, y, color="black", lw=1.8)
                ax.plot(-width, y, color="black", lw=1.8)

                for i in range(mass_bins):
                    mask = (y >= y_edges[i]) & (y <= y_edges[i + 1])

                    if np.any(mask):
                        label = (
                            f"{int(round(100 * i / mass_bins))}–"
                            f"{int(round(100 * (i + 1) / mass_bins))}% mass"
                            if k == 0 else None
                        )

                        h = ax.fill_betweenx(
                            y[mask], -width[mask], width[mask],
                            color=cmap(i), alpha=0.72, label=label
                        )

                        if k == 0:
                            local_handles.append(h)

                for ye in y_edges[1:-1]:
                    w_ye = np.interp(ye, y, width)
                    ax.hlines(
                        ye,
                        xmin=-0.99 * w_ye,
                        xmax=0.99 * w_ye,
                        color="white",
                        lw=1.3,
                        alpha=0.95
                    )

                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(0, y_max)

                ax.set_title(f"{title}\n$S = {data['sharpness']:.3f}$", fontsize=10)

            else:
                # Keep only visible part
                visible_mask = frac >= zoom_y
                if not np.any(visible_mask):
                    ax.set_visible(False)
                    continue

                frac_vis = frac[visible_mask]
                width_vis = width[visible_mask]

                # Stretch visible portion to full plot height
                y_plot = (frac_vis - zoom_y) / (1.0 - zoom_y)

                ax.plot(width_vis, y_plot, color="black", lw=1.8)
                ax.plot(-width_vis, y_plot, color="black", lw=1.8)

                # Fill only mass bins still visible after zoom
                for i in range(mass_bins):
                    edge_lo = frac_edges[i]
                    edge_hi = frac_edges[i + 1]

                    # Skip bins entirely below the zoom threshold
                    if edge_hi <= zoom_y:
                        continue

                    mask = (
                        (frac >= max(edge_lo, zoom_y)) &
                        (frac <= edge_hi)
                    )

                    if np.any(mask):
                        y_bin = (frac[mask] - zoom_y) / (1.0 - zoom_y)
                        w_bin = width[mask]

                        label = (
                            f"{int(round(100 * i / mass_bins))}–"
                            f"{int(round(100 * (i + 1) / mass_bins))}% mass"
                            if k == 0 else None
                        )

                        h = ax.fill_betweenx(
                            y_bin, -w_bin, w_bin,
                            color=cmap(i), alpha=0.72, label=label
                        )

                        if k == 0:
                            local_handles.append(h)

                # Draw visible mass-bin boundaries only
                for fe in frac_edges[1:-1]:
                    if fe <= zoom_y:
                        continue
                    ye_plot = (fe - zoom_y) / (1.0 - zoom_y)
                    w_fe = np.interp(fe, frac, width)
                    ax.hlines(
                        ye_plot,
                        xmin=-0.99 * w_fe,
                        xmax=0.99 * w_fe,
                        color="white",
                        lw=1.3,
                        alpha=0.95
                    )

                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(0, 1.0)

                ax.set_title(
                    f"{title}\n$S = {data['sharpness']:.3f}$, zoom from {zoom_y:.2f}",
                    fontsize=10
                )

            ax.spines["top"].set_visible(False)
            ax.grid(False)
            ax.set_xlabel(r"$1-\Delta(t)$", fontsize=9)

            ax.set_xticks([-0.5, 0.5])
            ax.set_xticklabels(["0.5", "0.5"], fontsize=8)

            # y ticks
            if zoom_y != 0.0:
                tick_pos = np.linspace(0.0, 1.0, 6)
                frac_tick_vals = zoom_y + tick_pos * (1.0 - zoom_y)

                if show_fractional:
                    tick_labels = [f"{v:.3f}" for v in frac_tick_vals]
                else:
                    tick_labels = [f"{(v * L):.3f}" for v in frac_tick_vals]

                ax.set_yticks(tick_pos)
                ax.set_yticklabels(tick_labels, fontsize=8)

            col = k % ncols

            if col == 0:
                if show_fractional:
                    ax.set_ylabel("Fractions of rearranged domain (u)")
                else:
                    ax.set_ylabel("Rearranged domain (t)")
                ax.spines["left"].set_visible(True)
                ax.spines["right"].set_visible(False)
                ax.yaxis.tick_left()
                ax.yaxis.set_label_position("left")
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)

            if k == 0 and local_handles:
                ax.legend(
                    handles=local_handles,
                    fontsize=8,
                    loc="best",
                    frameon=True,
                    title="Mass bins"
                )

        # Hide unused axes
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)

        plt.show()

    else:
        raise ValueError("mode must be 'simplified', 'ml', 'gini', or 'cplot'")

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
#
#    print("\n--- Concentration plots mode ---")
#    visualize_sharpness(pdfs, titles, mode="cplot", show_fractional=True, mass_bins=4, zoom_y=0.0)
