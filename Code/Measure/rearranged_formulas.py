"""
Predictive Sharpness - Supplementary Formulas
--------------------------------------------------------------------------
This script implements the supplementary functions from the paper "A Measure of Predictive Sharpness for Probabilistic Models".
This script is designed to work with the functions in sharpness_multi.py.

WHAT THIS GIVES YOU
-------------------
The core functionalities include:
- Mapping from the original PDF to the rearranged space.
- Computing and visualizing various components such as mean, median, mass above, relative rank, and local contributions to sharpness score in the rearranged space.

The following functions are included:
1. analyze_pdf() - Main function to perform analysis of the PDF.
2. map_density_to_t() - Maps a given density to its corresponding location in the rearranged space.
3. find_mode() - Finds the mode of the PDF in the rearranged space.
4. find_median() - Finds the median of the PDF in the rearranged space.
5. find_mean() - Finds the mean of the PDF in the rearranged space.
6. map_point() - Maps a specific point in the original space to the rearranged space.
7. map_plateau() - If a point lies lies within a region of constant density (plateau), finds that region in the rearranged space.
8. local_contribution() - Calculates the local contribution to the sharpness score for a given subinterval in the rearranged space.
9. local_contribution_y() - Calculates the local contribution to the sharpness score for a region defined in the original space.
10. minimum_nonzero() - Finds the least non-zero density point in the rearranged space.
11. region_support() - Checks whether a given region in the original space has non-zero support.
12. mass_above() - Calculates the mass above a given point.
13. relative_likelihood() - Computes the relative likelihood of a point.
14. relative_rank() - Computes the relative rank of a point within the rearranged space.
15. plot_overlays() - Function to visualize the results of the analysis.

Example usage is provided for analyzing a 2D Gaussian PDF.

"""

import numpy as np
import matplotlib.pyplot as plt

# === 1a. Midpoint grid sampler === # From sharpness_multi.py
def midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=False):

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

# --- Main function for analysis ---
def analyze_pdf(pdf, bounds=None, bins=None, normalize=False):
    """
    Provides formulas for analyzing a given probability density function (PDF).

    Parameters:
    ----------
    pdf : callable or tuple of (dvals, coords)
        If a callable function, passes the function to midpoint_discretize() to convert it to dvals, coords tuple.
        If a tuple, it should contain the precomputed PDF values (dvals) and the corresponding coordinates (coords).
    bounds : list of tuples, optional
        The domain bounds for the discretization. Required if pdf is callable.
    bins : int, optional
        The number of bins for the discretization. Required if pdf is callable.
    normalize : bool, optional
        Normalization while passing midpoint_discretize() (True/False).

    Returns:
    -------
    dict : A dictionary containing functions for various calculations, including sharpness, median, mass above, relative rank, relative likelihood, and more.
    """
    if callable(pdf):
        if bounds is None or bins is None:
            raise ValueError("bounds and bins required for callable pdf")
        dvals, coords = midpoint_discretize(pdf, bounds, bins, normalize, return_coords=True)
        grids = np.meshgrid(*coords, indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=-1)

    elif isinstance(pdf, tuple):
        dvals, coords = pdf
        dvals = np.asarray(dvals, float).ravel()

        if not isinstance(coords, list):
            raise ValueError("coords must be a list of arrays for each dimension of the PDF")
        dims = len(coords)
        expected_shape = tuple(len(c) for c in coords)
        if len(dvals) != np.prod(expected_shape):
            raise ValueError(f"Shape mismatch: dvals length {len(dvals)} does not match the product of coords dimensions {np.prod(expected_shape)}")

        coords = [np.asarray(c, float) for c in coords]
        grids = np.meshgrid(*coords, indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=-1)

    else:
        raise ValueError("Pass either a callable pdf or (dvals, coords) tuple")

    # Rearranged space vars
    N = dvals.size
    L = 1.0 / dvals.mean()
    v = L / N
    d_sorted = np.sort(dvals)
    t = np.arange(N, dtype=float) * v
    m = np.cumsum(d_sorted[::-1])[::-1] * v
    dL = d_sorted * (L - t)
    score = (m[:-1] - dL[:-1]).sum() / N

    overlays = []  # shared overlay list

    def add_overlay(kind, **kwargs):
        overlays.append((kind, kwargs))

    def plot_ml_visualization():
        """Visualizes the mass-length components of the sharpness measure."""
        plt.figure(figsize=(10, 6))
        plt.plot(t, m, label="m(t)", color="#ffaf00", linewidth=2)
        plt.plot(t, dL, label=r"$d_*(t) \cdot L(t)$", color="#f46920", linewidth=2)
        plt.xlabel("t (Rearranged Domain)")
        plt.ylabel("Integrand Value")
        plt.title(f"ML Sharpness Components for PDF (S = {score:.3f})")
        plt.grid(True)
        plt.legend()

    def plot_overlays():
        """Plots overlays for subfunctions."""
        plot_ml_visualization()
        for kind, params in overlays:
            if kind == "axvline":
                plt.axvline(**params)
            elif kind == "axvline_between":
                t_val = params["t_val"]
                idx = np.argmin(np.abs(t - t_val))
                m_val = m[idx]
                dL_val = dL[idx]
                # Plot the vertical line between the m(t) and dL(t) curves
                plt.plot([t_val, t_val], [m_val, dL_val], color=params.get("color", "darkgray"), 
                         linestyle=params.get("linestyle", "--"), linewidth=params.get("linewidth", 2), 
                         label=params.get("label", ""))
            elif kind == "vlines_between":
                t_idx = params["t_idx"]
                color = params.get('color', 'steelblue')
                label = params.get('label', 'Regions') 
                if not any(overlay[0] == "vlines_between_legend" for overlay in overlays):
                    plt.plot([], [], color=color, lw=10, label=label)
                    overlays.append(("vlines_between_legend", {"color": color, "lw": 10, "label": label}))

                # Loop over the region's t-values
                for idx in t_idx:
                    t_val = t[idx]
                    m_val = m[idx]
                    dL_val = dL[idx]
                    # Plot the vertical lines between m(t) and d*(t) * L(t) curves
                    plt.vlines(t_val, ymin=min(m_val, dL_val), ymax=max(m_val, dL_val), 
                         color=color, alpha=0.01)
            elif kind == "point":
                plt.plot(params["x"], params["y"], 'o',
                         color=params.get("color", "black"),
                         label=params.get("label", None),
                         markersize=params.get("markersize", 8))
            elif kind == "fill_between":
                t_vals = params["t_vals"]
                m_vals = params["m_vals"]
                dL_vals = params["dL_vals"]
                color = params.get("color", "lightblue")
                alpha = params.get("alpha", 0.5)
                # Fill the area between m(t) and d*(t)L(t)
                plt.fill_between(t_vals, m_vals, dL_vals, color=color, alpha=alpha)
            elif kind == "dotted_line":
                # Plot a dotted line from the point to y=0 or y=1
                plt.plot([params["x"], params["x"]], [params["y"], params["y_end"]],
                         color=params.get("color", "black"), linestyle=params.get("linestyle", ":"),
                         linewidth=params.get("lw", 1))
            elif kind == "contribution_label":
                contribution_text = f"Contribution: {params['contribution']} ({params['percentage']}%)"
                plt.plot([], [], color=params.get("color", "lightblue"), lw=10, 
                         alpha=params.get("alpha", 0.5), label=contribution_text)

        plt.legend()
        plt.tight_layout()
        plt.show()
        overlays.clear()

    # === Analysis functions ===
    def map_density_to_t(density):
        j = np.argmin(np.abs(d_sorted - density))
        return t[j]

    def find_mode(ml_visualize=False):
        """
        Find the mode of the PDF in the rearranged space, and optionally visualize its position.

        Parameters:
        ----------
        ml_visualize : bool, optional
            If True, visualizations (vertical line) will be added to the plot at the mode position.

        Returns:
        -------
        mode_coordinates : list of float
            The coordinates of the mode in the original space.
        t_mode : float
            The position of the mode in the rearranged space.
        """
        idx = np.argmax(dvals)
        density = dvals[idx]
        t_mode = map_density_to_t(density)
        if ml_visualize:
            add_overlay("axvline", x=t_mode, color="red", linestyle="--", label=f"Mode (t = {round(t_mode, 2)})")

        return [round(coord, 2) for coord in points[idx].tolist()], float(round(t_mode, 2))

    def find_median(ml_visualize=False):
        """
        Find the median of the PDF in the rearranged space, and optionally visualize its position.

        Parameters:
        ----------
        ml_visualize : bool, optional
            If True, visualizations (vertical line) will be added to the plot at the median position.

        Returns:
        -------
        median_coordinates : list of float
            The coordinates of the median in the original space.
        t_median : float
            The position of the median in the rearranged space.
        """
        sort_idx = np.argsort(dvals)
        cum_mass = np.cumsum(dvals[sort_idx]) * v
        median_idx = np.searchsorted(cum_mass, 0.5)
        y_med = points[sort_idx[median_idx]]
        density = dvals[sort_idx[median_idx]]
        t_median = map_density_to_t(density)
        if ml_visualize:
            add_overlay("axvline", x=t_median, color="darkgreen", linestyle="--", label=f"Median (t = {round(t_median, 2)})")
        return [round(coord, 2) for coord in y_med.tolist()], float(round(t_median, 2))

    def find_mean(ml_visualize=False):
        """
        Find the mean of the PDF in the rearranged space, and optionally visualize its position.

        Parameters:
        ----------
        ml_visualize : bool, optional
            If True, visualizations (vertical line) will be added to the plot at the mean position.

        Returns:
        -------
        mean_coordinates : list of float
            The coordinates of the mean in the original space.
        t_mean : float
            The position of the mean in the rearranged space.
        """
        mean_density = np.mean(dvals)
        mean_idx = np.argmin(np.abs(dvals - mean_density))
        mean_point = points[mean_idx]
        t_mean = map_density_to_t(mean_density)
        if ml_visualize:
            add_overlay("axvline", x=t_mean, color="blue", linestyle="--", label=f"Mean (t = {round(t_mean, 2)})")
        return [round(coord, 2) for coord in points[mean_idx].tolist()], float(round(t_mean, 2))

    def map_point(y_p, ml_visualize=False):
        """
        Maps a specific point from the original space to the rearranged space based on its density.

        Given a point 'y_p', this function finds its corresponding location in the rearranged space
        based on the closest matching density.

        Parameters:
        ----------
        y_p : list of float
            The coordinates of the point in the original space.
        ml_visualize : bool, optional
            If True, a point will be visualized on the plot at the corresponding rearranged space position.

        Returns:
        -------
        t_point : float
            The position of the point in the rearranged space.
        """
        y_p = np.asarray(y_p)
        for i, coord in enumerate(coords):
            if not (np.min(coord) - 0.1 <= y_p[i] <= np.max(coord) + 0.1):
                raise ValueError("Value out of bounds")
        idx = np.argmin(np.sum((points - y_p)**2, axis=1))
        density = dvals[idx]
        t_point = map_density_to_t(density)
        if ml_visualize:
            m_val = m[np.argmin(np.abs(t - t_point))]
            add_overlay("point", x=t_point, y=m_val, color="blue", label=f"Point (t = {round(t_point, 2)})")
        return round(t_point, 4)

    def map_plateau(y_p, tolerance=1e-20, ml_visualize=False):
        """
        Maps a plateau (region of constant density) from the original space to the rearranged space.

        If a plateau exists at the point 'y_p', this function finds the entire region in the rearranged space
        that corresponds to the same density.

        Parameters:
        ----------
        y_p : list of float
            The coordinates of the point in the original space.
        tolerance : float, optional
            The tolerance for detecting plateau regions.
        ml_visualize : bool, optional
            If True, vertical lines will be drawn in the plot to indicate the start and end of the plateau region.

        Returns:
        -------
        t_a, t_b : float
            The start and end positions of the plateau in the rearranged space.
        """
        y_p = np.asarray(y_p)
        for i, coord in enumerate(coords):
            if not (np.min(coord) - 0.1 <= y_p[i] <= np.max(coord) + 0.1):
                raise ValueError("Value out of bounds")
        idx = np.argmin(np.sum((points - y_p)**2, axis=1))
        density = dvals[idx]
        t_indices = np.where(np.isclose(d_sorted, density, atol=tolerance))[0]
        if len(t_indices) == 0:
            # If no matching values found, use the closest match
            t_point = map_density_to_t(density)
            if ml_visualize:
                m_val = m[np.argmin(np.abs(t - t_point))]
                add_overlay("point", x=t_point, y=m_val, color="blue", label=f"Point (t = {round(t_point, 2)})")
            return round(t_point, 4)

        # Otherwise, return the first and last t-values corresponding to the plateau
        t_a, t_b = t[t_indices[0]], t[t_indices[-1]]
        if ml_visualize:
            add_overlay("axvline", x=t_a, color="magenta", linestyle="--", linewidth=1, label=f"Range Start (t = {round(t_a, 2)})")
            add_overlay("axvline", x=t_b, color="magenta", linestyle="--", linewidth=1, label=f"Range End (t = {round(t_b, 2)})")

        return float(round(t_a, 4)), float(round(t_b, 4))

    def local_contribution(t_a, t_b, ml_visualize=False):
        """
        Calculate the local contribution to the sharpness score for a specific subinterval in the rearranged space.

        Parameters:
        ----------
        t_a : float
            The starting point of the subinterval in the rearranged space.
        t_b : float
            The ending point of the subinterval in the rearranged space.
        ml_visualize : bool, optional
            If True, visualizations will be generated for the contribution, showing vertical lines and filled areas.

        Returns:
        -------
        contribution : float
            The local contribution to the sharpness score for the given subinterval.
        contribution_percentage : float
            The percentage of the total sharpness score contributed by this subinterval.
        """
        mask = (t >= t_a) & (t <= t_b)
        contribution = np.sum(m[mask] - dL[mask]) / N
        contribution_percentage = (contribution / score) * 100
        if ml_visualize:
            t_min = np.min(t[mask])
            t_max = np.max(t[mask])
            add_overlay("axvline_between", t_val=t_min, color="darkgray", linestyle="--", linewidth=2, label=f"Range Start (t = {round(t_a, 2)})")
            add_overlay("axvline_between", t_val=t_max, color="darkgray", linestyle="--", linewidth=2, label=f"Range End (t = {round(t_b, 2)})")
            add_overlay("fill_between", t_vals=t[mask], m_vals=m[mask], dL_vals=dL[mask], color="lightblue", alpha=0.5)
            add_overlay("contribution_label", contribution=round(contribution, 3), 
            percentage=round(contribution_percentage, 2))

        return float(round(contribution, 5)), float(round(contribution_percentage, 2))

    def local_contribution_y(region_bounds, ml_visualize=False):
        """
        Calculate the local contribution to sharpness score for a region of the original domain defined by 'region_bounds'.

        Parameters:
        ----------
        region_bounds : tuple of tuples
            The region defined by the set of bounds, e.g., 1D: ((x_min, x_max), ) or 2D: ((x_min, x_max), (y_min, y_max)).
        ml_visualize : bool, optional
            If True, visualizations will be generated for the defined region.
            NOTE: Visualizations represent the locations in the rearranged domain where
                  values are found but widths of the vertical lines are not scaled by bin width.
                  Visualizations are computationally heavy for large bounds and multidimensional pdfs.

        Returns:
        -------
        contribution : float
            The local contribution to sharpness score of the defined region.
        contribution_percentage : float
            The percentage contribution relative to the total sharpness score.
        """
        if not isinstance(region_bounds, tuple) or not all(isinstance(b, tuple) and len(b) == 2 for b in region_bounds):
            raise ValueError("region_bounds must be a tuple of tuples with two values for each dimension")

        for i, (min_val, max_val) in enumerate(region_bounds):
            if not (np.min(coords[i]) - 0.1 <= min_val <= np.max(coords[i]) + 0.1) or not (np.min(coords[i]) - 0.1 <= max_val <= np.max(coords[i]) + 0.1):
                raise ValueError("Value(s) out of bounds")

        mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(region_bounds):
            mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)
        region_indices = np.where(mask)[0]
        densities = dvals[region_indices]
        t_indices = np.searchsorted(d_sorted, densities)

        contribution = np.sum(m[t_indices] - dL[t_indices]) / N
        contribution_percentage = (contribution / score) * 100

        if ml_visualize:
            for i in range(1, len(t_indices)):
                if d_sorted[t_indices[i]] == d_sorted[t_indices[i - 1]]:
                    t_indices[i] = t_indices[i - 1] + 1 
            for idx in t_indices:
                add_overlay("vlines_between", t_idx=[idx], color="lightseagreen", label="Contributing regions")

        return float(round(contribution, 7)), float(round(contribution_percentage, 2))

    def minimum_nonzero(ml_visualize=False):
        """
        Find the least non-zero density point in the PDF, and returns the position of this point in the rearranged space.

        Parameters:
        ----------
        ml_visualize : bool, optional
            If True, a vertical line will be drawn on the plot at the location of the least non-zero density point.

        Returns:
        -------
        t_min : float
            The position of the least non-zero density point in the rearranged space.
        """
        d_min = np.min(dvals[dvals > 0])
        min_idx = np.argmin(np.abs(dvals - d_min))
        min_point = points[min_idx]
        t_min = map_density_to_t(d_min)
        if ml_visualize:
            add_overlay("axvline", x=t_min, color="black", linestyle="--", lw=2, label=f"Least non-zero (t = {round(t_min, 2)})")
        return round(t_min, 2)

    def region_support(region_bounds, ml_visualize=False):
        """
        Check whether the region defined by 'region_bounds' has non-zero support.

        Parameters:
        ----------
        region_bounds : tuple of tuples
            The region defined by the set of bounds, e.g., 1D: ((x_min, x_max), ) or 2D: ((x_min, x_max), (y_min, y_max)).
        ml_visualize : bool, optional
            If True, visualizations will be generated for the region of support.
            NOTE: Visualizations represent the locations in the rearranged domain where
                  values are found but widths of the vertical lines are not scaled by bin width.
                  Visualizations are computationally heavy for large bounds and multidimensional pdfs.

        Returns:
        -------
        support : bool
            Returns True if any of the points in the region have non-zero support, otherwise False.
        """
        if not isinstance(region_bounds, tuple) or not all(isinstance(b, tuple) and len(b) == 2 for b in region_bounds):
            raise ValueError("region_bounds must be a tuple of tuples with two values for each dimension")
        for i, (min_val, max_val) in enumerate(region_bounds):
            if not (np.min(coords[i]) - 0.1 <= min_val <= np.max(coords[i]) + 0.1) or not (np.min(coords[i]) - 0.1 <= max_val <= np.max(coords[i]) + 0.1):
                raise ValueError("Value(s) out of bounds")

        mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(region_bounds):
            mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)
        region_indices = np.where(mask)[0]
        densities = dvals[region_indices]
        support = np.any(densities > 0)

        if ml_visualize:
            positive_densities_mask = densities > 0
            positive_densities = densities[positive_densities_mask]
            t_indices = np.searchsorted(d_sorted, positive_densities)
            for i in range(1, len(t_indices)):
                if d_sorted[t_indices[i]] == d_sorted[t_indices[i - 1]]:
                    t_indices[i] = t_indices[i - 1] + 1 
            for idx in t_indices:
                add_overlay("vlines_between", t_idx=[idx], label="Regions of support")

        return support

    def mass_above(y_p, ml_visualize=False):
        """
        Calculate the mass of outcomes that have higher density than a specific point 'y_p'.

        Parameters:
        ----------
        y_p : list of float
            The coordinates of the point in the original space.
        ml_visualize : bool, optional
            If True, a vertical line will be drawn on the plot at the position corresponding to the mass above.

        Returns:
        -------
        mass : float
            The mass of higher density regions in the rearranged space.
        """
        y_p = np.asarray(y_p)
        for i, coord in enumerate(coords):
            if not (np.min(coord) - 0.1 <= y_p[i] <= np.max(coord) + 0.1):
                raise ValueError("Value out of bounds")
        idx = np.argmin(np.sum((points - y_p)**2, axis=1))
        density = dvals[idx]
        higher_density_idx = np.argmax(d_sorted > density)
        if higher_density_idx == 0:
            t_higher = t[-1] # final t-value (no higher densities)
        else:
            t_higher = t[higher_density_idx]
        t_idx = np.argmin(np.abs(t - t_higher))
        mass = m[t_idx]
        if ml_visualize:
            t_indices = np.where(np.isclose(d_sorted, density))[0]  # Find all indices of equal densities
            t_higher = t[t_indices[-1]]
            add_overlay("axvline", x=t_higher, color="purple", linestyle="--", lw=2, label=f"Mass above: {round(mass, 3)}")
        return round(mass, 3)

    def relative_likelihood(y_p, ml_visualize=False):
        """
        Compute the relative likelihood of an observed point and optionally visualize the relative likelihood in the rearranged space.

        The relative likelihood is calculated in the original domain as the density at 'y_p' divided by the maximum density.

        Parameters:
        ----------
        y_p : list of float
            The coordinates of the observed point in the original space.
        ml_visualize : bool, optional
            If True, relative likelihood is visualized by a dotted line on the plot.

        Returns:
        -------
        likelihood : float
            The relative likelihood of the observed point.
        """
        y_p = np.asarray(y_p)
        for i, coord in enumerate(coords):
            if not (np.min(coord) - 0.1 <= y_p[i] <= np.max(coord) + 0.1):
                raise ValueError("Value out of bounds")
        idx = np.argmin(np.sum((points - y_p)**2, axis=1))
        density = dvals[idx]
        d_max = np.max(dvals)
        likelihood = density / d_max
        if ml_visualize:
            t_indices = np.where(np.isclose(d_sorted, density))[0]  # Find all indices of equal densities
            t_p = t[t_indices[-1]]
            m_val = m[np.argmin(np.abs(t - t_p))]
            add_overlay("point", x=t_p, y=m_val, color="black", markersize=5, label=f"Relative Likelihood: {round(likelihood, 3)}")
            add_overlay("dotted_line", x=t_p, y=m_val, y_end=1, color="black", linestyle=":", lw=2)

        return round(likelihood, 10)

    def relative_rank(y_p, ml_visualize=False):
        """
        Compute the relative rank of an observed point 'y_p' in the rearranged space.

        Parameters:
        ----------
        y_p : list of float
            The coordinates of the observed point in the original space.
        ml_visualize : bool, optional
            If True, relative rank is visualized by a dotted line on the plot.

        Returns:
        -------
        rank : float
            The relative rank of the observed point within the rearranged space.
        """
        t_p = map_point(y_p)
        d_p = d_sorted[np.argmin(np.abs(t - t_p))]
        count_lower = np.sum(d_sorted < d_p)
        rank = count_lower / len(t)
        if ml_visualize:
            m_val = m[np.argmin(np.abs(t - t_p))]
            add_overlay("point", x=t_p, y=m_val, color="grey", markersize=5, label=f"Relative Rank: {round(rank, 3)}")
            add_overlay("dotted_line", x=t_p, y=m_val, y_end=0, color="grey", linestyle=":", lw=2)
        return round(rank, 3)

    return {
        "sharpness": score,
        "mode": find_mode,
        "median": find_median,
        "mean": find_mean,
        "point": map_point,
        "rl": relative_likelihood,
        "m_above": mass_above,
        "rr": relative_rank,
        "point_p": map_plateau,
        "local_contribution": local_contribution,
        "local_contribution_y" : local_contribution_y,
	"min_nonzero": minimum_nonzero,
        "r_support": region_support,
        "plot_overlays": plot_overlays,
    }

# === Example usage ===
if __name__ == "__main__":
    # Example: 2D Gaussian PDF
    def pdf_2d(pts):
        mean = np.array([0.0, 0.0])
        cov_inv = np.linalg.inv(np.array([[1.0, 0.0], [0.0, 1.0]]))
        norm_const = 1.0 / (2 * np.pi)
        diffs = pts - mean
        exps = -0.5 * np.sum(diffs @ cov_inv * diffs, axis=1)
        return norm_const * np.exp(exps)

    bounds = [(-4, 4), (-4, 4)]
    bins = 250

    # Get the analysis results
    analysis = analyze_pdf(pdf_2d, bounds, bins, normalize=True)

    # Example usage of the functions from the analysis
    print("Example calculations for 2D PDF over (-4, 4), (-4, 4)")
    print("Sharpness:", analysis["sharpness"])
    print("Mode:", analysis["mode"]())
    print("Median:", analysis["median"]())
    print("Mean:", analysis["mean"]())
    print("Point (1,1) location in rearranged space:", analysis["point"]([1, 1]))
    print("Relative likelihood of (1,1):", analysis["rl"]([1, 1]))
    print("Mass above (1,1):", analysis["m_above"]([1, 1]))
    print("Relative rank of (1,1):", analysis["rr"]([1, 1]))
    print("Plateau mapping of (1, 1):", analysis["point_p"]([1, 1]))
    print("Local contribution of (25.0, 35.0) in rearranged space:", analysis["local_contribution"](25.0, 35.0))
    print("Local contribution of region ((0, 1), (0, 1)) in original space:", analysis["local_contribution_y"](((0, 1), (0, 1))))
    print("Least non-zero y location in rearranged space:", analysis["min_nonzero"]())
    print("Support in region ((0, 1), (0, 1)) in original space:", analysis["r_support"](((0, 1), (0, 1))))

    # Optionally, plot the pdf
    def plot_pdf(pdf, bounds, bins):
        x, y = np.mgrid[bounds[0][0]:bounds[0][1]:complex(bins), bounds[1][0]:bounds[1][1]:complex(bins)]
        pdf_values = pdf(np.vstack([x.ravel(), y.ravel()]).T).reshape(x.shape)
        plt.figure(figsize=(8, 6))
        cp = plt.contourf(x, y, pdf_values, 20, cmap='viridis')
        plt.colorbar(cp)
        plt.title("2D Gaussian PDF")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    # Call for plotting the PDF example:
    # plot_pdf(pdf_2d, bounds, bins)

# === Example calls ===
# Print the mode and plot
#result = analysis["mode"](ml_visualize=True)
#print(f"Mode: {result}")
#analysis["plot_overlays"]()

# Print the median and plot
#result = analysis["median"](ml_visualize=True)
#print(f"Median: {result}")
#analysis["plot_overlays"]()

# Print the mean and plot
#result = analysis["mean"](ml_visualize=True)
#print(f"Mean: {result}")
#analysis["plot_overlays"]()

# Print the point and plot
#result = analysis["point"]([1, 1], ml_visualize=True)
#print(f"Point: {result}")
#analysis["plot_overlays"]()

# Print the relative likelihood and plot
#result = analysis["rl"]([1, 1], ml_visualize=True)
#print(f"Relative likelihood: {result}")
#analysis["plot_overlays"]()

# Print the mass above a point and plot
#result = analysis["m_above"]([1, 1], ml_visualize=True)
#print(f"Mass above: {result}")
#analysis["plot_overlays"]()

# Print the relative rank and plot
#result = analysis["rr"]([1, 1], ml_visualize=True)
#print(f"Relative rank: {result}")
#analysis["plot_overlays"]()

# Print the plateau mapping and plot
#result = analysis["point_p"]([1, 1], ml_visualize=True)
#print(f"Plateau mapping: {result}")
#analysis["plot_overlays"]()

# Print the local contribution and plot
#result = analysis["local_contribution"](25.0, 35.0, ml_visualize=True)
#print(f"Local contribution: {result}")
#analysis["plot_overlays"]()

# Print the local contribution for region in original space and plot
#result = analysis["local_contribution_y"](((0, 1), (0, 1)), ml_visualize=True)
#print(f"Local contribution for region in original space: {result}")
#analysis["plot_overlays"]()

# Print the minimum non-zero point and plot
#result = analysis["min_nonzero"](ml_visualize=True)
#print(f"Least non-zero y location: {result}")
#analysis["plot_overlays"]()

# Print the region support and plot
#result = analysis["r_support"](((0, 1), (0, 1)), ml_visualize=True)
#print(f"Region support: {result}")
#analysis["plot_overlays"]()

## === Plot multiple subfunctions ===
#analysis = analyze_pdf(pdf_2d, bounds, bins)
## Collect overlays from multiple calls
#analysis["mean"](ml_visualize=True)
#analysis["point"]([1, 1], ml_visualize=True)
#analysis["m_above"]([1, 1], ml_visualize=True)
## Plot all together:

#analysis["plot_overlays"]()
