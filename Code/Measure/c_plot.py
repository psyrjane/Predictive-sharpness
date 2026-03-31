""""
Predictive Sharpness - Supplementary Formulas (Concentration Plot Version)
--------------------------------------------------------------------------
This script implements the supplementary functions from the paper "A Measure of Predictive Sharpness for Probabilistic Models" for the concentration plot version.
This script is designed to work with the functions in sharpness_multi.py. The script reproduces
rearranged_formulas.py for the concentration plot version.

User-selected plot properties:
1. compress_y : bool
   - False -> show y-axis on the full rearranged-domain scale [0, |Omega|]
   - True  -> compress y-axis to fractions of the rearranged domain [0, 1]

2. mass_bins : int in [0, 10]
   - Number of cumulative-mass bands to color on the concentration plot
   - 0 means no coloring

3. zoom_y : float
   - 0.0 -> no zoom
   - in (0, 1) -> begin visualization from that fraction of the rearranged domain
     and stretch the visible portion to the full plot height

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

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# ============================================================
# 1a. Midpoint grid sampler
# ============================================================
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


# ============================================================
# Helpers
# ============================================================
def _infer_grid_widths_and_measure(coords):
    widths = []
    spans = []
    for c in coords:
        c = np.asarray(c, float)
        if c.size >= 2:
            w = float(np.median(np.diff(c)))
            span = (c[-1] - c[0]) + w
        else:
            w = 1.0
            span = 1.0
        widths.append(w)
        spans.append(span)
    widths = np.asarray(widths, float)
    return widths, float(np.prod(widths)), float(np.prod(spans))


def _validate_mass_bins(mass_bins):
    mass_bins = int(mass_bins)
    if not (0 <= mass_bins <= 10):
        raise ValueError("mass_bins must be an integer between 0 and 10.")
    return mass_bins


def _validate_zoom_y(zoom_y):
    zoom_y = float(zoom_y)
    if zoom_y != 0.0 and not (0.000001 <= zoom_y <= 0.999999):
        raise ValueError("zoom_y must be 0.0 (no zoom) or between 0.000001 and 0.999999.")
    return zoom_y


# ============================================================
# Main analysis function
# ============================================================
def c_plot_pdf(
    pdf,
    bounds=None,
    bins=None,
    normalize=False,
    compress_y=True,
    mass_bins=5,
    show_region_arrows=False,
    zoom_y=0.0,
):
    """
    Analyze a PDF in the rearranged space and visualize overlays on the
    mirrored concentration plot.

    Parameters
    ----------
    pdf : callable or tuple (dvals, coords)
        Either a callable PDF or a precomputed tuple of density values and coords.
    bounds : list of tuples, optional
        Bounds for discretization if pdf is callable.
    bins : int or list, optional
        Number of bins for discretization if pdf is callable.
    normalize : bool, optional
        Normalize the discretized PDF to total mass 1.
    compress_y : bool, optional
        If True, show the concentration-plot y-axis as fractions of the
        rearranged domain [0,1]; otherwise show [0, |Omega|] when zoom_y=0.
        Under zoom, the visible portion is stretched to [0,1] and tick labels
        follow the cplot behavior.
    mass_bins : int, optional
        Number of cumulative-mass color bands to show, from 0 to 10.
    show_region_arrows : bool, optional
        If true, show the range of contributions for different mass bins
        in local_contribution_y.
    zoom_y : float, optional
        Fraction of the rearranged domain from which to begin visualization.
        If 0.0, no zoom is applied. If nonzero, the visible portion is
        stretched to the full plot height.

    Returns
    -------
    dict
        Functions for analysis and plotting overlays.
    """
    if callable(pdf):
        if bounds is None or bins is None:
            raise ValueError("bounds and bins required for callable pdf")

        dvals, coords = midpoint_discretize(pdf, bounds, bins, normalize=False, return_coords=True)
        coords = [np.asarray(c, float) for c in coords]

        domain_measure = float(np.prod([b[1] - b[0] for b in bounds]))
        cell_volume = domain_measure / dvals.size

        if normalize:
            total_mass = np.sum(dvals) * cell_volume
            if total_mass <= 0:
                raise ValueError("PDF has zero total mass over the given bounds.")
            dvals = dvals / total_mass

    elif isinstance(pdf, tuple):
        dvals, coords = pdf
        dvals = np.asarray(dvals, float).ravel()

        if not isinstance(coords, list):
            raise ValueError("coords must be a list of arrays for each dimension of the PDF")
        coords = [np.asarray(c, float) for c in coords]

        expected_shape = tuple(len(c) for c in coords)
        if len(dvals) != np.prod(expected_shape):
            raise ValueError(
                f"Shape mismatch: dvals length {len(dvals)} does not match "
                f"the product of coords dimensions {np.prod(expected_shape)}"
            )

        _, cell_volume, domain_measure = _infer_grid_widths_and_measure(coords)

        if normalize:
            total_mass = np.sum(dvals) * cell_volume
            if total_mass <= 0:
                raise ValueError("PDF has zero total mass over the given grid.")
            dvals = dvals / total_mass

    else:
        raise ValueError("Pass either a callable pdf or (dvals, coords) tuple")

    grids = np.meshgrid(*coords, indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=-1)

    N = dvals.size
    L = domain_measure if domain_measure > 0 else 1.0 / max(dvals.mean(), 1e-16)
    v = L / N

    d_sorted = np.sort(np.clip(dvals, 0, None))
    t_left = np.arange(N, dtype=float) * v
    t_mid = t_left + 0.5 * v

    # Mass-length components
    m = np.cumsum(d_sorted[::-1])[::-1] * v
    dL = d_sorted * (L - t_left)
    delta = m - dL

    # Concentration-plot half-width
    width = np.clip(1.0 - delta, 0.0, 1.0)

    # Sharpness score
    score = float(np.sum(delta) / N)

    # Used for cumulative-mass banding
    cum_mass_used = np.cumsum(d_sorted) * v

    plot_state = {
        "compress_y": bool(compress_y),
        "mass_bins": _validate_mass_bins(mass_bins),
        "show_region_arrows": bool(show_region_arrows),
        "zoom_y": _validate_zoom_y(zoom_y),
    }
    overlays = []

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def configure_plot(compress_y=None, mass_bins=None, show_region_arrows=None, zoom_y=None):
        if compress_y is not None:
            plot_state["compress_y"] = bool(compress_y)
        if mass_bins is not None:
            plot_state["mass_bins"] = _validate_mass_bins(mass_bins)
        if show_region_arrows is not None:
            plot_state["show_region_arrows"] = bool(show_region_arrows)
        if zoom_y is not None:
            plot_state["zoom_y"] = _validate_zoom_y(zoom_y)

    def add_overlay(kind, **kwargs):
        overlays.append((kind, kwargs))

    def _display_y(t):
        t = np.asarray(t, dtype=float)
        return t / L if plot_state["compress_y"] else t

    def _width_at_t(t_val):
        return float(np.interp(t_val, t_mid, width, left=width[0], right=width[-1]))

    def _point_index(y_p):
        y_p = np.asarray(y_p, float)
        if y_p.shape[0] != points.shape[1]:
            raise ValueError("Point dimensionality does not match the PDF dimensionality.")
        for i, coord in enumerate(coords):
            if not (np.min(coord) - 0.1 <= y_p[i] <= np.max(coord) + 0.1):
                raise ValueError("Value out of bounds")
        return int(np.argmin(np.sum((points - y_p) ** 2, axis=1)))

    def _mass_edge_t(p):
        if p <= 0:
            return 0.0
        if p >= 1:
            return L
        i = int(np.searchsorted(cum_mass_used, p, side="left"))
        i = min(i, N - 1)
        prev = 0.0 if i == 0 else cum_mass_used[i - 1]
        qi = d_sorted[i]
        if qi <= 0:
            return float(t_left[i])
        return float(t_left[i] + (p - prev) / qi)

    def _lighten_color(color, frac=0.82):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1.0 - frac) * rgb + frac * np.ones(3))

    def _mass_bin_spec():
        n_bins = plot_state["mass_bins"]
        if n_bins <= 0:
            return np.array([0.0, L]), []

        base_colors = list(plt.get_cmap("tab10").colors)
        mass_edges = np.linspace(0.0, 1.0, n_bins + 1)
        t_edges = np.array([_mass_edge_t(p) for p in mass_edges], dtype=float)
        colors = [base_colors[i % len(base_colors)] for i in range(n_bins)]
        return t_edges, colors

    def _scalar_plot_y(t_val):
        frac = float(t_val) / L
        zoom = plot_state["zoom_y"]

        if zoom == 0.0:
            return (frac if plot_state["compress_y"] else float(t_val)), True

        if frac < zoom:
            return None, False

        return (frac - zoom) / (1.0 - zoom), True

    def _array_plot_y(t_vals):
        t_vals = np.asarray(t_vals, dtype=float)
        frac = t_vals / L
        zoom = plot_state["zoom_y"]

        if zoom == 0.0:
            y = frac if plot_state["compress_y"] else t_vals
            mask = np.ones_like(frac, dtype=bool)
            return y, mask, frac

        mask = frac >= zoom
        y = (frac[mask] - zoom) / (1.0 - zoom)
        return y, mask, frac

    # ------------------------------------------------------------
    # Base concentration-plot visualization
    # ------------------------------------------------------------
    def plot_concentration_visualization():
        fig, ax = plt.subplots(figsize=(10, 6))

        t_edges, colors = _mass_bin_spec()
        frac = t_mid / L
        frac_edges = t_edges / L
        zoom = plot_state["zoom_y"]

        if zoom == 0.0:
            if plot_state["compress_y"]:
                y = frac
                y_edges = frac_edges
                y_max = 1.0
            else:
                y = t_mid
                y_edges = t_edges
                y_max = L

            ax.plot(width, y, color="black", linewidth=2)
            ax.plot(-width, y, color="black", linewidth=2)

            if plot_state["mass_bins"] > 0:
                for i in range(len(colors)):
                    mask = (y >= y_edges[i]) & (y <= y_edges[i + 1])
                    if np.any(mask):
                        ax.fill_betweenx(
                            y,
                            -width,
                            width,
                            where=mask,
                            color=colors[i],
                            alpha=0.72,
                            linewidth=0,
                        )

                for ye in y_edges[1:-1]:
                    w_ye = np.interp(ye, y, width)
                    ax.hlines(
                        ye,
                        xmin=-0.99 * w_ye,
                        xmax=0.99 * w_ye,
                        color="white",
                        lw=1.2,
                        alpha=0.95,
                    )

            ax.set_ylim(0, y_max)
            ax.set_title(f"Concentration Plot for PDF (S = {score:.3f})")

        else:
            visible_mask = frac >= zoom
            if not np.any(visible_mask):
                raise ValueError("No visible portion remains after zoom.")

            frac_vis = frac[visible_mask]
            width_vis = width[visible_mask]
            y_plot = (frac_vis - zoom) / (1.0 - zoom)

            ax.plot(width_vis, y_plot, color="black", linewidth=2)
            ax.plot(-width_vis, y_plot, color="black", linewidth=2)

            if plot_state["mass_bins"] > 0:
                for i in range(len(colors)):
                    edge_lo = frac_edges[i]
                    edge_hi = frac_edges[i + 1]

                    if edge_hi <= zoom:
                        continue

                    mask = (frac >= max(edge_lo, zoom)) & (frac <= edge_hi)
                    if np.any(mask):
                        y_bin = (frac[mask] - zoom) / (1.0 - zoom)
                        w_bin = width[mask]
                        ax.fill_betweenx(
                            y_bin,
                            -w_bin,
                            w_bin,
                            color=colors[i],
                            alpha=0.72,
                            linewidth=0,
                        )

                for fe in frac_edges[1:-1]:
                    if fe <= zoom:
                        continue
                    ye_plot = (fe - zoom) / (1.0 - zoom)
                    w_fe = np.interp(fe, frac, width)
                    ax.hlines(
                        ye_plot,
                        xmin=-0.99 * w_fe,
                        xmax=0.99 * w_fe,
                        color="white",
                        lw=1.2,
                        alpha=0.95,
                    )

            ax.set_ylim(0, 1.0)
            ax.set_title(f"Concentration Plot for PDF (S = {score:.3f}, zoom from {zoom:.2f})")

            tick_pos = np.linspace(0.0, 1.0, 6)
            frac_tick_vals = zoom + tick_pos * (1.0 - zoom)

            if plot_state["compress_y"]:
                tick_labels = [f"{v:.3f}" for v in frac_tick_vals]
            else:
                tick_labels = [f"{(v * L):.3f}" for v in frac_tick_vals]

            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=8)

        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(r"$1-\Delta(t)$")

        if plot_state["compress_y"]:
            ax.set_ylabel(r"fractions of the rearranged domain (u)")
        else:
            ax.set_ylabel(r"rearranged domain $(t)$")

        ax.spines["top"].set_visible(False)
        ax.grid(False)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xticklabels(["1.0", "0.5", "0", "0.5", "1.0"])
        return fig, ax

    def plot_overlays():
        fig, ax = plot_concentration_visualization()
        used_labels = set()
        extra_handles = []

        def _use_label(label):
            if label is None or label == "":
                return None
            if label in used_labels:
                return None
            used_labels.add(label)
            return label

        for kind, params in overlays:
            if kind == "axhline":
                y_val, visible = _scalar_plot_y(params["t_val"])
                if visible:
                    ax.axhline(
                        y=y_val,
                        color=params.get("color", "black"),
                        linestyle=params.get("linestyle", "--"),
                        linewidth=params.get("linewidth", params.get("lw", 2)),
                        label=_use_label(params.get("label")),
                    )

            elif kind == "axhline_between":
                t_val = params["t_val"]
                y_val, visible = _scalar_plot_y(t_val)
                if visible:
                    w_val = _width_at_t(t_val)
                    ax.hlines(
                        y_val,
                        xmin=-0.99 * w_val,
                        xmax=0.99 * w_val,
                        color=params.get("color", "darkgray"),
                        linestyle=params.get("linestyle", "--"),
                        linewidth=params.get("linewidth", 2),
                        label=_use_label(params.get("label")),
                    )

            elif kind == "hlines_between":
                t_idx = np.asarray(params["t_idx"], int)
                color = params.get("color", "steelblue")
                alpha = params.get("alpha", 0.02)
                label = _use_label(params.get("label"))

                if label is not None:
                    ax.plot([], [], color=color, lw=8, alpha=min(1.0, max(alpha * 50, 0.25)), label=label)

                for idx in t_idx:
                    idx = int(np.clip(idx, 0, N - 1))
                    y_val, visible = _scalar_plot_y(t_mid[idx])
                    if not visible:
                        continue
                    w_val = width[idx]
                    ax.hlines(
                        y_val,
                        xmin=-w_val,
                        xmax=w_val,
                        color=color,
                        alpha=alpha,
                        linewidth=params.get("linewidth", 2),
                    )

            elif kind == "fill_betweenx":
                t_vals = np.asarray(params["t_vals"], float)
                x_left = np.asarray(params["x_left"], float)
                x_right = np.asarray(params["x_right"], float)

                y_plot, mask, _ = _array_plot_y(t_vals)
                if np.any(mask):
                    ax.fill_betweenx(
                        y_plot,
                        x_left[mask],
                        x_right[mask],
                        color=params.get("color", "lightblue"),
                        alpha=params.get("alpha", 0.5),
                    )

            elif kind == "dotted_line":
                y_val, visible = _scalar_plot_y(params["t_val"])
                if visible:
                    ax.plot(
                        [params["x"], params["x_end"]],
                        [y_val, y_val],
                        color=params.get("color", "black"),
                        linestyle=params.get("linestyle", ":"),
                        linewidth=params.get("lw", 1.5),
                        label=_use_label(params.get("label")),
                    )

            elif kind == "contribution_label":
                contribution_text = f"Contribution: {params['contribution']} ({params['percentage']}%)"
                ax.plot(
                    [],
                    [],
                    color=params.get("color", "lightblue"),
                    lw=10,
                    alpha=params.get("alpha", 0.5),
                    label=_use_label(contribution_text),
                )

            elif kind == "region_bin_contribution":
                t_edges = np.asarray(params["t_edges"], float)
                ratios = np.asarray(params["ratios"], float)
                t_lows = np.asarray(params["t_lows"], float)
                t_highs = np.asarray(params["t_highs"], float)
                zero_starts = np.asarray(params["zero_starts"], float)
                zero_ends = np.asarray(params["zero_ends"], float)
                colors = params["colors"]
                light_frac = params.get("light_frac", 0.90)

                frac = t_mid / L
                frac_edges = t_edges / L
                zoom = plot_state["zoom_y"]

                for i in range(len(ratios)):
                    base_color = colors[i]
                    light_color = _lighten_color(base_color, frac=light_frac)

                    if zoom == 0.0:
                        if i < len(ratios) - 1:
                            mask_bin = (t_mid >= t_edges[i]) & (t_mid < t_edges[i + 1])
                        else:
                            mask_bin = (t_mid >= t_edges[i]) & (t_mid <= t_edges[i + 1])

                        if not np.any(mask_bin):
                            continue

                        # Lighten the whole bin
                        ax.fill_betweenx(
                            _display_y(t_mid[mask_bin]),
                            -width[mask_bin],
                            width[mask_bin],
                            color=light_color,
                            alpha=1.0,
                            linewidth=0,
                        )

                        # Full-width fill from t_min to t_max
                        if np.isfinite(t_lows[i]) and np.isfinite(t_highs[i]) and (t_highs[i] > t_lows[i]):
                            mask_span = ((t_mid + 0.5 * v) > t_lows[i]) & ((t_mid - 0.5 * v) < t_highs[i])
                            if np.any(mask_span):
                                ax.fill_betweenx(
                                    _display_y(t_mid[mask_span]),
                                    -width[mask_span],
                                    width[mask_span],
                                    color=base_color,
                                    alpha=0.95,
                                    linewidth=0,
                                )

                        # Zero-density plateau -> center bar
                        if np.isfinite(zero_starts[i]) and np.isfinite(zero_ends[i]) and (zero_ends[i] > zero_starts[i]):
                            y0, vis0 = _scalar_plot_y(zero_starts[i])
                            y1, vis1 = _scalar_plot_y(zero_ends[i])
                            if vis0 or vis1:
                                y_low = max(0.0, y0 if vis0 else 0.0)
                                y_high = y1 if vis1 else 0.0
                                if y_high > y_low:
                                    ax.plot(
                                        [0.0, 0.0],
                                        [y_low, y_high],
                                        color=base_color,
                                        linewidth=2.6,
                                        solid_capstyle="butt",
                                    )

                        # Arrows
                        if plot_state["show_region_arrows"]:
                            arrow_len = 0.045
                            arrow_side = "right" if i % 2 == 0 else "left"

                            for t_arrow in (t_lows[i], t_highs[i]):
                                if np.isfinite(t_arrow):
                                    y_arrow, visible = _scalar_plot_y(t_arrow)
                                    if not visible:
                                        continue

                                    if arrow_side == "right":
                                        x_tip = _width_at_t(t_arrow)
                                        x_tail = min(1.04, x_tip + arrow_len)
                                    else:
                                        x_tip = -_width_at_t(t_arrow)
                                        x_tail = max(-1.04, x_tip - arrow_len)

                                    ax.annotate(
                                        "",
                                        xy=(x_tip, y_arrow),
                                        xytext=(x_tail, y_arrow),
                                        arrowprops=dict(
                                            arrowstyle="->",
                                            color=base_color,
                                            lw=1.2,
                                            shrinkA=0,
                                            shrinkB=0,
                                            mutation_scale=7,
                                        ),
                                    )

                    else:
                        edge_lo = frac_edges[i]
                        edge_hi = frac_edges[i + 1]

                        if edge_hi <= zoom:
                            continue

                        mask_bin = (frac >= max(edge_lo, zoom)) & (frac <= edge_hi)
                        if not np.any(mask_bin):
                            continue

                        y_bin = (frac[mask_bin] - zoom) / (1.0 - zoom)
                        w_bin = width[mask_bin]

                        # Lighten the whole visible bin
                        ax.fill_betweenx(
                            y_bin,
                            -w_bin,
                            w_bin,
                            color=light_color,
                            alpha=1.0,
                            linewidth=0,
                        )

                        # Full-width fill from t_min to t_max
                        if np.isfinite(t_lows[i]) and np.isfinite(t_highs[i]) and (t_highs[i] > t_lows[i]):
                            span_lo = t_lows[i] / L
                            span_hi = t_highs[i] / L
                            if span_hi > zoom:
                                mask_span = (frac >= max(span_lo, zoom)) & (frac <= span_hi)
                                if np.any(mask_span):
                                    y_span = (frac[mask_span] - zoom) / (1.0 - zoom)
                                    w_span = width[mask_span]
                                    ax.fill_betweenx(
                                        y_span,
                                        -w_span,
                                        w_span,
                                        color=base_color,
                                        alpha=0.95,
                                        linewidth=0,
                                    )

                        # Zero-density plateau -> center bar
                        if np.isfinite(zero_starts[i]) and np.isfinite(zero_ends[i]) and (zero_ends[i] > zero_starts[i]):
                            z0 = zero_starts[i] / L
                            z1 = zero_ends[i] / L
                            if z1 > zoom:
                                z0_clip = max(z0, zoom)
                                y_zero = (np.array([z0_clip, z1]) - zoom) / (1.0 - zoom)
                                ax.plot(
                                    [0.0, 0.0],
                                    y_zero,
                                    color=base_color,
                                    linewidth=2.6,
                                    solid_capstyle="butt",
                                )

                        # Arrows
                        if plot_state["show_region_arrows"]:
                            arrow_len = 0.045
                            arrow_side = "right" if i % 2 == 0 else "left"

                            for t_arrow in (t_lows[i], t_highs[i]):
                                if np.isfinite(t_arrow):
                                    y_arrow, visible = _scalar_plot_y(t_arrow)
                                    if not visible:
                                        continue

                                    if arrow_side == "right":
                                        x_tip = _width_at_t(t_arrow)
                                        x_tail = min(1.04, x_tip + arrow_len)
                                    else:
                                        x_tip = -_width_at_t(t_arrow)
                                        x_tail = max(-1.04, x_tip - arrow_len)

                                    ax.annotate(
                                        "",
                                        xy=(x_tip, y_arrow),
                                        xytext=(x_tail, y_arrow),
                                        arrowprops=dict(
                                            arrowstyle="->",
                                            color=base_color,
                                            lw=1.2,
                                            shrinkA=0,
                                            shrinkB=0,
                                            mutation_scale=7,
                                        ),
                                    )

                    mass_start = 100 * i / len(ratios)
                    mass_end = 100 * (i + 1) / len(ratios)

                    extra_handles.append(
                        Patch(
                            facecolor=base_color,
                            edgecolor="none",
                            label=f"{mass_start:.0f}–{mass_end:.0f}% bin: {ratios[i]:.2f}"
                        )
                    )

        handles, labels = ax.get_legend_handles_labels()
        handles = handles + extra_handles
        labels = labels + [h.get_label() for h in extra_handles]
        if labels:
            ax.legend(handles, labels)
        plt.tight_layout()
        plt.show()
        overlays.clear()

    # ------------------------------------------------------------
    # Analysis functions
    # ------------------------------------------------------------
    def map_density_to_t(density):
        j = int(np.argmin(np.abs(d_sorted - density)))
        return float(t_mid[j])

    def find_mode(ml_visualize=False):
        idx = int(np.argmax(dvals))
        density = float(dvals[idx])
        t_mode = map_density_to_t(density)
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_mode,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mode (t = {t_mode:.2f})",
            )
        return [round(coord, 2) for coord in points[idx].tolist()], float(round(t_mode, 4))

    def find_median(ml_visualize=False):
        sort_idx = np.argsort(dvals)
        cum_mass = np.cumsum(dvals[sort_idx]) * cell_volume
        median_idx = int(np.searchsorted(cum_mass, 0.5))
        median_idx = min(median_idx, len(sort_idx) - 1)
        y_med = points[sort_idx[median_idx]]
        density = dvals[sort_idx[median_idx]]
        t_median = map_density_to_t(density)
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_median,
                color="darkgreen",
                linestyle="--",
                linewidth=2,
                label=f"Median (t = {t_median:.2f})",
            )
        return [round(coord, 2) for coord in y_med.tolist()], float(round(t_median, 4))

    def find_mean(ml_visualize=False):
        mean_density = np.mean(dvals)
        mean_idx = np.argmin(np.abs(dvals - mean_density))
        mean_point = points[mean_idx]
        t_mean = map_density_to_t(mean_density)
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_mean,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Mean (t = {t_mean:.2f})",
            )
        return [round(coord, 2) for coord in mean_point.tolist()], float(round(t_mean, 4))

    def map_point(y_p, ml_visualize=False):
        idx = _point_index(y_p)
        density = float(dvals[idx])
        t_point = map_density_to_t(density)
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_point,
                color="royalblue",
                linestyle="-",
                linewidth=2,
                label=f"Point (t = {t_point:.2f})",
            )
        return float(round(t_point, 4))

    def map_plateau(y_p, tolerance=1e-20, ml_visualize=False):
        idx = _point_index(y_p)
        density = float(dvals[idx])
        t_indices = np.where(np.isclose(d_sorted, density, atol=tolerance))[0]

        if len(t_indices) == 0:
            t_point = map_density_to_t(density)
            if ml_visualize:
                add_overlay(
                    "axhline_between",
                    t_val=t_point,
                    color="blue",
                    linestyle="-",
                    linewidth=2,
                    label=f"Point (t = {t_point:.2f})",
                )
            return float(round(t_point, 4))

        t_a, t_b = float(t_mid[t_indices[0]]), float(t_mid[t_indices[-1]])
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_a,
                color="magenta",
                linestyle="--",
                linewidth=1.5,
                label=f"Range Start (t = {t_a:.2f})",
            )
            add_overlay(
                "axhline_between",
                t_val=t_b,
                color="magenta",
                linestyle="--",
                linewidth=1.5,
                label=f"Range End (t = {t_b:.2f})",
            )
        return float(round(t_a, 4)), float(round(t_b, 4))

    def local_contribution(t_a, t_b, ml_visualize=False):
        t0, t1 = sorted((float(t_a), float(t_b)))
        mask = (t_mid >= t0) & (t_mid <= t1)

        contribution = float(np.sum(delta[mask]) / N)
        contribution_percentage = float((contribution / score) * 100) if score > 0 else 0.0

        if ml_visualize and np.any(mask):
            add_overlay(
                "axhline_between",
                t_val=float(t_mid[mask][0]),
                color="darkgray",
                linestyle="--",
                linewidth=2,
                label=f"Range Start (t = {t0:.2f})",
            )
            add_overlay(
                "axhline_between",
                t_val=float(t_mid[mask][-1]),
                color="darkgray",
                linestyle="--",
                linewidth=2,
                label=f"Range End (t = {t1:.2f})",
            )
            add_overlay(
                "fill_betweenx",
                t_vals=t_mid[mask],
                x_left=-width[mask],
                x_right=width[mask],
                color="lightblue",
                alpha=0.5,
            )
            add_overlay(
                "contribution_label",
                contribution=round(contribution, 3),
                percentage=round(contribution_percentage, 2),
            )

        return float(round(contribution, 7)), float(round(contribution_percentage, 2))

    def local_contribution_y(region_bounds, ml_visualize=False):
        if not isinstance(region_bounds, tuple) or not all(isinstance(b, tuple) and len(b) == 2 for b in region_bounds):
            raise ValueError("region_bounds must be a tuple of tuples with two values for each dimension")

        mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(region_bounds):
            if not (np.min(coords[i]) - 0.1 <= min_val <= np.max(coords[i]) + 0.1) or not (
                np.min(coords[i]) - 0.1 <= max_val <= np.max(coords[i]) + 0.1
            ):
                raise ValueError("Value(s) out of bounds")
            mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)

        region_indices = np.where(mask)[0]
        densities = dvals[region_indices]
        t_indices = np.searchsorted(d_sorted, densities, side="left")
        t_indices = np.clip(t_indices, 0, N - 1)

        contribution = float(np.sum(delta[t_indices]) / N)
        contribution_percentage = float((contribution / score) * 100) if score > 0 else 0.0

        if ml_visualize and len(t_indices) > 0 and plot_state["mass_bins"] > 0:
            t_edges, bin_colors = _mass_bin_spec()
            n_bins = len(bin_colors)

            ratios = np.zeros(n_bins, dtype=float)
            t_lows = np.full(n_bins, np.nan, dtype=float)
            t_highs = np.full(n_bins, np.nan, dtype=float)
            zero_starts = np.full(n_bins, np.nan, dtype=float)
            zero_ends = np.full(n_bins, np.nan, dtype=float)

            # Spread repeated selected densities across their plateau in rearranged space
            t_selected_vis = np.full(len(densities), np.nan, dtype=float)
            assigned = np.zeros(len(densities), dtype=bool)
            top_density = float(d_sorted[-1])

            for k in range(len(densities)):
                if assigned[k]:
                    continue

                    dens0 = densities[k]
                dens0 = densities[k]
                same_mask = (~assigned) & np.isclose(densities, dens0, atol=1e-14, rtol=1e-12)
                same_idx = np.where(same_mask)[0]

                plateau_idx = np.where(np.isclose(d_sorted, dens0, atol=1e-14, rtol=1e-12))[0]

                if len(plateau_idx) == 0:
                    t_selected_vis[same_idx] = t_mid[t_indices[same_idx]]
                elif len(plateau_idx) == 1:
                    t_selected_vis[same_idx] = t_mid[plateau_idx[0]]
                else:
                    n_sel = len(same_idx)

                    # For a flat top plateau, place the selected segment at the top end.
                    # Otherwise place it from the lower end of the plateau.
                    if np.isclose(dens0, top_density, atol=1e-14, rtol=1e-12) and np.allclose(delta[plateau_idx], 0.0, atol=1e-12):
                        chosen = plateau_idx[max(0, len(plateau_idx) - n_sel):]
                    else:
                        chosen = plateau_idx[:n_sel]

                    if len(chosen) < n_sel:
                        chosen = np.pad(chosen, (0, n_sel - len(chosen)), mode="edge")

                    t_selected_vis[same_idx] = t_mid[chosen[:n_sel]]

                assigned[same_idx] = True

            for i in range(n_bins):
                if i < n_bins - 1:
                    mask_all = (t_mid >= t_edges[i]) & (t_mid < t_edges[i + 1])
                    mask_sel_vis = (t_selected_vis >= t_edges[i]) & (t_selected_vis < t_edges[i + 1])
                else:
                    mask_all = (t_mid >= t_edges[i]) & (t_mid <= t_edges[i + 1])
                    mask_sel_vis = (t_selected_vis >= t_edges[i]) & (t_selected_vis <= t_edges[i + 1])

                total_bin_contrib = np.sum(delta[mask_all])

                if np.any(mask_sel_vis):
                    sel_idx = t_indices[mask_sel_vis]
                    sel_densities = densities[mask_sel_vis]
                    sel_t_vis = t_selected_vis[mask_sel_vis]

                    selected_bin_contrib = np.sum(delta[sel_idx])

                    if total_bin_contrib > 0:
                        ratios[i] = selected_bin_contrib / total_bin_contrib
                    else:
                        ratios[i] = 0.0

                    zero_mask = np.isclose(sel_densities, 0.0, atol=1e-14, rtol=1e-12)
                    if np.any(zero_mask):
                        zero_t = sel_t_vis[zero_mask]
                        zero_starts[i] = np.min(zero_t) - 0.5 * v
                        zero_ends[i] = np.max(zero_t) + 0.5 * v

                    regular_mask = ~zero_mask
                    if np.any(regular_mask):
                        regular_t = sel_t_vis[regular_mask]
                        t_lows[i] = np.min(regular_t) - 0.5 * v
                        t_highs[i] = np.max(regular_t) + 0.5 * v

            add_overlay(
                "region_bin_contribution",
                t_edges=t_edges,
                ratios=np.clip(ratios, 0.0, 1.0),
                t_lows=t_lows,
                t_highs=t_highs,
                zero_starts=zero_starts,
                zero_ends=zero_ends,
                colors=bin_colors,
                light_frac=0.90,
                label="Region contribution by mass bin",
            )

        return float(round(contribution, 7)), float(round(contribution_percentage, 2))

    def minimum_nonzero(ml_visualize=False):
        positive = dvals[dvals > 0]
        if positive.size == 0:
            raise ValueError("PDF has no positive densities on the grid.")
        d_min = float(np.min(positive))
        t_min = map_density_to_t(d_min)
        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_min,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Least non-zero (t = {t_min:.2f})",
            )
        return float(round(t_min, 4))

    def region_support(region_bounds, ml_visualize=False):
        if not isinstance(region_bounds, tuple) or not all(isinstance(b, tuple) and len(b) == 2 for b in region_bounds):
            raise ValueError("region_bounds must be a tuple of tuples with two values for each dimension")

        mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(region_bounds):
            if not (np.min(coords[i]) - 0.1 <= min_val <= np.max(coords[i]) + 0.1) or not (
                np.min(coords[i]) - 0.1 <= max_val <= np.max(coords[i]) + 0.1
            ):
                raise ValueError("Value(s) out of bounds")
            mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)

        region_indices = np.where(mask)[0]
        densities = dvals[region_indices]
        support = bool(np.any(densities > 0))

        if ml_visualize and support:
            positive_densities = densities[densities > 0]
            t_indices = np.searchsorted(d_sorted, positive_densities, side="left")
            t_indices = np.clip(t_indices, 0, N - 1)
            add_overlay(
                "hlines_between",
                t_idx=t_indices,
                color="orange",
                alpha=0.03,
                label="Regions of support",
            )

        return support

    def mass_above(y_p, ml_visualize=False):
        idx = _point_index(y_p)
        density = float(dvals[idx])

        greater = np.where(d_sorted > density)[0]
        if greater.size == 0:
            mass = 0.0
            t_higher = float(t_mid[-1])
        else:
            first_idx = int(greater[0])
            mass = float(m[first_idx])
            t_higher = float(t_mid[first_idx])

        if ml_visualize:
            add_overlay(
                "axhline_between",
                t_val=t_higher,
                color="purple",
                linestyle="--",
                linewidth=2,
                label=f"Mass above: {mass:.3f}",
            )

        return float(round(mass, 6))

    def relative_likelihood(y_p, ml_visualize=False):
        idx = _point_index(y_p)
        density = float(dvals[idx])
        d_max = float(np.max(dvals))
        likelihood = float(density / d_max) if d_max > 0 else 0.0

        if ml_visualize:
            t_p = map_density_to_t(density)
            x_p = _width_at_t(t_p)
            add_overlay(
                "axhline_between",
                t_val=t_p,
                color="black",
                linestyle="-",
                linewidth=2,
                label=f"Relative Likelihood: {likelihood:.3f}",
            )
            add_overlay(
                "dotted_line",
                x=x_p,
                x_end=1.0,
                t_val=t_p,
                color="black",
                linestyle=":",
                lw=2,
            )

        return float(round(likelihood, 10))

    def relative_rank(y_p, ml_visualize=False):
        idx = _point_index(y_p)
        density = float(dvals[idx])
        count_lower = int(np.sum(d_sorted < density))
        rank = float(count_lower / len(t_mid))

        if ml_visualize:
            t_p = map_density_to_t(density)
            x_p = _width_at_t(t_p)
            add_overlay(
                "axhline_between",
                t_val=t_p,
                color="grey",
                linestyle="-",
                linewidth=2,
                label=f"Relative Rank: {rank:.3f}",
            )
            add_overlay(
                "dotted_line",
                x=x_p,
                x_end=0.0,
                t_val=t_p,
                color="grey",
                linestyle=":",
                lw=2,
            )

        return float(round(rank, 6))

    return {
        "sharpness": score,
        "configure_plot": configure_plot,
        "mode": find_mode,
        "median": find_median,
        "mean": find_mean,
        "point": map_point,
        "rl": relative_likelihood,
        "m_above": mass_above,
        "rr": relative_rank,
        "point_p": map_plateau,
        "local_contribution": local_contribution,
        "local_contribution_y": local_contribution_y,
        "min_nonzero": minimum_nonzero,
        "r_support": region_support,
        "plot_overlays": plot_overlays,
    }


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    def pdf_2d(pts):
        mean = np.array([0.0, 0.0])
        cov_inv = np.linalg.inv(np.array([[1.0, 0.0], [0.0, 1.0]]))
        norm_const = 1.0 / (2 * np.pi)
        diffs = pts - mean
        exps = -0.5 * np.sum(diffs @ cov_inv * diffs, axis=1)
        return norm_const * np.exp(exps)

    bounds = [(-4, 4), (-4, 4)]
    bins = 250

    c_plot = c_plot_pdf(
        pdf_2d,
        bounds=bounds,
        bins=bins,
        normalize=True,
        compress_y=True,
        mass_bins=4,
        zoom_y=0.0,
    )

    print("Example calculations for 2D PDF over (-4, 4), (-4, 4)")
    print("Sharpness:", c_plot["sharpness"])
    print("Mode:", c_plot["mode"]())
    print("Median:", c_plot["median"]())
    print("Mean:", c_plot["mean"]())
    print("Point (1,1) location in rearranged space:", c_plot["point"]([1, 1]))
    print("Relative likelihood of (1,1):", c_plot["rl"]([1, 1]))
    print("Mass above (1,1):", c_plot["m_above"]([1, 1]))
    print("Relative rank of (1,1):", c_plot["rr"]([1, 1]))
    print("Plateau mapping of (1,1):", c_plot["point_p"]([1, 1]))
    print("Local contribution of (2.5, 3.5) in rearranged space:", c_plot["local_contribution"](2.5, 3.5))
    print("Local contribution of region ((0, 1), (0, 1)) in original space:", c_plot["local_contribution_y"](((0, 1), (0, 1))))
    print("Least non-zero y location in rearranged space:", c_plot["min_nonzero"]())
    print("Support in region ((0, 1), (0, 1)) in original space:", c_plot["r_support"](((0, 1), (0, 1))))

    # Example configure
    # c_plot["configure_plot"](compress_y=True, mass_bins=3, zoom_y=0.0)

    # Example overlay calls
    # c_plot["mean"](ml_visualize=True)
    # c_plot["point"]([1, 1], ml_visualize=True)
    # c_plot["local_contribution_y"](((1, 4), (1, 4)), ml_visualize=True)
    # c_plot["plot_overlays"]()
