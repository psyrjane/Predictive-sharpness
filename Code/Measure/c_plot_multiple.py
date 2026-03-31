""""
Predictive Sharpness - Local Concentration y for multiple plots (Concentration Plot Version)
--------------------------------------------------------------------------
Implements local_contribution_y() from c_plot.py for multiple pdfs simultaneously.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

def plot_local_contribution_y_grid(
    pdfs,
    region_bounds,
    bounds=None,
    bins=None,
    normalize=False,
    compress_y=True,
    mass_bins=5,
    show_region_arrows=False,
    per_row=3,
    titles=None,
    figsize_per_panel=(4.8, 6.0),
    return_results=False,
    zoom_y=0.0,
):
    """
    Plot local_contribution_y overlays for one or many PDFs / (dvals, coords) tuples.
    
    NOTE: Requires sharpness_multi.py !

    Parameters
    ----------
    pdfs : callable, tuple, or list
        A single callable PDF, a single (dvals, coords) tuple, or a list of them.
        List elements may also be ("title", pdf_obj) pairs.
    region_bounds : tuple of tuples
        Region in the original domain, e.g. ((x_min, x_max), (y_min, y_max), ...).
    bounds : list of tuples, optional
        Required for callable PDFs.
    bins : int or list, optional
        Required for callable PDFs.
    normalize : bool, optional
        Normalize callable/tuple input to total mass 1.
    compress_y : bool, optional
        If True, plot y-axis as fractions of the rearranged domain [0,1].
    mass_bins : int, optional
        Number of cumulative-mass bins, must be between 1 and 10.
    show_region_arrows : bool, optional
        Kept for backward compatibility; not used in this version.
    per_row : int, optional
        Number of panels per row.
    titles : list of str, optional
        Optional titles, same length as the number of PDFs.
    figsize_per_panel : tuple, optional
        Size per panel as (width, height).
    return_results : bool, optional
        If True, return a list of per-panel result dictionaries.
    zoom_y : float, default=0.0
        Fraction of the rearranged domain from which to begin visualization.
        If 0.0, no zoom is applied. If nonzero, must be between
        0.000001 and 0.999999. The visible portion is stretched to the
        full plot height.

    Returns
    -------
    list of dict, optional
        Returned only if return_results=True.
    """

    if not (1 <= int(mass_bins) <= 10):
        raise ValueError("mass_bins must be an integer between 1 and 10.")
    if per_row < 1:
        raise ValueError("per_row must be at least 1.")
    if zoom_y != 0.0:
        if not (0.000001 <= zoom_y <= 0.999999):
            raise ValueError(
                "zoom must be 0.0 (no zoom) or between 0.000001 and 0.999999."
            )

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

    def _lighten_color(color, frac=0.82):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple((1.0 - frac) * rgb + frac * np.ones(3))

    def _resample_for_plot(t, w, max_points=2000):
        t = np.asarray(t)
        w = np.asarray(w)
        if len(t) <= max_points:
            return t, w
        t_new = np.linspace(t[0], t[-1], max_points)
        w_new = np.interp(t_new, t, w)
        return t_new, w_new

    def _normalize_input_list(pdfs, titles):
        if callable(pdfs) or (
            isinstance(pdfs, tuple)
            and len(pdfs) == 2
            and not isinstance(pdfs[0], str)
        ):
            items = [pdfs]
        elif isinstance(pdfs, (list, tuple)):
            items = list(pdfs)
        else:
            raise TypeError("pdfs must be a callable, a (dvals, coords) tuple, or a list of them.")

        normalized_items = []
        for k, item in enumerate(items):
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                title, pdf_obj = item
            else:
                title = None
                pdf_obj = item
            normalized_items.append((title, pdf_obj))

        if titles is not None:
            if len(titles) != len(normalized_items):
                raise ValueError("titles must have the same length as the number of PDFs.")
            normalized_items = [(titles[i], normalized_items[i][1]) for i in range(len(normalized_items))]
        else:
            normalized_items = [
                (title if title is not None else f"PDF {i+1}", pdf_obj)
                for i, (title, pdf_obj) in enumerate(normalized_items)
            ]

        return normalized_items

    def _prepare_pdf(pdf_obj):
        if callable(pdf_obj):
            if bounds is None or bins is None:
                raise ValueError("bounds and bins are required for callable PDFs.")

            bounds_arr = np.array(bounds, dtype=float)
            dims = bounds_arr.shape[0]

            if np.isscalar(bins):
                bins_arr = np.array([int(bins)] * dims, dtype=int)
            else:
                bins_arr = np.array(bins, dtype=int)

            widths = (bounds_arr[:, 1] - bounds_arr[:, 0]) / bins_arr
            coords = [
                bounds_arr[i, 0] + (np.arange(bins_arr[i]) + 0.5) * widths[i]
                for i in range(dims)
            ]

            dvals = midpoint_discretize(pdf_obj, bounds, bins, normalize=False)
            dvals = np.asarray(dvals, float).ravel()

            coords = [np.asarray(c, float) for c in coords]
            domain_measure = float(np.prod(bounds_arr[:, 1] - bounds_arr[:, 0]))
            cell_volume = float(np.prod(widths))

            if normalize:
                total_mass = np.sum(dvals) * cell_volume
                if total_mass <= 0:
                    raise ValueError("PDF has zero total mass over the given bounds.")
                dvals = dvals / total_mass

        elif isinstance(pdf_obj, tuple) and len(pdf_obj) == 2:
            dvals, coords = pdf_obj
            dvals = np.asarray(dvals, float).ravel()

            if not isinstance(coords, list):
                raise ValueError("coords must be a list of arrays for tuple input.")
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
            raise TypeError("Each PDF entry must be a callable or a (dvals, coords) tuple.")

        grids = np.meshgrid(*coords, indexing="ij")
        points = np.stack([g.ravel() for g in grids], axis=-1)

        dims = len(coords)
        if len(region_bounds) != dims:
            raise ValueError("region_bounds dimensionality does not match the PDF dimensionality.")

        N = dvals.size
        L = domain_measure if domain_measure > 0 else 1.0 / max(dvals.mean(), 1e-16)
        v = L / N

        d_sorted = np.sort(np.clip(dvals, 0, None))
        t_left = np.arange(N, dtype=float) * v
        t_mid = t_left + 0.5 * v

        m = np.cumsum(d_sorted[::-1])[::-1] * v
        dL = d_sorted * (L - t_left)
        delta = m - dL
        width = np.clip(1.0 - delta, 0.0, 1.0)
        score = float(np.sum(delta) / N)
        cum_mass_used = np.cumsum(d_sorted) * v

        mask = np.ones(len(points), dtype=bool)
        for i, (min_val, max_val) in enumerate(region_bounds):
            coord = coords[i]
            if not (
                np.min(coord) - 0.1 <= min_val <= np.max(coord) + 0.1
                and np.min(coord) - 0.1 <= max_val <= np.max(coord) + 0.1
            ):
                raise ValueError("Value(s) out of bounds")
            mask &= (points[:, i] >= min_val) & (points[:, i] <= max_val)

        region_indices = np.where(mask)[0]
        densities = dvals[region_indices]
        t_indices = np.searchsorted(d_sorted, densities, side="left")
        t_indices = np.clip(t_indices, 0, N - 1)

        contribution = float(np.sum(delta[t_indices]) / N)
        contribution_percentage = float((contribution / score) * 100) if score > 0 else 0.0

        all_t_indices = np.searchsorted(d_sorted, dvals, side="left")
        all_t_indices = np.clip(all_t_indices, 0, N - 1)
        all_cell_contrib = delta[all_t_indices] / N
        region_size = len(region_indices)

        if region_size > 0:
            max_contribution_same_size = float(np.sum(np.sort(all_cell_contrib)[-region_size:]))
        else:
            max_contribution_same_size = 0.0

        if max_contribution_same_size > 0:
            relative_to_max_same_size = contribution / max_contribution_same_size
        else:
            relative_to_max_same_size = 0.0

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

        mass_edges = np.linspace(0.0, 1.0, mass_bins + 1)
        t_edges = np.array([_mass_edge_t(p) for p in mass_edges], dtype=float)
        bin_colors = list(plt.get_cmap("tab10").colors)[:mass_bins]

        ratios = np.zeros(mass_bins, dtype=float)
        t_lows = np.full(mass_bins, np.nan, dtype=float)
        t_highs = np.full(mass_bins, np.nan, dtype=float)

        zero_starts = np.full(mass_bins, np.nan, dtype=float)
        zero_ends = np.full(mass_bins, np.nan, dtype=float)

        # Visualization-only t locations:
        # spread repeated selected densities across their plateau in rearranged space
        t_selected_vis = np.full(len(densities), np.nan, dtype=float)
        assigned = np.zeros(len(densities), dtype=bool)

        top_density = float(d_sorted[-1])

        for k in range(len(densities)):
            if assigned[k]:
                continue

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

        for i in range(mass_bins):
            if i < mass_bins - 1:
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

        return {
            "coords": coords,
            "points": points,
            "dvals": dvals,
            "L": L,
            "v": v,
            "N": N,
            "t_mid": t_mid,
            "width": width,
            "delta": delta,
            "score": score,
            "contribution": contribution,
            "contribution_percentage": contribution_percentage,
            "max_contribution_same_size": max_contribution_same_size,
            "relative_to_max_same_size": relative_to_max_same_size,
            "t_edges": t_edges,
            "ratios": np.clip(ratios, 0.0, 1.0),
            "t_lows": t_lows,
            "t_highs": t_highs,
            "zero_starts": zero_starts,
            "zero_ends": zero_ends,
            "colors": bin_colors,
        }

    items = _normalize_input_list(pdfs, titles)
    n = len(items)
    ncols = min(per_row, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    results = []

    for ax, (title, pdf_obj) in zip(axes, items):
        res = _prepare_pdf(pdf_obj)
        results.append({"title": title, **res})

        t_mid = res["t_mid"]
        width = res["width"]
        t_edges = res["t_edges"]
        ratios = res["ratios"]
        t_lows = res["t_lows"]
        t_highs = res["t_highs"]
        zero_starts = res["zero_starts"]
        zero_ends = res["zero_ends"]
        colors = res["colors"]
        L = res["L"]
        score = res["score"]

        frac = t_mid / L
        frac_edges = t_edges / L

        t_draw, width_draw = _resample_for_plot(t_mid, width, max_points=2000)

        if zoom_y == 0.0:
            y_draw = t_draw / L if compress_y else t_draw

            ax.plot(width_draw, y_draw, color="black", linewidth=2)
            ax.plot(-width_draw, y_draw, color="black", linewidth=2)

            # Base light mass bins
            for i in range(mass_bins):
                if i < mass_bins - 1:
                    mask_bin = (t_mid >= t_edges[i]) & (t_mid < t_edges[i + 1])
                else:
                    mask_bin = (t_mid >= t_edges[i]) & (t_mid <= t_edges[i + 1])

                if np.any(mask_bin):
                    t_bin = t_mid[mask_bin]
                    w_bin = width[mask_bin]
                    t_bin_draw, w_bin_draw = _resample_for_plot(t_bin, w_bin, max_points=500)

                    ax.fill_betweenx(
                        (t_bin_draw / L) if compress_y else t_bin_draw,
                        -w_bin_draw,
                        w_bin_draw,
                        color=_lighten_color(colors[i], frac=0.90),
                        alpha=1.0,
                        linewidth=0,
                    )

            # White separators
            for te in t_edges[1:-1]:
                w_te = np.interp(te, t_mid, width)
                y_te = te / L if compress_y else te
                ax.hlines(
                    y_te,
                    xmin=-0.99 * w_te,
                    xmax=0.99 * w_te,
                    color="white",
                    lw=1.2,
                    alpha=0.95,
                )

        else:
            frac_draw = t_draw / L
            visible_mask = frac_draw >= zoom_y
            if not np.any(visible_mask):
                ax.set_visible(False)
                continue

            frac_draw_vis = frac_draw[visible_mask]
            width_draw_vis = width_draw[visible_mask]
            y_draw = (frac_draw_vis - zoom_y) / (1.0 - zoom_y)

            ax.plot(width_draw_vis, y_draw, color="black", linewidth=2)
            ax.plot(-width_draw_vis, y_draw, color="black", linewidth=2)

            # Base light mass bins
            for i in range(mass_bins):
                edge_lo = frac_edges[i]
                edge_hi = frac_edges[i + 1]

                if edge_hi <= zoom_y:
                    continue

                mask_bin = (
                    (frac >= max(edge_lo, zoom_y)) &
                    (frac <= edge_hi)
                )

                if np.any(mask_bin):
                    frac_bin = frac[mask_bin]
                    w_bin = width[mask_bin]
                    y_bin = (frac_bin - zoom_y) / (1.0 - zoom_y)
                    y_bin_draw, w_bin_draw = _resample_for_plot(y_bin, w_bin, max_points=500)

                    ax.fill_betweenx(
                        y_bin_draw,
                        -w_bin_draw,
                        w_bin_draw,
                        color=_lighten_color(colors[i], frac=0.90),
                        alpha=1.0,
                        linewidth=0,
                    )

            # White separators
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
                    lw=1.2,
                    alpha=0.95,
                )

        # Overlay
        extra_handles = []
        for i in range(mass_bins):
            base_color = colors[i]

            # Full-width fill from t_min to t_max
            if np.isfinite(t_lows[i]) and np.isfinite(t_highs[i]) and (t_highs[i] > t_lows[i]):
                if zoom_y == 0.0:
                    mask_span = ((t_mid + 0.5 * res["v"]) > t_lows[i]) & ((t_mid - 0.5 * res["v"]) < t_highs[i])

                    if np.any(mask_span):
                        t_span = t_mid[mask_span]
                        w_span = width[mask_span]
                        t_span_draw, w_span_draw = _resample_for_plot(t_span, w_span, max_points=500)

                        ax.fill_betweenx(
                            (t_span_draw / L) if compress_y else t_span_draw,
                            -w_span_draw,
                            w_span_draw,
                            color=base_color,
                            alpha=0.95,
                            linewidth=0,
                        )
                else:
                    span_lo = t_lows[i] / L
                    span_hi = t_highs[i] / L

                    if span_hi > zoom_y:
                        mask_span = (
                            (frac >= max(span_lo, zoom_y)) &
                            (frac <= span_hi)
                        )

                        if np.any(mask_span):
                            frac_span = frac[mask_span]
                            w_span = width[mask_span]
                            y_span = (frac_span - zoom_y) / (1.0 - zoom_y)
                            y_span_draw, w_span_draw = _resample_for_plot(y_span, w_span, max_points=500)

                            ax.fill_betweenx(
                                y_span_draw,
                                -w_span_draw,
                                w_span_draw,
                                color=base_color,
                                alpha=0.95,
                                linewidth=0,
                            )

            # Zero-density plateau -> center bar
            if np.isfinite(zero_starts[i]) and np.isfinite(zero_ends[i]) and (zero_ends[i] > zero_starts[i]):
                if zoom_y == 0.0:
                    ax.plot(
                        [0.0, 0.0],
                        (np.array([zero_starts[i], zero_ends[i]]) / L) if compress_y else np.array([zero_starts[i], zero_ends[i]]),
                        color=base_color,
                        linewidth=2.6,
                        solid_capstyle="butt",
                    )
                else:
                    z0 = zero_starts[i] / L
                    z1 = zero_ends[i] / L
                    if z1 > zoom_y:
                        z0_clip = max(z0, zoom_y)
                        y_zero = (np.array([z0_clip, z1]) - zoom_y) / (1.0 - zoom_y)
                        ax.plot(
                            [0.0, 0.0],
                            y_zero,
                            color=base_color,
                            linewidth=2.6,
                            solid_capstyle="butt",
                        )

            mass_start = 100 * i / mass_bins
            mass_end = 100 * (i + 1) / mass_bins
            extra_handles.append(
                Patch(
                    facecolor=base_color,
                    edgecolor="none",
                    label=f"{mass_start:.0f}–{mass_end:.0f}% bin: {ratios[i]:.2f}",
                )
            )

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0, 1.0 if (compress_y or zoom_y != 0.0) else L)

        if zoom_y == 0.0:
            ax.set_title(
                f"{title}\nS(f) = {score:.3f}\nRegion/max same-size = {res['relative_to_max_same_size']:.3f} ({100 * res['relative_to_max_same_size']:.1f}%)",
                fontsize=10,
            )
        else:
            ax.set_title(
                f"{title}\nS(f) = {score:.3f}, zoom from {zoom_y:.2f}\nRegion/max same-size = {res['relative_to_max_same_size']:.3f} ({100 * res['relative_to_max_same_size']:.1f}%)",
                fontsize=10,
            )

        ax.spines["top"].set_visible(False)
        ax.grid(False)
        ax.set_xlabel(r"$1-\Delta(t)$", fontsize=9)
        ax.set_xticks([-0.5, 0.5])
        ax.set_xticklabels(["0.5", "0.5"], fontsize=8)

        if zoom_y != 0.0:
            tick_pos = np.linspace(0.0, 1.0, 6)
            frac_tick_vals = zoom_y + tick_pos * (1.0 - zoom_y)

            if compress_y:
                tick_labels = [f"{v:.3f}" for v in frac_tick_vals]
            else:
                tick_labels = [f"{(v * L):.3f}" for v in frac_tick_vals]

            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labels, fontsize=8)

        if ax is axes[0]:
            if compress_y:
                ax.set_ylabel("fractions of the rearranged domain (u)")
            else:
                ax.set_ylabel("rearranged domain (t)")
            ax.spines["right"].set_visible(False)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if extra_handles:
            ax.legend(handles=extra_handles, fontsize=8, loc="best")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.show()

    if return_results:
        return results



# EXAMPLE CALL
# pdf_specs = [
#    ("Slow + fast + fast + uniform", pdf_smooth_quadrants_2),
#    ("Excluded + uniform + slow + fast", pdf_smooth_quadrants_3),
#    ("Spiked Gaussian", pdf_spiked_gaussian),
#]
#
# plot_local_contribution_y_grid(
#    pdf_specs,
#    region_bounds=((0, 1), (1, 2), (0, 2)),
#    bounds=bounds8,
#    bins=60,
#    normalize=True,
#    compress_y=True,
#    mass_bins=5,
#    show_region_arrows=False,
#    per_row=3,
#    zoom_y=0.0,
# )
