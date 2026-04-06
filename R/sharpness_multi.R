# ============================================================
# Predictive Sharpness for Multidimensional Cases
# Midpoint Discretization + Array Preparation + Visualizations
# ============================================================
#
# This script implements the continuous sharpness measure S(f)
# from the paper:
#   "A Measure of Predictive Sharpness for Probabilistic Models"
#
# R translation of the Python reference implementation.
# Implemented on 6.4.2026 by AI (ChatGPT commercial) and checked manually for accuracy.
# Visualizations not cleaned up yet as in the Python version.
#
# It supports multidimensional use, midpoint discretization of
# callable PDFs, preparation of array-based PDFs, and visualization,
# using the three equivalent sharpness formulas:
#
#   1) Simplified
#   2) Mass–Length (ML)
#   3) Gini-style
#
# ------------------------------------------------------------
# WHAT THIS GIVES YOU
# ------------------------------------------------------------
#
# 1) Midpoint-based discretization of callable PDFs:
#
#    midpoint_discretize(pdf, bounds, bins,
#                        normalize = FALSE,
#                        return_coords = FALSE)
#
#    - Evaluates one or more callable PDFs on midpoint grids
#    - Returns flattened density arrays regardless of dimension
#    - Can optionally normalize so the discretized PDF integrates to 1
#    - Can optionally return midpoint coordinate arrays
#
#
# 2) Preparation of existing array-based PDFs:
#
#    prepare_array_pdf(array_pdf, coords,
#                      bounds = NULL,
#                      normalize = TRUE,
#                      return_coords = FALSE)
#
#    - Cleans and prepares one or more discretized PDF arrays
#    - Works with uniform grids in physical coordinates
#    - Pads with zeros if the target domain is larger
#    - Crops if the target domain is smaller
#    - Reshapes flat arrays when coordinates imply a multidimensional grid
#    - Can normalize to integrate to 1
#
#
# 3) Sharpness calculation from discretized density values:
#
#    sharpness_multi(dvals,
#                    mode = "simplified" | "ml" | "gini",
#                    plot_data = FALSE)
#
#    - Computes S(f) from density values over a bounded domain
#    - Returns a value in [0, 1]
#    - Optionally returns arrays needed for plotting
#
#
# 4) Visualizations of the three formulations:
#
#    visualize_sharpness(pdfs, titles = NULL,
#                        mode = "simplified" | "ml" | "gini" | "cplot",
#                        show_fractional = TRUE,
#                        mass_bins = 4,
#                        zoom_y = 0.0)
#
#    - simplified: plots t * f^(uparrow)(t)
#    - ml: plots m(t) and f^(uparrow)(t) * L(t)
#    - gini: plots Lorenz-style cumulative mass curves
#    - cplot: plots the concentration-plot version of the ML view
#
#
# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
#
# All sharpness scores are normalized to [0, 1]:
#
#   0   -> maximally diffuse (uniform distribution over the domain)
#   1   -> maximally sharp (degenerate / Dirac-like prediction)
#
#
# ------------------------------------------------------------
# ASSUMPTIONS / REQUIREMENTS
# ------------------------------------------------------------
#
# For density arrays used in sharpness calculations:
#
#   - Values should be finite or cleanable into finite values
#   - Negative values are clipped to zero
#   - Arrays should correspond to a uniform grid
#   - Densities should integrate approximately to 1 over the domain
#     (or use normalize = TRUE)
#
# For midpoint_discretize():
#
#   - pdf must be a function or list of functions
#   - bounds must be given as a list or matrix of (lower, upper) pairs
#   - bins must be either:
#       * a single integer, or
#       * one integer per dimension
#
# For prepare_array_pdf():
#
#   - coords must match the array dimensions
#   - bounds, if supplied, define the target domain
#   - input arrays are assumed to lie on a uniform grid
#
#
# ------------------------------------------------------------
# CLEANING / ROBUSTNESS BEHAVIOR
# ------------------------------------------------------------
#
# The script automatically handles problematic values:
#
#   - NaN      -> 0
#   - -Inf     -> 0
#   - +Inf     -> large finite spike
#   - negative -> clipped to 0
#
# If normalization is requested and the total mass is zero over the
# requested domain, the script stops with an error.
#
#
# ------------------------------------------------------------
# PRACTICAL TIPS
# ------------------------------------------------------------
#
# - Use mode = "simplified" for fastest computation
# - Use "ml" or "gini" when you want interpretable plotting outputs
# - Higher bin counts improve approximation accuracy
# - In 1D, bins = 10000 is often a good default
# - For dimensions > 4, full tensor grids may become expensive;
#   Monte Carlo or quasi-Monte Carlo evaluation is more practical
#
#
# ------------------------------------------------------------
# TYPICAL WORKFLOWS
# ------------------------------------------------------------
#
# 1) Callable PDFs -> discretization -> sharpness:
#
#    bounds <- list(c(0, 4))
#    bins <- 10000
#    pdfs <- midpoint_discretize(my_pdfs, bounds, bins, normalize = TRUE)
#    sharpness_multi(pdfs[[1]], mode = "simplified")
#
#
# 2) Existing discretized PDF array -> sharpness:
#
#    sharpness_multi(my_pdf_array, mode = "simplified")
#
#
# 3) Existing array -> domain adjustment / cleaning / normalization:
#
#    pdf_prepared <- prepare_array_pdf(my_pdf_array, coords,
#                                      bounds = list(c(0, 7)),
#                                      normalize = TRUE)
#
#
# 4) Discretized PDFs -> visualization:
#
#    visualize_sharpness(my_pdfs, titles, mode = "gini")
#
#
# ------------------------------------------------------------
# NOTE
# ------------------------------------------------------------
#
# This script focuses on the continuous multidimensional sharpness
# workflow. For a simpler discrete + continuous-1D core implementation,
# use the companion core formulas script instead.
#
# ============================================================

.clean_density_values <- function(x) {
  x <- as.numeric(x)

  pos_inf <- is.infinite(x) & x > 0
  neg_inf <- is.infinite(x) & x < 0
  bad_na  <- is.na(x) | is.nan(x)

  if (any(pos_inf)) {
    finite_vals <- x[is.finite(x)]
    replacement <- if (length(finite_vals) > 0) max(finite_vals) * 1e6 else 1e6
    x[pos_inf] <- replacement
  }

  x[neg_inf | bad_na] <- 0
  x[x < 0] <- 0
  x
}

.is_coord_set <- function(x) {
  is.list(x) && length(x) > 0 &&
    all(vapply(x, function(el) is.numeric(el) || is.integer(el), logical(1)))
}

.as_bounds_matrix <- function(bounds) {
  if (is.matrix(bounds)) {
    out <- bounds
  } else if (is.list(bounds)) {
    out <- do.call(rbind, lapply(bounds, function(z) as.numeric(z)))
  } else {
    stop("bounds must be a matrix or list of (lower, upper) pairs")
  }
  storage.mode(out) <- "double"
  if (ncol(out) != 2) stop("bounds must have exactly 2 columns")
  out
}

.eval_pdf <- function(f, pts, dims) {
  if (dims == 1) {
    x <- as.numeric(pts)
    res <- tryCatch(
      f(x),
      error = function(e1) {
        tryCatch(
          vapply(x, function(z) f(z), numeric(1)),
          error = function(e2) rep(0, length(x))
        )
      }
    )
    if (length(res) == 1L && length(x) > 1L) res <- rep(res, length(x))
    return(as.numeric(res))
  }

  res <- tryCatch(
    f(pts),
    error = function(e1) {
      tryCatch(
        do.call(f, as.data.frame(pts)),
        error = function(e2) {
          tryCatch(
            apply(pts, 1, function(row) f(row)),
            error = function(e3) rep(0, nrow(pts))
          )
        }
      )
    }
  )

  if (length(res) == 1L && nrow(pts) > 1L) res <- rep(res, nrow(pts))
  as.numeric(res)
}

midpoint_discretize <- function(pdf, bounds, bins, normalize = FALSE, return_coords = FALSE) {
  bounds <- .as_bounds_matrix(bounds)
  dims <- nrow(bounds)

  if (length(bins) == 1L) bins <- rep(as.integer(bins), dims)
  bins <- as.integer(bins)
  if (length(bins) != dims) stop("bins must be length 1 or match the number of dimensions")

  widths <- (bounds[, 2] - bounds[, 1]) / bins
  coords <- lapply(seq_len(dims), function(i) {
    bounds[i, 1] + ((seq_len(bins[i]) - 0.5) * widths[i])
  })

  pts <- if (dims == 1L) {
    coords[[1]]
  } else {
    as.matrix(expand.grid(coords, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE))
  }

  if (is.function(pdf)) {
    pdfs <- list(pdf)
    single <- TRUE
  } else if (is.list(pdf) && all(vapply(pdf, is.function, logical(1)))) {
    pdfs <- pdf
    single <- FALSE
  } else {
    stop("pdf must be a function or a list of functions")
  }

  results <- vector("list", length(pdfs))

  for (i in seq_along(pdfs)) {
    dvals <- tryCatch(.eval_pdf(pdfs[[i]], pts, dims), error = function(e) {
      rep(0, if (dims == 1L) length(pts) else nrow(pts))
    })

    dvals <- .clean_density_values(dvals)

    if (normalize) {
      cell_volume <- prod(widths)
      total_mass <- sum(dvals) * cell_volume
      if (!is.finite(total_mass) || total_mass <= 0) {
        stop("PDF has zero total mass over the given bounds.")
      }
      dvals <- dvals / total_mass
    }

    results[[i]] <- as.numeric(dvals)
  }

  if (single) {
    if (return_coords) return(list(dvals = results[[1]], coords = coords))
    return(results[[1]])
  }

  if (return_coords) list(dvals = results, coords = coords) else results
}

prepare_array_pdf <- function(array_pdf, coords, bounds = NULL, normalize = TRUE, return_coords = FALSE) {
  if (!is.list(coords)) coords <- list(coords)

  multiple <- is.list(array_pdf) && !is.data.frame(array_pdf)

  if (multiple) {
    pdfs <- array_pdf
    coords_list <- if (.is_coord_set(coords)) rep(list(coords), length(pdfs)) else coords
  } else {
    pdfs <- list(array_pdf)
    coords_list <- list(coords)
  }

  if (length(coords_list) != length(pdfs)) {
    stop("coords must be a single coordinate set or a list with one coordinate set per PDF")
  }

  results <- vector("list", length(pdfs))
  coords_results <- vector("list", length(pdfs))

  for (k in seq_along(pdfs)) {
    arr <- pdfs[[k]]
    coord_set <- coords_list[[k]]
    if (!is.list(coord_set)) coord_set <- list(coord_set)

    arr <- as.numeric(arr)
    arr_dim <- dim(pdfs[[k]])

    if (is.null(arr_dim)) {
      if (length(coord_set) > 1L) {
        expected_shape <- vapply(coord_set, length, integer(1))
        if (length(arr) == prod(expected_shape)) {
          arr <- array(arr, dim = expected_shape)
          arr_dim <- dim(arr)
        } else {
          stop("For multidimensional coords, a 1D array_pdf must have size equal to prod(lengths(coords)).")
        }
      }
    } else {
      arr <- array(arr, dim = arr_dim)
    }

    dims <- if (is.null(dim(arr))) 1L else length(dim(arr))
    if (length(coord_set) != dims) {
      stop(sprintf("coords must have %d arrays, got %d", dims, length(coord_set)))
    }

    coord_lengths <- vapply(coord_set, length, integer(1))
    arr_shape <- if (dims == 1L) length(arr) else dim(arr)

    if (any(coord_lengths != arr_shape)) {
      stop("Coordinate lengths must match array dimensions.")
    }

    widths <- vapply(coord_set, function(cvec) {
      if (length(cvec) > 1L) (cvec[length(cvec)] - cvec[1]) / (length(cvec) - 1) else 1.0
    }, numeric(1))

    coord_starts <- vapply(coord_set, function(cvec) cvec[1], numeric(1))
    coord_ends   <- vapply(coord_set, function(cvec) cvec[length(cvec)], numeric(1))
    coord_bins   <- coord_lengths

    if (is.null(bounds)) {
      bounds_arr <- cbind(coord_starts, coord_ends)
      target_bins <- coord_bins
      same_grid <- TRUE
    } else {
      bounds_arr <- .as_bounds_matrix(bounds)
      same_grid <- isTRUE(all.equal(bounds_arr[, 1], coord_starts, tolerance = 1e-8)) &&
        isTRUE(all.equal(bounds_arr[, 2], coord_ends, tolerance = 1e-8))
      target_bins <- if (same_grid) {
        coord_bins
      } else {
        pmax(1L, as.integer(round((bounds_arr[, 2] - bounds_arr[, 1]) / widths)))
      }
    }

    target_coords <- if (same_grid) {
      coord_set
    } else {
      lapply(seq_len(dims), function(i) {
        bounds_arr[i, 1] + ((seq_len(target_bins[i]) - 0.5) * widths[i])
      })
    }

    dvals <- if (dims == 1L) numeric(target_bins[1]) else array(0, dim = target_bins)

    orig_idx <- vector("list", dims)
    tgt_idx  <- vector("list", dims)
    has_overlap <- TRUE

    for (i in seq_len(dims)) {
      o_start0 <- ceiling((target_coords[[i]][1] - coord_set[[i]][1]) / widths[i])
      o_end0   <- o_start0 + target_bins[i]
      t_start0 <- 0L
      t_end0   <- target_bins[i]

      if (o_start0 < 0) {
        t_start0 <- -o_start0
        o_start0 <- 0L
      }
      if (o_end0 > coord_bins[i]) {
        t_end0 <- t_end0 - (o_end0 - coord_bins[i])
        o_end0 <- coord_bins[i]
      }

      if (t_end0 <= t_start0 || o_end0 <= o_start0) {
        has_overlap <- FALSE
        break
      }

      orig_idx[[i]] <- (o_start0 + 1L):o_end0
      tgt_idx[[i]]  <- (t_start0 + 1L):t_end0
    }

    if (has_overlap) {
      if (dims == 1L) {
        dvals[tgt_idx[[1]]] <- arr[orig_idx[[1]]]
      } else {
        sub_arr <- do.call("[", c(list(arr), orig_idx, list(drop = FALSE)))
        dvals <- do.call("[<-", c(list(dvals), tgt_idx, list(value = sub_arr)))
      }
    }

    dvals <- .clean_density_values(dvals)

    if (normalize) {
      cell_volume <- prod(widths)
      total_mass <- sum(dvals) * cell_volume
      if (!is.finite(total_mass) || total_mass <= 0) {
        stop("PDF has zero mass over the given bounds.")
      }
      dvals <- dvals / total_mass
    }

    results[[k]] <- as.numeric(dvals)
    coords_results[[k]] <- target_coords
  }

  if (multiple) {
    if (return_coords) return(list(dvals = results, coords = coords_results))
    return(results)
  }

  if (return_coords) list(dvals = results[[1]], coords = coords_results[[1]]) else results[[1]]
}

sharpness_multi <- function(dvals, mode = c("simplified", "ml", "gini"), plot_data = FALSE) {
  mode <- match.arg(mode)
  dvals <- as.numeric(dvals)
  dvals <- dvals[is.finite(dvals)]
  N <- length(dvals)

  if (N == 0L) stop("dvals must contain at least one finite value")
  if (mean(dvals) <= 0) stop("dvals must have positive mean")

  L <- 1 / mean(dvals)
  v <- L / N
  d_sorted <- sort(dvals)

  if (mode == "simplified") {
    weights <- (seq_len(N) - 0.5)
    t <- weights * v
    integral <- v * sum(d_sorted * t)
    score <- (2 / L) * integral - 1
    score <- max(0, min(1, score))

    if (plot_data) {
      return(list(score = score, t = t, y = t * d_sorted))
    }
    return(score)
  }

  if (mode == "ml") {
    idx <- seq_len(N) - 1
    t <- idx * v
    m <- rev(cumsum(rev(d_sorted))) * v
    dL <- d_sorted * (L - t)
    score <- sum(m[-N] - dL[-N]) / N
    score <- max(0, min(1, score))

    if (plot_data) {
      return(list(score = score, t = t, m = m, dL = dL))
    }
    return(score)
  }

  cum_mass <- c(0, cumsum(d_sorted) * v)
  lorenz_area <- sum((cum_mass[-length(cum_mass)] + cum_mass[-1]) / 2) * (1 / N)
  score <- 1 - 2 * lorenz_area
  score <- max(0, min(1, score))

  if (plot_data) {
    u <- seq(0, 1, length.out = N + 1)
    return(list(score = score, u = u, cum_mass = cum_mass))
  }
  score
}

.concentration_plot_data <- function(dvals, mass_bins = 4L) {
  dvals <- as.numeric(dvals)
  N <- length(dvals)
  L <- 1 / mean(dvals)
  v <- L / N

  ml <- sharpness_multi(dvals, mode = "ml", plot_data = TRUE)
  score <- ml$score
  t_left <- ml$t
  m <- ml$m
  dL <- ml$dL

  q <- numeric(length(m))
  if (N > 1L) q[-N] <- (m[-N] - m[-1]) / v
  q[N] <- m[N] / v

  t_mid <- t_left + 0.5 * v
  delta <- m - dL
  width <- 1 - delta

  cum_mass <- cumsum(q) * v
  mass_edges <- seq(0, 1, length.out = mass_bins + 1L)
  t_edges <- approx(
    x = c(0, cum_mass),
    y = seq(0, L, length.out = N + 1L),
    xout = mass_edges,
    ties = "ordered",
    rule = 2
  )$y

  idx <- unique(round(seq(1, N, length.out = min(4500L, N))))

  list(
    t = t_mid[idx],
    width = width[idx],
    t_edges = t_edges,
    sharpness = as.numeric(score),
    L = as.numeric(L)
  )
}

visualize_sharpness <- function(
  pdfs,
  titles = NULL,
  mode = c("gini", "simplified", "ml", "cplot"),
  show_fractional = TRUE,
  mass_bins = 4,
  zoom_y = 0.0
) {
  mode <- match.arg(mode)

  if (!is.list(pdfs)) pdfs <- list(pdfs)
  n <- length(pdfs)

  if (is.null(titles)) {
    titles <- sprintf("PDF %d", seq_len(n))
  } else {
    titles <- as.character(titles)
    if (length(titles) < n) {
      titles <- c(titles, sprintf("PDF %d", (length(titles) + 1):n))
    }
  }

  if (mode == "simplified") {
    plot_list <- lapply(pdfs, function(pdf) sharpness_multi(pdf, mode = "simplified", plot_data = TRUE))
    xlim <- range(unlist(lapply(plot_list, `[[`, "t")))
    ylim <- range(unlist(lapply(plot_list, `[[`, "y")))

    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)

    plot(NA, xlim = xlim, ylim = ylim, xlab = "t", ylab = expression(t %.% f^"\u2191" * "(" * t * ")"),
         main = "Integrands for PDFs")

    for (i in seq_along(plot_list)) {
      lines(plot_list[[i]]$t, plot_list[[i]]$y, col = i, lwd = 2)
    }

    legend_labels <- sprintf("%s, S = %.3f", titles, vapply(plot_list, `[[`, numeric(1), "score"))
    legend("topright", legend = legend_labels, col = seq_along(plot_list), lwd = 2, bty = "n")
    grid()
    return(invisible(NULL))
  }

  if (mode == "ml") {
    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)
    par(mfrow = c(n, 1), mar = c(4, 4, 3, 1))

    for (i in seq_along(pdfs)) {
      out <- sharpness_multi(pdfs[[i]], mode = "ml", plot_data = TRUE)
      yr <- range(c(out$m, out$dL))
      plot(out$t, out$m, type = "l", lwd = 2, col = "orange",
           xlab = "t", ylab = "Integrand value",
           main = sprintf("%s (S = %.3f)", titles[i], out$score), ylim = yr)
      lines(out$t, out$dL, lwd = 2, col = "tomato")
      legend("topright",
           legend = c(
             expression(m(t)),
             expression(f^"\u2191" * "(" * t * ")" %.% L(t))
           ),
           col = c("orange", "tomato"), lwd = 2, bty = "n")
      grid()
    }
    return(invisible(NULL))
  }

  if (mode == "gini") {
    plot_list <- lapply(pdfs, function(pdf) sharpness_multi(pdf, mode = "gini", plot_data = TRUE))

    op <- par(no.readonly = TRUE)
    on.exit(par(op), add = TRUE)

    plot(c(0, 1), c(0, 1), type = "l", lwd = 1,
         xlab = "Fraction of domain (u)", ylab = "Cumulative probability mass",
         main = "Gini-style curves for probability density functions")
    abline(0, 1, lty = 1)

    for (i in seq_along(plot_list)) {
      lines(plot_list[[i]]$u, plot_list[[i]]$cum_mass, col = i, lwd = 2)
    }

    legend_labels <- sprintf("%s, S = %.3f", titles, vapply(plot_list, `[[`, numeric(1), "score"))
    legend("topleft", legend = c("Uniform baseline", legend_labels),
           col = c("black", seq_along(plot_list)), lwd = c(1, rep(2, length(plot_list))), bty = "n")
    grid()
    return(invisible(NULL))
  }

  if (!is.numeric(mass_bins) || length(mass_bins) != 1L || mass_bins < 1 || mass_bins > 10 || mass_bins %% 1 != 0) {
    stop("mass_bins must be an integer between 1 and 10.")
  }
  mass_bins <- as.integer(mass_bins)

  if (zoom_y != 0.0 && (zoom_y < 1e-6 || zoom_y > 0.999999)) {
    stop("zoom_y must be 0.0 (no zoom) or between 0.000001 and 0.999999.")
  }

  data_list <- lapply(pdfs, .concentration_plot_data, mass_bins = mass_bins)
  ncols <- min(3L, n)
  nrows <- ceiling(n / 3)

  cols <- grDevices::palette.colors(max(3L, mass_bins), palette = "Okabe-Ito")[seq_len(mass_bins)]
  fill_cols <- vapply(cols, grDevices::adjustcolor, character(1), alpha.f = 0.72)
  legend_labels <- sprintf("%d–%d%% mass",
                           round(100 * (0:(mass_bins - 1)) / mass_bins),
                           round(100 * (1:mass_bins) / mass_bins))

  op <- par(no.readonly = TRUE)
  on.exit(par(op), add = TRUE)
  par(mfrow = c(nrows, ncols), mar = c(4, 3.5, 3, 1), oma = c(0, 0, 0, 0))

  for (k in seq_len(nrows * ncols)) {
    if (k > n) {
      plot.new()
      next
    }

    data <- data_list[[k]]
    t <- data$t
    width <- data$width
    t_edges <- data$t_edges
    L <- data$L

    frac <- t / L
    frac_edges <- t_edges / L
    col_idx <- ((k - 1L) %% ncols) + 1L

    if (zoom_y == 0.0) {
      if (show_fractional) {
        y <- frac
        y_edges <- frac_edges
        y_max <- 1.0
        ylab <- "Fractions of rearranged domain (u)"
        axis_ticks <- pretty(c(0, y_max), n = 5)
        axis_labels <- format(axis_ticks, trim = TRUE)
      } else {
        y <- t
        y_edges <- t_edges
        y_max <- L
        ylab <- "Rearranged domain (t)"
        axis_ticks <- pretty(c(0, y_max), n = 5)
        axis_labels <- format(axis_ticks, digits = 4, trim = TRUE)
      }

      plot(NA, xlim = c(-1.05, 1.05), ylim = c(0, y_max), xlab = expression(1-Delta(t)),
           ylab = if (col_idx == 1L) ylab else "", xaxt = "n", yaxt = "n", bty = "n",
           main = sprintf("%s\nS = %.3f", titles[k], data$sharpness))

      for (i in seq_len(mass_bins)) {
        mask <- y >= y_edges[i] & y <= y_edges[i + 1L]
        if (sum(mask) >= 2L) {
          polygon(c(-width[mask], rev(width[mask])), c(y[mask], rev(y[mask])),
                  col = fill_cols[i], border = NA)
        }
      }

      lines(-width, y, lwd = 1.8)
      lines(width, y, lwd = 1.8)

      for (ye in y_edges[-c(1, length(y_edges))]) {
        w_ye <- approx(y, width, xout = ye, rule = 2)$y
        segments(-0.99 * w_ye, ye, 0.99 * w_ye, ye, col = "white", lwd = 1.3)
      }

      axis(1, at = c(-0.5, 0.5), labels = c("0.5", "0.5"), cex.axis = 0.8)
      if (col_idx == 1L) axis(2, at = axis_ticks, labels = axis_labels, las = 1, cex.axis = 0.8)

    } else {
      visible_mask <- frac >= zoom_y
      if (!any(visible_mask)) {
        plot.new()
        next
      }

      frac_vis <- frac[visible_mask]
      width_vis <- width[visible_mask]
      y_plot <- (frac_vis - zoom_y) / (1.0 - zoom_y)

      plot(NA, xlim = c(-1.05, 1.05), ylim = c(0, 1), xlab = expression(1-Delta(t)),
           ylab = if (col_idx == 1L) if (show_fractional) "Fractions of rearranged domain (u)" else "Rearranged domain (t)" else "",
           xaxt = "n", yaxt = "n", bty = "n",
           main = sprintf("%s\nS = %.3f, zoom from %.2f", titles[k], data$sharpness, zoom_y))

      for (i in seq_len(mass_bins)) {
        edge_lo <- frac_edges[i]
        edge_hi <- frac_edges[i + 1L]
        if (edge_hi <= zoom_y) next

        mask <- frac >= max(edge_lo, zoom_y) & frac <= edge_hi
        if (sum(mask) >= 2L) {
          y_bin <- (frac[mask] - zoom_y) / (1.0 - zoom_y)
          w_bin <- width[mask]
          polygon(c(-w_bin, rev(w_bin)), c(y_bin, rev(y_bin)),
                  col = fill_cols[i], border = NA)
        }
      }

      lines(-width_vis, y_plot, lwd = 1.8)
      lines(width_vis, y_plot, lwd = 1.8)

      for (fe in frac_edges[-c(1, length(frac_edges))]) {
        if (fe <= zoom_y) next
        ye_plot <- (fe - zoom_y) / (1.0 - zoom_y)
        w_fe <- approx(frac, width, xout = fe, rule = 2)$y
        segments(-0.99 * w_fe, ye_plot, 0.99 * w_fe, ye_plot, col = "white", lwd = 1.3)
      }

      axis(1, at = c(-0.5, 0.5), labels = c("0.5", "0.5"), cex.axis = 0.8)
      tick_pos <- seq(0, 1, length.out = 6)
      frac_tick_vals <- zoom_y + tick_pos * (1.0 - zoom_y)
      tick_labels <- if (show_fractional) {
        format(round(frac_tick_vals, 3), nsmall = 3)
      } else {
        format(round(frac_tick_vals * L, 3), nsmall = 3)
      }
      if (col_idx == 1L) axis(2, at = tick_pos, labels = tick_labels, las = 1, cex.axis = 0.8)
    }

    if (k == 1L) {
      legend("topright", legend = legend_labels, fill = fill_cols, title = "Mass bins",
             cex = 0.8, bty = "o")
    }
  }

  invisible(NULL)
}

# ---- example ----
#
#bounds <- list(c(0, 4))
#bins <- 10000
#
#pdf_funcs <- list(
#  function(x) 0.6 * dnorm(x, mean = 1.2, sd = 0.3) +
#              0.4 * dnorm(x, mean = 3.0, sd = 0.4),
#  function(x) dnorm(x, mean = 2.8, sd = 0.5),
#  function(x) dnorm(x, mean = 2.8, sd = 0.1)
#)
#
#pdfs <- midpoint_discretize(pdf_funcs, bounds, bins, normalize = TRUE)
#
#titles <- c(
#  "0.6*N(1.2, 0.3^2) + 0.4*N(3.0, 0.4^2)",
#  "N(2.8, 0.5^2)",
#  "N(2.8, 0.1^2)"
#)
#
#for (i in seq_along(pdfs)) {
#  cat(sprintf("PDF %d: %.3f\n", i, sharpness_multi(pdfs[[i]], mode = "simplified")))
#}
#
#visualize_sharpness(pdfs, titles, mode = "simplified")
#visualize_sharpness(pdfs, titles, mode = "gini")
#visualize_sharpness(pdfs, titles, mode = "cplot", show_fractional = TRUE, mass_bins = 4, zoom_y = 0)
#visualize_sharpness(pdfs, titles, mode = "ml")
