# Predictive Sharpness Core Formulas (discrete + continuous 1D)
# R translation of the Python reference implementation.
# Implemented on 6.4.2026 by AI (ChatGPT commercial) and checked manually for accuracy.
#
# This script implements the core sharpness formulas from:
# "A Measure of Predictive Sharpness for Probabilistic Models".
#
# Included:
#   1) Discrete sharpness:
#      - s_discrete(p):    simplified discrete sharpness S(P)
#      - s_discrete_ml(p): mass-length discrete sharpness S(P)
#
#   2) Continuous 1D sharpness on [a, b]:
#      - sd(pdf, a, b, bins):      simplified continuous sharpness S(f)
#      - sd_ml(pdf, a, b, bins):   mass-length continuous sharpness S(f)
#      - sd_gini(pdf, a, b, bins): Gini-style continuous sharpness S(f)
#
#   3) Relative sharpness and domain transformations:
#      - s_rel(S1, S2)
#      - discrete_f(Sm, m, n), discrete_i(Sn, n, m)
#      - continuous_f(S_ell, ell, L), continuous_i(S_L, L, ell)

s_discrete <- function(p) {
  # Discrete sharpness S(P): simplified form.
  #
  # S(P) = sum_{j=1}^n [ (2j - n - 1) / (n - 1) ] * p_(j)
  # where p_(j) are the probabilities sorted in ascending order.

  if (is.list(p) && !is.null(names(p))) {
    p <- unname(unlist(p, use.names = FALSE))
  }
  p <- as.numeric(p)
  p <- c(p)

  n <- length(p)
  if (n < 2L) stop("p must contain at least 2 probabilities")

  p_sorted <- sort(p)
  j <- seq_len(n)
  w <- (2 * j - n - 1) / (n - 1)
  sum(w * p_sorted)
}

s_discrete_ml <- function(p) {
  # Discrete sharpness S(P): mass-length form.
  #
  # S(P) = (1/(n-1)) * sum_{j=1}^{n-1} [ m_(j) - p_(j) * L_(j) ]
  # with m_(j) = sum_{k=j}^n p_(k), L_(j) = n - j + 1.

  if (is.list(p) && !is.null(names(p))) {
    p <- unname(unlist(p, use.names = FALSE))
  }
  p <- as.numeric(p)
  p <- c(p)

  n <- length(p)
  if (n < 2L) stop("p must contain at least 2 probabilities")

  p_sorted <- sort(p)
  m <- rev(cumsum(rev(p_sorted)))
  num <- sum(m[-n]) - sum(p_sorted[-n] * seq(n, 2))
  num / (n - 1)
}

sd <- function(pdf, a, b, bins = 10000L) {
  # Continuous sharpness S(f): simplified form.
  #
  # S(f) = (2 / L) * integral_0^L t * f^uparrow(t) dt - 1
  # approximated by midpoint discretization on [a, b].

  a <- as.numeric(a)
  b <- as.numeric(b)
  bins <- as.integer(bins)

  if (!is.function(pdf)) stop("pdf must be a function")
  if (!(b > a)) stop("Require b > a")
  if (bins < 1L) stop("bins must be >= 1")

  L <- b - a
  w <- L / bins
  weights <- seq_len(bins) - 0.5
  x <- a + weights * w
  d_sorted <- sort(as.numeric(pdf(x)))
  t <- weights * w
  integral <- w * sum(d_sorted * t)
  (2 / L) * integral - 1
}

sd_ml <- function(pdf, a, b, bins = 10000L) {
  # Continuous sharpness S(f): mass-length form.
  #
  # S(f) = (1 / L) * integral_0^L [ m(t) - f^uparrow(t) * (L - t) ] dt
  # approximated by midpoint discretization on [a, b].

  a <- as.numeric(a)
  b <- as.numeric(b)
  bins <- as.integer(bins)

  if (!is.function(pdf)) stop("pdf must be a function")
  if (!(b > a)) stop("Require b > a")
  if (bins < 1L) stop("bins must be >= 1")

  L <- b - a
  w <- L / bins
  idx <- seq_len(bins) - 1
  t <- idx * w
  x <- a + (idx + 0.5) * w
  d_sorted <- sort(as.numeric(pdf(x)))
  m <- rev(cumsum(rev(d_sorted))) * w
  dL <- d_sorted * (L - t)
  sum((m - dL)[-bins]) / bins
}

sd_gini <- function(pdf, a, b, bins = 10000L) {
  # Continuous sharpness S(f): Gini-style / Lorenz-style form.
  #
  # S(f) = 1 - 2 * integral_0^1 L(u) du,
  # approximated by trapezoidal integration over cumulative mass.

  a <- as.numeric(a)
  b <- as.numeric(b)
  bins <- as.integer(bins)

  if (!is.function(pdf)) stop("pdf must be a function")
  if (!(b > a)) stop("Require b > a")
  if (bins < 1L) stop("bins must be >= 1")

  L <- b - a
  w <- L / bins
  x <- a + (seq_len(bins) - 0.5) * w
  d_sorted <- sort(as.numeric(pdf(x)))
  cum_mass <- c(0, cumsum(d_sorted) * w)
  lorenz <- sum((cum_mass[-length(cum_mass)] + cum_mass[-1]) / 2) / bins
  1 - 2 * lorenz
}

s_rel <- function(S1, S2, strict = TRUE) {
  # Relative sharpness change:
  #   (S2 - S1) / (1 - S1)

  S1 <- as.numeric(S1)
  S2 <- as.numeric(S2)

  if (strict && !(0 <= S1 && S1 < 1 && 0 <= S2 && S2 <= 1)) {
    stop("Require 0 <= S1 < 1 and 0 <= S2 <= 1")
  }
  if (S1 == 1) stop("Relative change undefined when S1 = 1")

  (S2 - S1) / (1 - S1)
}

discrete_f <- function(Sm, m, n, strict = TRUE) {
  # Forward discrete-domain transformation:
  #   S_n = 1 + ((m - 1)/(n - 1)) * (S_m - 1)

  Sm <- as.numeric(Sm)
  m <- as.integer(m)
  n <- as.integer(n)

  if (strict && (!(0 <= Sm && Sm <= 1) || m >= n)) {
    stop("Require 0 <= Sm <= 1 and m < n")
  }

  1 + ((m - 1) / (n - 1)) * (Sm - 1)
}

discrete_i <- function(Sn, n, m, strict = TRUE) {
  # Inverse discrete-domain transformation:
  #   S_m = 1 + ((n - 1)/(m - 1)) * (S_n - 1)

  Sn <- as.numeric(Sn)
  n <- as.integer(n)
  m <- as.integer(m)

  if (strict && (!(0 <= Sn && Sn <= 1) || n <= m)) {
    stop("Require 0 <= Sn <= 1 and n > m")
  }

  1 + ((n - 1) / (m - 1)) * (Sn - 1)
}

continuous_f <- function(S_ell, ell, L, strict = TRUE) {
  # Forward continuous-domain transformation:
  #   S_L = 1 + (ell / L) * (S_ell - 1)

  S_ell <- as.numeric(S_ell)
  ell <- as.numeric(ell)
  L <- as.numeric(L)

  if (strict && (!(0 <= S_ell && S_ell <= 1) || ell >= L)) {
    stop("Require 0 <= S_ell <= 1 and ell < L")
  }

  1 + (ell / L) * (S_ell - 1)
}

continuous_i <- function(S_L, L, ell, strict = TRUE) {
  # Inverse continuous-domain transformation:
  #   S_ell = 1 + (L / ell) * (S_L - 1)

  S_L <- as.numeric(S_L)
  L <- as.numeric(L)
  ell <- as.numeric(ell)

  if (strict && (!(0 <= S_L && S_L <= 1) || L <= ell)) {
    stop("Require 0 <= S_L <= 1 and L > ell")
  }

  1 + (L / ell) * (S_L - 1)
}

# === EXAMPLE INPUTS ===
#
#distributions <- list(
#  c(0.25, 0.25, 0.25, 0.25),
#  c(0.5, 0.25, 0.25, 0.0),
#  c(0.1, 0.7, 0.1, 0.1),
#  c(0.6, 0.0, 0.4, 0.0),
#  c(1.0, 0.0, 0.0, 0.0)
#)
#
#a <- 0.0
#b <- 4.0
#bins <- 10000L
#
#pdfs <- list(
#  list(
#    label = "Uniform",
#    pdf = function(x) {
#      vals <- rep(1, length(x))
#      vals / (sum(vals) * (b - a) / bins)
#    }
#  ),
#  list(
#    label = "Gaussian mu=2.8, sigma=1",
#    pdf = function(x) {
#      vals <- dnorm(x, mean = 2.8, sd = 1.0)
#      vals / (sum(vals) * (b - a) / bins)
#    }
#  ),
#  list(
#    label = "Mixture 0.5*N(1.2,0.3^2)+0.5*N(3.0,0.4^2)",
#    pdf = function(x) {
#      vals <- 0.5 * dnorm(x, mean = 1.2, sd = 0.3) +
#        0.5 * dnorm(x, mean = 3.0, sd = 0.4)
#      vals / (sum(vals) * (b - a) / bins)
#    }
#  )
#)

# === OUTPUT ===

#sharpness_scores <- numeric(length(distributions))

#for (i in seq_along(distributions)) {
#  p <- distributions[[i]]
#  s1 <- s_discrete(p)
#  s2 <- s_discrete_ml(p)
#  sharpness_scores[i] <- s1
#  cat(sprintf("Distribution %d: %s\n", i, paste(p, collapse = ", ")))
#  cat(sprintf("  Sharpness (simplified):   %.3f\n", s1))
#  cat(sprintf("  Sharpness (mass-length): %.3f\n", s2))
#}

#cat("\nContinuous Distributions on [0, 4]:\n\n")
#for (entry in pdfs) {
#  label <- entry$label
#  pdf <- entry$pdf
#  S_simpl <- sd(pdf, a, b, bins)
#  S_ml_val <- sd_ml(pdf, a, b, bins)
#  S_gini_val <- sd_gini(pdf, a, b, bins)
#  cat(sprintf("%-55s  S(sd)=%.4f | S(sd_ml)=%.4f | S(sd_gini)=%.4f\n",
#              label, S_simpl, S_ml_val, S_gini_val))
#}

# === OTHER FORMULAS ===

#cat("\n=== Discrete Relative Gain & Domain Transformation Example ===\n")
#m <- 4L
#n <- 7L
#S2 <- s_discrete(distributions[[2]])
#S4 <- s_discrete(distributions[[4]])
#rel_gain_discrete <- s_rel(S2, S4)
#cat(sprintf("DeltaS_REL (Dist 2 -> Dist 4): %.4f\n", rel_gain_discrete))

#S_val <- S2
#S_forward <- discrete_f(S_val, m, n)
#S_inverse <- discrete_i(S_forward, n, m)
#cat(sprintf("\nDomain transformation for Dist 2: %s\n", paste(distributions[[2]], collapse = ", ")))
#cat(sprintf("  S(P) = %.4f\n", S_val))
#cat(sprintf("  discrete_f (m=%d -> n=%d): %.4f\n", m, n, S_forward))
#cat(sprintf("  discrete_i (n=%d -> m=%d): %.4f\n", n, m, S_inverse))

#cat("\n=== Continuous Relative Gain & Domain Transformation Example ===\n")
#ell <- 4.0
#L <- 7.0
#S_gauss <- sd(pdfs[[2]]$pdf, a, b, bins)
#S_mix <- sd(pdfs[[3]]$pdf, a, b, bins)
#rel_gain_cont <- s_rel(S_gauss, S_mix)
#cat(sprintf("DeltaS_REL (Gaussian -> Mixture): %.4f\n", rel_gain_cont))

#S_val <- S_gauss
#S_forward <- continuous_f(S_val, ell, L)
#S_inverse <- continuous_i(S_forward, L, ell)
#cat("\nDomain transformation for Gaussian mu=2.8, sigma=1\n")
#cat(sprintf("  S(f) = %.4f\n", S_val))
#cat(sprintf("  continuous_f (ell=%.1f -> L=%.1f): %.4f\n", ell, L, S_forward))
#cat(sprintf("  continuous_i (L=%.1f -> ell=%.1f): %.4f\n", L, ell, S_inverse))
