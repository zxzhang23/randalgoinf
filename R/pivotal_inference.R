#' Pivotal Inference for Sketch-and-Solve Estimators
#'
#' Performs pivotal inference for different types of sketch-and-solve estimators
#' (I.I.D., Haar, or Hadamard).
#'
#' @param sketched_X Sketched design matrix (m x p)
#' @param sketched_y Sketched response vector (m x 1)
#' @param c Linear combination vector (p x 1)
#' @param n Original sample size
#' @param method Sketching method: "iid", "haar", or "hadamard"
#' @param alpha Significance level (default = 0.10)
#' @param partial Whether partial sketching was used (0=full, 1=partial)
#' @param Xty For partial sketching, the X'y vector (optional)
#'
#' @return List containing:
#' \item{confidence_interval}{90% confidence interval}
#' \item{point_estimate}{Point estimate of c'beta}
#' @family inference
#' @seealso \code{\link{general_inference}}, \code{\link{inf_pivotal_ihs}}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution obtained from full data
#' set.seed(123)
#' n <- 100; p <- 5; m <- 30
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X %*% c(1, -0.5, 0.3, 0.8, -0.2) + rnorm(n, sd = 0.1)
#' c_vec <- c(1, 0, 0, 0, 0)  # first coordinate
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Perform sketching
#' result <- ske_iid(X, y, m, c_vec)
#'
#' # Pivotal inference
#' inference <- inf_pivotal(
#'   result$sketched_data[, 1:p],
#'   result$sketched_data[, p+1],
#'   c_vec, n, "iid"
#' )
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("90\\% CI:", round(inference$confidence_interval, 3), "\\n")
#'
#' @export
inf_pivotal <- function(sketched_X, sketched_y, c, n, method = "iid",
                       alpha = 0.10, partial = 0, Xty = NULL) {
  m <- nrow(sketched_X)
  p <- ncol(sketched_X)
  xi <- p / n
  gamma <- m / n
  invsX <- solve(t(sketched_X) %*% sketched_X)

  if (partial == 0) {
    lssk <- invsX %*% (t(sketched_X) %*% sketched_y)
    center <- sum(c * lssk)
    sep <- sketched_y - sketched_X %*% lssk

    if (method == "iid") {
      est_v <- (gamma / (gamma - xi)) * sum(c * (invsX %*% c)) * sum(sep^2)
    } else if (method %in% c("haar", "hadamard")) {
      est_v <- (gamma * (1 - gamma) / ((gamma - xi) * (1 - xi))) * sum(c * (invsX %*% c)) * sum(sep^2)
    }
  } else {
    if (is.null(Xty)) stop("Xty must be provided for partial sketching")
    lssk <- invsX %*% Xty

    if (method == "iid") {
      center <- (m - p) * sum(c * lssk) / m
      est_v <- (gamma / (gamma - xi)) * ((sum((sketched_X %*% lssk)^2) * sum(c * (invsX %*% c)) + (sum(c * lssk))^2))
    } else if (method == "haar") {
      center <- ((gamma - xi) / (gamma * (1 - xi))) * sum(c * lssk)
      est_v <- ((1 - gamma) * (gamma - xi) / (gamma * (1 - xi)^3)) * ((sum((sketched_X %*% lssk)^2) * sum(c * (invsX %*% c)) + (sum(c * lssk))^2))
    } else if (method == "hadamard") {
      center <- ((gamma - xi) / (gamma * (1 - xi))) * sum(c * lssk)
      est_v <- ((1 - gamma) * (gamma - xi) / (gamma * (1 - xi)^3)) * ((sum((sketched_X %*% lssk)^2) * sum(c * (invsX %*% c)) + 2 * (sum(c * lssk))^2))
    }
  }

  rb <- qnorm(1 - alpha / 2, sd = sqrt(est_v / m))
  lb <- qnorm(alpha / 2, sd = sqrt(est_v / m))

  return(list(
    confidence_interval = c(center - rb, center - lb),
    point_estimate = center
  ))
}



#' Pivotal Inference for Iterative Sketching
#'
#' Performs pivotal inference for iterative sketch-and-solve estimators using
#' Gaussian Orthogonal Ensemble (GOE) simulation.
#'
#' @param X Design matrix (n x p)
#' @param y Response vector (n x 1)
#' @param c Linear combination vector (p x 1)
#' @param m Sketch size per iteration
#' @param iterations Number of iterations
#' @param alpha Significance level (default = 0.10)
#' @param num_simulations Number of simulations for quantile estimation (default = 500)
#' @param refresh_sketch Whether to refresh sketching matrix each iteration (default = TRUE)
#'
#' @return List containing:
#' \item{point_estimates}{Point estimates from each iteration}
#' \item{confidence_intervals}{Matrix of confidence intervals for each iteration}
#' \item{lower_bounds}{Lower confidence bounds for each iteration}
#' \item{upper_bounds}{Upper confidence bounds for each iteration}
#' @family inference
#' @seealso \code{\link{general_inference}}, \code{\link{inf_pivotal}}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution obtained from full data
#' set.seed(123)
#' n <- 5000; p <- 10; m <- 500
#' X <- matrix(rnorm(n * p), n, p)
#' true_beta <- rnorm(p)
#' y <- X %*% true_beta + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Pivotal inference for iterative sketching
#' inference <- inf_pivotal_ihs(X, y, c_vec, m = 500, iterations = 5,
#'                              num_simulations = 100)
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("Final point estimate:", round(inference$point_estimates[5], 3), "\\n")
#' cat("Final 90\\% CI: [", round(inference$lower_bounds[5], 3), ",",
#'     round(inference$upper_bounds[5], 3), "]\\n")
#'
#' @export
inf_pivotal_ihs <- function(X, y, c, m, iterations = 5, alpha = 0.10,
                           num_simulations = 100, refresh_sketch = TRUE) {
  # Get iterative sketching estimates
  iter_result <- ite_ske(X, y, c, m, iterations, refresh_sketch)
  iteration_estimates <- iter_result$linear_combinations

  # Extract or generate sketched matrix for inference
  n <- nrow(X)
  p <- ncol(X)

  if (refresh_sketch) {
    # For refreshed sketching, use the first iteration's sketching matrix
    S <- matrix(rnorm(m * n), m) / sqrt(m)
    sX <- S %*% X
  } else {
    # For fixed sketching, generate the single matrix used
    S <- matrix(rnorm(m * n), m) / sqrt(m)
    sX <- S %*% X
  }

  Xty <- t(X) %*% y

  # SVD decomposition
  svd_result <- svd(sX)
  U <- svd_result$u
  D <- svd_result$d
  V <- svd_result$v

  # Precompute matrices
  if (refresh_sketch) {
    A <- V %*% diag(D^(-1))
  } else {
    A <- -V %*% diag(D^(-1))  # Note the negative sign for fixed sketching
  }
  Uty <- diag(D^(-1)) %*% t(V) %*% Xty

  # Simulation for quantiles
  simulated_values <- matrix(0, num_simulations, iterations)

  for (sim in 1:num_simulations) {
    GUty <- Uty
    if (refresh_sketch) {
      for (iter in 1:iterations) {
        G <- generate_GOE(p)
        GUty <- G %*% GUty
        simulated_values[sim, iter] <- sum(c * (A %*% GUty)) / m^(iter / 2)
      }
    } else {
      G <- generate_GOE(p)  # Single GOE matrix for fixed sketching
      for (iter in 1:iterations) {
        GUty <- G %*% GUty
        simulated_values[sim, iter] <- sum(c * (A %*% GUty)) / m^(iter / 2)
      }
    }
  }

  # Compute confidence bounds
  upper_quantiles <- apply(simulated_values, 2, quantile, probs = 1 - alpha / 2)
  lower_quantiles <- apply(simulated_values, 2, quantile, probs = alpha / 2)

  lower_bounds <- iteration_estimates - upper_quantiles
  upper_bounds <- iteration_estimates - lower_quantiles

  # Combine into confidence intervals matrix
  confidence_intervals <- cbind(lower_bounds, upper_bounds)
  colnames(confidence_intervals) <- c("Lower", "Upper")
  rownames(confidence_intervals) <- paste("Iteration", 1:iterations)

  return(list(
    point_estimates = iteration_estimates,
    confidence_intervals = confidence_intervals,
    lower_bounds = lower_bounds,
    upper_bounds = upper_bounds
  ))
}


# Helper function for generating Gaussian Orthogonal Ensemble (GOE) matrix
generate_GOE <- function(n) {
  Z <- matrix(rnorm(n * n), n, n)
  GOE <- (Z + t(Z)) / sqrt(2)
  return(GOE)
} 
