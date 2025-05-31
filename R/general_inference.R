#' General Inference for Randomized Algorithms that involve several runs of the algorithms
#'
#' @description Performs sub-randomization, multi-run plug-in, and multi-run aggregation inference
#'   for general randomized algorithms given large and small size solutions with 
#'   convergence rate parameters.
#'
#' @param theta_m Large size approximation solution (main estimate)
#' @param theta_b_list List of small size solutions (reduced-size estimates)
#' @param tau_m Convergence speed parameter for large size solution
#' @param tau_b Convergence speed parameter for small size solutions
#' @param method Inference method: "subrand", "plugin", or "multirun"
#' @param alpha Significance level for confidence intervals (default 0.10 for 90% CI)
#' @return List containing point estimate and confidence interval
#' @family inference
#' @seealso \code{\link{inf_pivotal}}, \code{\link{inf_pivotal_ihs}}
#' @export
#' @examples
#' # Example 1: Hadamard sketch-and-solve
#' # TARGET: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 1000; p <- 50
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X %*% rnorm(p) + rnorm(n, sd = 0.1)
#' c_vec <- c(1, 1, 1, rep(0, p-3))  # Linear combination of first 3 coordinates
#' 
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target_ols <- sum(c_vec * ols_beta)
#' 
#' # Algorithm parameters
#' m <- 300  # Large sketch size
#' b <- 100  # Small sketch size
#' 
#' # Main estimate (large sketch)
#' main_sketch <- ske_hadamard(X, y, m, c_vec)
#' theta_m <- main_sketch$linear_combination
#' 
#' # Reduced-size estimates (small sketches)
#' theta_b_list <- list()
#' K <- 50  # Number of reduced-size estimates
#' for (k in 1:K) {
#'   sub_sketch <- ske_hadamard(X, y, b, c_vec)
#'   theta_b_list[[k]] <- sub_sketch$linear_combination
#' }
#' 
#' # Convergence parameters for Hadamard sketching
#' n_eff <- 2^ceiling(log2(n))
#' tau_m <- sqrt((m - p) * (n_eff - p) / (n_eff - m))
#' tau_b <- sqrt((b - p) * (n_eff - p) / (n_eff - b))
#' 
#' # Apply general inference
#' result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, method = "subrand")
#' cat("True target (OLS):", round(true_target_ols, 3), "\\n")
#' cat("90\\% CI:", round(result$confidence_interval, 3), "\\n")
#' 
#' # Apply plug-in method (asymptotic normality holds)
#' plugin_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, method = "plugin")
#' cat("Plug-in 90\\% CI:", round(plugin_result$confidence_interval, 3), "\\n")
#' 
#' # Example 2: PRA logistic regression  
#' # TARGET: True regression parameter (population parameter)
#' \\dontrun{
#' set.seed(456)
#' d <- 5
#' theta_true <- seq(0, d, d/4) / d  # True population parameters
#' target_coef_index <- d  # Inference on last coordinate
#' true_target_param <- theta_true[target_coef_index]  # True population parameter
#' data_gen <- create_logistic_generator(theta_true)
#' 
#' # Algorithm parameters
#' m <- 10000  # Large sample size
#' b <- 400   # Small sample size
#' lr_main <- 0.5 * (1:m)^(-0.505)
#' lr_sub <- 0.5 * (1:b)^(-0.505)
#' 
#' # Get main estimate (large sample size)
#' main_result <- PRA_logistic(lr_main, m, data_gen, d, average = TRUE)
#' theta_m <- main_result$averaged_weights[target_coef_index]
#' 
#' # Get reduced-size estimates 
#' theta_b_list <- list()
#' K <- 50  # Number of reduced-size estimates
#' for (k in 1:K) {
#'   sub_result <- PRA_logistic(lr_sub, b, data_gen, d, average = TRUE)
#'   theta_b_list[[k]] <- sub_result$averaged_weights[target_coef_index]
#' }
#' 
#' # Convergence parameters for PRA (sqrt rate)
#' tau_m <- sqrt(m)
#' tau_b <- sqrt(b)
#' 
#' # Apply general inference
#' pra_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, method = "subrand")
#' cat("True target (population param):", true_target_param, "\\n")
#' cat("Point estimate:", round(pra_result$point_estimate, 3), "\\n")
#' cat("90\\% CI:", round(pra_result$confidence_interval, 3), "\\n")
#' }
general_inference <- function(theta_m, theta_b_list, tau_m, tau_b, 
                             method = "subrand", alpha = 0.10) {
  
  # Input validation
  if (!is.numeric(theta_m)) stop("theta_m must be numeric")
  if (!is.list(theta_b_list)) stop("theta_b_list must be a list")
  if (tau_m <= 0 || tau_b <= 0) stop("Convergence parameters must be positive")
  if (alpha <= 0 || alpha >= 1) stop("alpha must be between 0 and 1")
  if (!method %in% c("subrand", "plugin", "multirun")) {
    stop("method must be 'subrand', 'plugin', or 'multirun'")
  }
  
  # Convert to numeric vectors
  theta_b_vec <- unlist(theta_b_list)
  K <- length(theta_b_vec)  # Number of reduced-size estimates
  
  if (method == "subrand") {
    # Sub-randomization inference
    scaled_diffs <- tau_b * (theta_b_vec - theta_m)
    lb <- quantile(scaled_diffs, alpha / 2) / (tau_m - tau_b)
    rb <- quantile(scaled_diffs, (1 - alpha / 2)) / (tau_m - tau_b)
    
    center <- theta_m
    
    result <- list(
      method = "sub-randomization",
      point_estimate = center,
      confidence_interval = c(center - rb, center - lb)
    )
    
  } else if (method == "plugin") {
    # Multi-run plug-in inference    
    v <- var(theta_b_vec)
    rb <- qnorm(1 - alpha / 2, sd = sqrt(v) * tau_b / tau_m)
    lb <- qnorm(alpha / 2, sd = sqrt(v) * tau_b / tau_m)
    center <- theta_m
    
    result <- list(
      method = "multi-run plug-in",
      point_estimate = center,
      confidence_interval = c(center - rb, center - lb)
    )
    
  } else if (method == "multirun") {
    # Multi-run aggregation inference
    aggregated_est <- mean(theta_b_vec)
    var_aggregated <- var(theta_b_vec) / K  
    se_aggregated <- sqrt(var_aggregated)
    
    # Confidence interval centered at aggregated estimate
    z_alpha <- qnorm(1 - alpha/2)
    ci_lower <- aggregated_est - z_alpha * se_aggregated
    ci_upper <- aggregated_est + z_alpha * se_aggregated
    
    result <- list(
      method = "multi-run aggregation",
      point_estimate = aggregated_est,
      confidence_interval = c(ci_lower, ci_upper)
    )
  }
  
  result$alpha <- alpha
  result$K <- K
  result$tau_m <- tau_m
  result$tau_b <- tau_b
  result$coverage_probability <- 1 - alpha
  
  return(result)
}
