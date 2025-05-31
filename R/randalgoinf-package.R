#' randalgoinf: Statistical Inference via Randomized Algorithms
#'
#' This package implements various statistical inference methods for the 
#' deterministic target of a sequence of randomized algorithms as developed 
#' in Zhang et al. (2023). The main inference framework is provided by 
#' \code{\link{general_inference}}, which implements sub-randomization, 
#' multi-run plug-in, and multi-run aggregation methods that require multiple runs 
#' of the randomized algorithm. The framework's applicability is demonstrated 
#' through inference for least squares parameters via sketch-and-solve, partial 
#' sketching, and iterative Hessian sketching. These methods are also applied 
#' in stochastic optimization, including SGD with Polyak-Ruppert 
#' averaging and momentum-based approaches. The package also incorporates 
#' classical pivotal inference methods, supported by newly developed theoretical 
#' results for sketched least squares estimators.
#'
#' @section Key Inference Methods:
#' 
#' **Sub-randomization Inference**: An inference method for constructing confidence 
#' intervals when using randomized algorithms. This method relies on running the 
#' randomized algorithm once for a larger sketch size m and several times for a 
#' smaller sketch size b < m. Sub-randomization is valid as long as the output of the 
#' randomized algorithm has an arbitrary, possibly unknown, limiting distribution.
#' 
#' **Multi-run Plug-in**: This method is based on the assumption that the output of the randomized 
#' algorithm is asymptotically normal, so the asymptotic variance can be estimated 
#' from K output estimates.
#' 
#' **Multi-run Aggregation**: An alternative multi-run approach that is based on 
#' the assumption that the output is nearly unbiased. This method aggregates 
#' results across multiple runs which is similarly to multi-run
#' plug-in inference, but centers the estimated error distribution at the empirical 
#' mean of the outputs. The scaling then depends on the square root of the 
#' number of runs.
#'
#' **Pivotal Methods**: This approach is based on estimating the variance (covariance) of the estimator
#' to construct pivotal quantities with known distributions.
#' 
#' @section Sketching functions:
#' \itemize{
#'   \item \code{\link{ske_iid}}: I.I.D. random sketching with various distributions
#'   \item \code{\link{ske_haar}}: Haar (uniform orthogonal) sketching  
#'   \item \code{\link{ske_hadamard}}: Hadamard (SRHT) sketching for fast computation
#'   \item \code{\link{ite_ske}}: Iterative Hessian sketching algorithm
#' }
#'
#' @section Inference methods:
#' 
#' **Main Framework:**
#' \itemize{
#'   \item \code{\link{general_inference}}: General inference framework for sub-randomization, multi-run plug-in, and multi-run aggregation
#' }
#' 
#' **Specialized Methods:**
#' \itemize{
#'   \item \code{\link{inf_pivotal}}: Pivotal inference for sketch-and-solve estimators
#'   \item \code{\link{inf_pivotal_ihs}}: Pivotal inference for iterative sketching (uses Gaussian sketch)
#' }
#'
#' @section Stochastic Optimization:
#' \itemize{
#'   \item \code{\link{PRA_logistic}}: SGD with Polyak-Ruppert averaging for logistic regression
#'   \item \code{\link{sgd_l2svm}}: SGD for L2-regularized SVM with optional momentum
#' }
#'
#' @section Utility functions:
#' \itemize{
#'   \item \code{\link{mem_eff_ske}}: Memory-efficient sketching for large datasets
#' }
#'
#' @examples
#' # Basic sketching and inference example
#' # TARGET for sketching: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 100; p <- 5; m <- 30; b <- 15
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X \%*\% c(1, -0.5, 0.3, 0.8, -0.2) + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)  # sum of coordinates of solution
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target_ols <- sum(c_vec * ols_beta)
#'
#' # Perform different types of sketching
#' result_iid <- ske_iid(X, y, m, c_vec)
#' result_haar <- ske_haar(X, y, m, c_vec)
#' result_hadamard <- ske_hadamard(X, y, m, c_vec)
#'
#' # View estimates
#' cat("True target (OLS):", round(true_target_ols, 3), "\\n")
#' cat("IID estimate:", round(result_iid$linear_combination, 3), "\\n")
#' cat("Haar estimate:", round(result_haar$linear_combination, 3), "\\n") 
#' cat("Hadamard estimate:", round(result_hadamard$linear_combination, 3), "\\n")
#'
#' # General inference framework using Hadamard sketching
#' # Generate main estimate (large sketch)
#' main_sketch <- ske_hadamard(X, y, m, c_vec)
#' theta_m <- main_sketch$linear_combination
#'
#' # Generate reduced-size estimates (small sketches)
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
#' # Apply different inference methods
#' subrand_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
#'                                    method = "subrand")
#' plugin_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
#'                                   method = "plugin")
#' multirun_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
#'                                     method = "multirun")
#'
#' cat("Sub-randomization 90\\% CI:", round(subrand_result$confidence_interval, 3), "\\n")
#' cat("Multi-run plug-in 90\\% CI:", round(plugin_result$confidence_interval, 3), "\\n")
#' cat("Multi-run aggregation 90\\% CI:", round(multirun_result$confidence_interval, 3), "\\n")
#'
#' # Pivotal inference for comparison
#' inference_pivotal <- inf_pivotal(
#'   result_haar$sketched_data[, 1:p], 
#'   result_haar$sketched_data[, p+1], 
#'   c_vec, n, "haar"
#' )
#' cat("Pivotal 90\\% CI:", round(inference_pivotal$confidence_interval, 3), "\\n")
#'
#' # SGD Example: PRA for Logistic Regression
#' d <- 5  # Number of features
#' theta_true <- seq(0, d, d/4) / d  # True population parameters
#' true_target_param <- theta_true[5]  # Target: last coordinate
#' data_gen <- randalgoinf:::create_logistic_generator(theta_true)
#'
#' # Set learning rate schedules  
#' n_main <- 1000; n_sub <- 500
#' lr_main <- 0.5 * (1:n_main)^(-0.505)
#' lr_sub <- 0.5 * (1:n_sub)^(-0.505)
#'
#' # Get main estimate (large sample size)
#' main_result <- PRA_logistic(lr_main, n_main, data_gen, 5, average = TRUE)
#' theta_m_pra <- main_result$averaged_weights[5]  # Last coordinate
#'
#' # Get reduced-size estimates 
#' theta_b_list_pra <- list()
#' for (k in 1:30) {
#'   sub_result <- PRA_logistic(lr_sub, n_sub, data_gen, 5, average = TRUE)
#'   theta_b_list_pra[[k]] <- sub_result$averaged_weights[5]
#' }
#'
#' # Apply general inference (sqrt rate for PRA)
#' pra_inference <- general_inference(theta_m_pra, theta_b_list_pra, 
#'                                   sqrt(n_main), sqrt(n_sub), method = "subrand")
#' cat("True target (population param):", true_target_param, "\\n")
#' cat("PRA 90\\% CI:", round(pra_inference$confidence_interval, 3), "\\n")
#'
#' # Multi-run plug-in inference for PRA
#' pra_plugin <- general_inference(theta_m_pra, theta_b_list_pra, 
#'                                sqrt(n_main), sqrt(n_sub), method = "plugin")
#' cat("PRA Multi-run plug-in 90\\% CI:", round(pra_plugin$confidence_interval, 3), "\\n")
#' }
#'
#' @references
#' Zhang, Z. et al. (2023). Statistical Inference via Randomized Algorithms. 
#' arXiv preprint. \url{https://arxiv.org/pdf/2307.11255}
#'
#' @docType package
#' @name randalgoinf-package
#' @import mvtnorm
#' @import phangorn
#' @import data.table
NULL 