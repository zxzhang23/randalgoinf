# Iterative Hessian Sketching Coverage Experiment
# Runs 100 replications to evaluate empirical coverage rates using general_inference
#
# NOTE: For iterative Hessian sketching, the limiting distribution is NOT normal.

# Load the package
library(randalgoinf)

# Experiment parameters
n_replications <- 100
alpha <- 0.10  # 90% confidence intervals
target_coverage <- 1 - alpha  

# Setup (small size example)
n <- 1000  # Sample size
p <- 5  # Number of features
c_vec <- c(1, rep(0, p-1))  # First coordinate

# Algorithm parameters
m <- 500          # Large sketch size (main estimate)
b <- 100          # Small sketch size (reduced-size estimates)
K <- 50           # Number of reduced-size estimates
iterations <- 5   # Number of iterations for iterative sketching

# Generate fixed dataset for the experiment (consistent across all replications)
set.seed(123)  # Fixed seed for reproducible data generation
X <- matrix(rnorm(n * p), n, p)
true_beta <- rnorm(p)
y <- X %*% true_beta + rnorm(n, sd = 0.1)

# Calculate true target parameter (OLS solution)
ols_beta <- qr.solve(X, y)
true_target <- sum(c_vec * ols_beta)

cat("=== Iterative Hessian Sketching Coverage Experiment ===\n")
cat("Running", n_replications, "replications with 90% confidence intervals\n")
cat("Fixed dataset: n =", n, ", p =", p, ", true target =", round(true_target, 4), "\n")
cat("Algorithm parameters: m =", m, ", b =", b, ", K =", K, ", iterations =", iterations, "\n\n")

# Storage for results
coverage_subrand <- logical(n_replications)
ci_length_subrand <- numeric(n_replications)
ci_center_subrand <- numeric(n_replications)

# Progress tracking
cat("Progress: ")

# Run the experiment
for (rep in 1:n_replications) {
  # Progress indicator
  if (rep %% 10 == 0) cat(rep, "")
  
  # Compute main estimate (large sketch size with multiple iterations)
  # Each replication uses different random sketching matrices
  main_result <- ite_ske(X, y, c_vec, m, iterations = iterations, refresh_sketch = TRUE)
  theta_m <- main_result$final_linear_combination
  
  # Compute reduced-size estimates (small sketch size, repeated K times)
  theta_b_list <- list()
  for (k in 1:K) {
    sub_result <- ite_ske(X, y, c_vec, b, iterations = iterations, refresh_sketch = TRUE)
    theta_b_list[[k]] <- sub_result$final_linear_combination
  }
  
  # Set convergence parameters for iterative Hessian sketching
  # For iterative sketching with t iterations, convergence rate is m^(t/2)
  tau_m <- m^(iterations/2)
  tau_b <- b^(iterations/2)
  
  # Apply sub-randomization inference (the only appropriate method for this algorithm)
  result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                     method = "subrand", alpha = alpha)
  
  # Check coverage (does the confidence interval contain the true target?)
  coverage_subrand[rep] <- (result_subrand$confidence_interval[1] <= true_target) & 
                          (true_target <= result_subrand$confidence_interval[2])
  
  # Store confidence interval statistics
  ci_length_subrand[rep] <- diff(result_subrand$confidence_interval)
  ci_center_subrand[rep] <- result_subrand$point_estimate
}

cat("\n\n=== Coverage Results ===\n")
cat("Target coverage: 90%\n")
cat("True target:", round(true_target, 4), "\n\n")

# Calculate and display coverage rate
coverage_rate_subrand <- mean(coverage_subrand)

cat("Coverage Rate:\n")
cat(sprintf("Sub-randomization:     %.1f%%\n", coverage_rate_subrand * 100))

cat("\nAverage CI Length:\n")
cat(sprintf("Sub-randomization:     %.4f\n", mean(ci_length_subrand)))

