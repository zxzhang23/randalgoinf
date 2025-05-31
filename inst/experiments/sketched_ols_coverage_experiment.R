# Coverage Ratio Experiment for General Inference Methods
# Run 100 replications to check empirical coverage rates

# Load the package
library(randalgoinf)

# Experiment parameters 
n_replications <- 100
alpha <- 0.10  # 90% confidence intervals
target_coverage <- 1 - alpha  # Should be 0.90

# Setup (small size example)
n <- 1000  # Large sample size
p <- 50    # Number of features
m <- 300   # Large sketch size (main estimate)
b <- 100   # Small sketch size (reduced-size estimates)
K <- 50    # Number of reduced-size estimates

# Linear combination of interest (sum of first 3 coefficients)
c_vec <- c(1, 1, 1, rep(0, 7))

# Generate fixed data for the experiment (same across all replications)
set.seed(123)  # Fixed seed for data generation
X <- matrix(rnorm(n * p), n, p)
true_beta <- runif(p, 0, 1)
y <- X %*% true_beta + rnorm(n, sd = 0.2)

# Calculate true target (OLS solution) - this is what we're trying to estimate
ols_beta <- qr.solve(X, y)
true_target <- sum(c_vec * ols_beta)

cat("=== Coverage Ratio Experiment ===\n")
cat("Running", n_replications, "replications with 90% confidence intervals\n")
cat("Fixed data: n =", n, ", p =", p, ", true target =", round(true_target, 4), "\n\n")

# Storage for results
coverage_subrand <- logical(n_replications)
coverage_plugin <- logical(n_replications)
coverage_multirun <- logical(n_replications)
ci_length_subrand <- numeric(n_replications)
ci_length_plugin <- numeric(n_replications)
ci_length_multirun <- numeric(n_replications)

# Progress tracking
cat("Progress: ")

# Run the experiment
for (rep in 1:n_replications) {
  # Progress indicator
  if (rep %% 10 == 0) cat(rep, "")
  
  # Get main estimate (large sketch size)
  main_sketch <- ske_hadamard(X, y, m, c_vec)
  theta_m <- main_sketch$linear_combination
  
  # Get reduced-size estimates (small sketch size, repeated K times)
  theta_b_list <- list()
  for (k in 1:K) {
    sub_sketch <- ske_hadamard(X, y, b, c_vec)
    theta_b_list[[k]] <- sub_sketch$linear_combination
  }
  
  # Set convergence parameters for Hadamard sketching
  n_eff <- 2^ceiling(log2(n))  # Padded size for Hadamard transform
  tau_m <- sqrt((m - p) * (n_eff - p) / (n_eff - m))
  tau_b <- sqrt((b - p) * (n_eff - p) / (n_eff - b))
  
  # Apply general inference methods
  result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                     method = "subrand", alpha = alpha)
  result_plugin <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                    method = "plugin", alpha = alpha)
  result_multirun <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                      method = "multirun", alpha = alpha)
  
  # Check coverage (does CI contain true target?)
  coverage_subrand[rep] <- (result_subrand$confidence_interval[1] <= true_target) & 
                          (true_target <= result_subrand$confidence_interval[2])
  coverage_plugin[rep] <- (result_plugin$confidence_interval[1] <= true_target) & 
                         (true_target <= result_plugin$confidence_interval[2])
  coverage_multirun[rep] <- (result_multirun$confidence_interval[1] <= true_target) & 
                           (true_target <= result_multirun$confidence_interval[2])
  
  # Store CI lengths
  ci_length_subrand[rep] <- diff(result_subrand$confidence_interval)
  ci_length_plugin[rep] <- diff(result_plugin$confidence_interval)
  ci_length_multirun[rep] <- diff(result_multirun$confidence_interval)
}

cat("\n\n=== Coverage Results ===\n")
cat("Target coverage: 90%\n")
cat("True target:", round(true_target, 4), "\n\n")

# Calculate and display coverage rates
coverage_rate_subrand <- mean(coverage_subrand)
coverage_rate_plugin <- mean(coverage_plugin)
coverage_rate_multirun <- mean(coverage_multirun)

cat("Coverage Rates:\n")
cat(sprintf("Sub-randomization:     %.1f%%\n", coverage_rate_subrand * 100))
cat(sprintf("Multi-run plug-in:     %.1f%%\n", coverage_rate_plugin * 100))
cat(sprintf("Multi-run aggregation: %.1f%%\n", coverage_rate_multirun * 100))

cat("\nAverage CI Lengths:\n")
cat(sprintf("Sub-randomization:     %.4f\n", mean(ci_length_subrand)))
cat(sprintf("Multi-run plug-in:     %.4f\n", mean(ci_length_plugin)))
cat(sprintf("Multi-run aggregation: %.4f\n", mean(ci_length_multirun)))
