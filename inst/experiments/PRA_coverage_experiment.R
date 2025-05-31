# PRA Logistic Regression Coverage Experiment
# Run 100 replications to check empirical coverage rates using general_inference

# Load the package
library(randalgoinf)

# Experiment parameters
n_replications <- 50
alpha <- 0.10  # 90% confidence intervals
target_coverage <- 1 - alpha  # Should be 0.90

# Setup 
d <- 5  # Number of features
theta_true <- seq(0, d, d/4) / d  # True coefficients
target_coef_index <- d  # We'll perform inference on the last coefficient
true_target <- theta_true[target_coef_index]

# Algorithm parameters
m <- 30000   # Large sample size (main estimate)
b <- 400    # Small sample size (reduced-size estimates)
K <- 50     # Number of reduced-size estimates

# Learning rate schedules
learning_rates_m <- 0.5 * (1:m)^(-0.505)
learning_rates_b <- 0.5 * (1:b)^(-0.505)

# Create data generator (fixed across all replications)
data_gen <- create_logistic_generator(theta_true)

cat("=== PRA Logistic Regression Coverage Experiment ===\n")
cat("Running", n_replications, "replications with 90% confidence intervals\n")
cat("Problem: d =", d, ", target coefficient =", target_coef_index, ", true target =", round(true_target, 4), "\n")
cat("Algorithm: m =", m, ", b =", b, ", K =", K, "\n\n")

# Storage for results
coverage_subrand <- logical(n_replications)
coverage_plugin <- logical(n_replications)
ci_length_subrand <- numeric(n_replications)
ci_length_plugin <- numeric(n_replications)

# Progress tracking
cat("Progress: ")
set.seed(456)

# Run the experiment
for (rep in 1:n_replications) {
  # Progress indicator
  if (rep %% 10 == 0) cat(rep, "")
  
  # Set seed for this replication (different randomness each time)
  
  # Get main estimate (large sample size)
  main_result <- PRA_logistic(learning_rates_m, m, data_gen, d, average = TRUE)
  theta_m <- main_result$averaged_weights[target_coef_index]
  
  # Find a good starting point using burn-in
  burnin_result <- PRA_logistic(learning_rates_b, 1000, data_gen, d, average = TRUE)
  start_point <- burnin_result$averaged_weights
  
  # Get reduced-size estimates (small sample size, repeated K times)
  theta_b_list <- list()
  for (k in 1:K) {
    sub_result <- PRA_logistic(learning_rates_b, b, data_gen, d, 
                              average = TRUE, start_point = start_point)
    theta_b_list[[k]] <- sub_result$averaged_weights[target_coef_index]
  }
  
  # Set convergence parameters for PRA logistic regression
  tau_m <- sqrt(m)
  tau_b <- sqrt(b)
  
  # Apply general inference methods
  result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                     method = "subrand", alpha = alpha)
  result_plugin <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                    method = "plugin", alpha = alpha)
  
  # Check coverage (does CI contain true target?)
  coverage_subrand[rep] <- (result_subrand$confidence_interval[1] <= true_target) & 
                          (true_target <= result_subrand$confidence_interval[2])
  coverage_plugin[rep] <- (result_plugin$confidence_interval[1] <= true_target) & 
                         (true_target <= result_plugin$confidence_interval[2])
  
  # Store CI lengths
  ci_length_subrand[rep] <- diff(result_subrand$confidence_interval)
  ci_length_plugin[rep] <- diff(result_plugin$confidence_interval)
}

cat("\n\n=== Coverage Results ===\n")
cat("Target coverage: 90%\n")
cat("True target:", round(true_target, 4), "\n\n")

# Calculate and display coverage rates
coverage_rate_subrand <- mean(coverage_subrand)
coverage_rate_plugin <- mean(coverage_plugin)

cat("Coverage Rates:\n")
cat(sprintf("Sub-randomization:     %.1f%%\n", coverage_rate_subrand * 100))
cat(sprintf("Multi-run plug-in:     %.1f%%\n", coverage_rate_plugin * 100))

cat("\nAverage CI Lengths:\n")
cat(sprintf("Sub-randomization:     %.4f\n", mean(ci_length_subrand)))
cat(sprintf("Multi-run plug-in:     %.4f\n", mean(ci_length_plugin)))
