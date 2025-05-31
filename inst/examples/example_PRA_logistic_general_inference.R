# PRA Logistic Regression Inference Example
# Demonstrates using general_inference function for Polyak-Ruppert Averaging
# in logistic regression

# Load the package
library(randalgoinf)

cat("=== PRA Logistic Regression Inference Example ===\n\n")

# Problem setup (exactly as in package examples)
set.seed(123)
d <- 5  # Number of features
theta_true <- seq(0, d, d/4) / d  # True coefficients
target_coef_index <- d  # We'll perform inference on the last coefficient
true_target <- theta_true[target_coef_index]

cat("True coefficients:", round(theta_true, 3), "\n")
cat("Target coefficient (index", target_coef_index, " ):", true_target, "\n\n")

# Algorithm parameters
m <- 10000  # Large sample size (main estimate)
b <- 500    # Small sample size (reduced-size estimates)
K <- 50     # Number of reduced-size estimates

# Learning rate schedule (decreasing)
learning_rates_m <- 0.5 * (1:m)^(-0.505)
learning_rates_b <- 0.5 * (1:b)^(-0.505)

# Create data generator
data_gen <- create_logistic_generator(theta_true)

cat("Algorithm parameters:\n")
cat("- Main sample size (m):", m, "\n")
cat("- Reduced-size sample size (b):", b, "\n")
cat("- Number of reduced-size estimates (K):", K, "\n")
cat("- Learning rate: 0.5 * t^(-0.505)\n\n")

# Get main estimate (large sample size)
cat("Running main PRA estimate (m =", m, ")...\n")
main_result <- PRA_logistic(learning_rates_m, m, data_gen, d, average = TRUE)
theta_m <- main_result$averaged_weights[target_coef_index]

cat("Main estimate:", round(theta_m, 4), "\n\n")

# Get reduced-size estimates (small sample size, repeated K times)
cat("Running", K, "reduced-size estimates (b =", b, "each)...\n")

# Find a good starting point using burn-in
burnin_result <- PRA_logistic(learning_rates_b, 1000, data_gen, d, average = TRUE)
start_point <- burnin_result$averaged_weights

theta_b_list <- list()
for (k in 1:K) {
  if (k %% 10 == 0) cat("Reduced-size estimate", k, "of", K, "\n")
  
  sub_result <- PRA_logistic(learning_rates_b, b, data_gen, d, 
                            average = TRUE, start_point = start_point)
  theta_b_list[[k]] <- sub_result$averaged_weights[target_coef_index]
}

cat("Reduced-size estimates completed.\n\n")

# Set convergence parameters for PRA logistic regression
# For PRA, convergence rate is typically sqrt(sample_size)
tau_m <- sqrt(m)
tau_b <- sqrt(b)

cat("Convergence parameters:\n")
cat("- tau_m (main):", round(tau_m, 2), "\n")
cat("- tau_b (reduced-size):", round(tau_b, 2), "\n\n")

# Apply general inference methods
cat("=== Inference Results ===\n")

# Sub-randomization
result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                   method = "subrand", alpha = 0.10)

# Multi-run plug-in
result_plugin <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                  method = "plugin", alpha = 0.10)

# Display results
cat("True target value:", round(true_target, 4), "\n\n")

cat("Sub-randomization:\n")
cat("  Point estimate:", round(result_subrand$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_subrand$confidence_interval[1], 4), ", ", 
    round(result_subrand$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_subrand$confidence_interval), 4), "\n")
cat("  Contains true value:", 
    (result_subrand$confidence_interval[1] <= true_target) & 
    (true_target <= result_subrand$confidence_interval[2]), "\n\n")

cat("Multi-run plug-in:\n")
cat("  Point estimate:", round(result_plugin$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_plugin$confidence_interval[1], 4), ", ", 
    round(result_plugin$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_plugin$confidence_interval), 4), "\n")
cat("  Contains true value:", 
    (result_plugin$confidence_interval[1] <= true_target) & 
    (true_target <= result_plugin$confidence_interval[2]), "\n\n")



