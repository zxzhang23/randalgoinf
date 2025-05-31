# SGD L2-SVM Inference Example
# Demonstrates using general_inference function for Stochastic Gradient Descent
# on L2-Support Vector Machine problems

# Load the package
library(randalgoinf)

cat("=== SGD L2-SVM Inference Example ===\n\n")

# Problem setup
set.seed(123)
d <- 5  # Dimension of the parameter space
target_coef_index <- 1  # We'll perform inference on the first coefficient

# Create L2-SVM data generator with default parameters
data_gen <- create_l2svm_generator()

cat("Problem setup:\n")
cat("- Dimension (d):", d, "\n")
cat("- Target coefficient index:", target_coef_index, "\n")
cat("- Data: Mixture of two multivariate normals\n")
cat("- Mean 1: c(1, 1, 1, 0, 0), Mean 2: c(0, 0, 1, 1, 1)\n")
cat("- Mixture weights: c(0.2, 0.8)\n\n")

# Algorithm parameters (following sgd_l2svm.R example values)
m <- 10000  # Large sample size (main estimate)
b <- 600    # Small sample size (reduced-size estimates)
K <- 50     # Number of reduced-size estimates
eta <- 0.4  # Learning rate parameter
a <- 0.6    # Learning rate decay parameter
lambda <- 0 # L2 regularization parameter

# Generate learning rate schedules
learning_rates_m <- eta * (1:m)^(-a)
learning_rates_b <- eta * (1:b)^(-a)

cat("Algorithm parameters:\n")
cat("- Main sample size (m):", m, "\n")
cat("- Reduced-size sample size (b):", b, "\n")
cat("- Number of reduced-size estimates (K):", K, "\n")
cat("- Learning rate: eta * t^(-a) with eta =", eta, ", a =", a, "\n")
cat("- L2 regularization (lambda):", lambda, "\n\n")

# Get main estimate (large sample size)
cat("Running main SGD estimate (m =", m, ")...\n")
main_result <- sgd_l2svm(learning_rates_m, m, data_gen, d, lambda = lambda, use_momentum = TRUE, momentum_coef = 1)
theta_m <- main_result$w[target_coef_index]

cat("Main estimate:", round(theta_m, 4), "\n\n")

# Get reduced-size estimates (small sample size, repeated K times)
cat("Running", K, "reduced-size estimates (b =", b, "each)...\n")

theta_b_list <- list()
for (k in 1:K) {
  if (k %% 10 == 0) cat("Reduced-size estimate", k, "of", K, "\n")
  
  sub_result <- sgd_l2svm(learning_rates_b, b, data_gen, d, lambda = lambda, use_momentum = TRUE, momentum_coef = 1)
  theta_b_list[[k]] <- sub_result$w[target_coef_index]
}

cat("Reduced-size estimates completed.\n")
cat("Range: [", round(min(unlist(theta_b_list)), 4), ",", 
    round(max(unlist(theta_b_list)), 4), "]\n")
cat("Mean:", round(mean(unlist(theta_b_list)), 4), "\n\n")

# Set convergence parameters for SGD L2-SVM
# For SGD with polynomial learning rate eta * t^(-a), convergence rate is t^(a/2)
tau_m <- m^(a/2)
tau_b <- b^(a/2)

cat("Convergence parameters:\n")
cat("- tau_m (main):", round(tau_m, 2), "\n")
cat("- tau_b (reduced-size):", round(tau_b, 2), "\n")
cat("- Convergence ratio (tau_m/tau_b):", round(tau_m/tau_b, 2), "\n\n")

# Apply general inference methods
cat("=== Inference Results ===\n")

# Sub-randomization
result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                   method = "subrand", alpha = 0.10)

# Multi-run plug-in
result_plugin <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                  method = "plugin", alpha = 0.10)

# Display results
cat("Sub-randomization:\n")
cat("  Point estimate:", round(result_subrand$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_subrand$confidence_interval[1], 4), ", ", 
    round(result_subrand$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_subrand$confidence_interval), 4), "\n\n")

cat("Multi-run plug-in:\n")
cat("  Point estimate:", round(result_plugin$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_plugin$confidence_interval[1], 4), ", ", 
    round(result_plugin$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_plugin$confidence_interval), 4), "\n\n")


