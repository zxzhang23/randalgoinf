# Iterative Hessian Sketching Inference Example
# Demonstrates using general_inference function for Iterative Hessian Sketching
# in least squares regression

# Load the package
library(randalgoinf)

cat("=== Iterative Hessian Sketching Inference Example ===\n\n")

# Problem setup (following R/inference_ihs.R examples)
set.seed(123)
n <- 5000  # Sample size
p <- 10    # Number of features
c_vec <- rep(1, p)  # Linear combination vector (sum of all coefficients)

# Generate data
X <- matrix(rnorm(n * p), n, p)
true_beta <- rnorm(p)
y <- X %*% true_beta + rnorm(n, sd = 0.1)

# Calculate OLS solution and true target
ols_beta <- qr.solve(X, y)
true_target <- sum(c_vec * ols_beta)

cat("Problem setup:\n")
cat("- Sample size (n):", n, "\n")
cat("- Features (p):", p, "\n")
cat("- Target: sum of all coefficients\n")
cat("- True target (OLS):", round(true_target, 4), "\n\n")

# Algorithm parameters
m <- 500          # Large sketch size (main estimate)
b <- 100          # Small sketch size (reduced-size estimates)
K <- 50           # Number of reduced-size estimates
iterations <- 5   # Number of iterations for iterative sketching

cat("Algorithm parameters:\n")
cat("- Main sketch size (m):", m, "\n")
cat("- Reduced-size sketch size (b):", b, "\n")
cat("- Number of reduced-size estimates (K):", K, "\n")
cat("- Iterations per sketch:", iterations, "\n\n")

# Get main estimate (large sketch size with multiple iterations)
cat("Running main iterative sketching estimate (m =", m, ", iterations =", iterations, ")...\n")
main_result <- ite_ske(X, y, c_vec, m, iterations = iterations, refresh_sketch = TRUE)
theta_m <- main_result$final_linear_combination

cat("Main estimate:", round(theta_m, 4), "\n\n")

# Get reduced-size estimates (small sketch size, repeated K times)
cat("Running", K, "reduced-size estimates (b =", b, ", iterations =", iterations, "each)...\n")

theta_b_list <- list()
for (k in 1:K) {
  if (k %% 10 == 0) cat("Reduced-size estimate", k, "of", K, "\n")
  
  sub_result <- ite_ske(X, y, c_vec, b, iterations = iterations, refresh_sketch = TRUE)
  theta_b_list[[k]] <- sub_result$final_linear_combination
}

cat("Reduced-size estimates completed.\n")
cat("Range: [", round(min(unlist(theta_b_list)), 4), ",", 
    round(max(unlist(theta_b_list)), 4), "]\n")
cat("Mean:", round(mean(unlist(theta_b_list)), 4), "\n\n")

# Set convergence parameters for iterative Hessian sketching
# For iterative sketching with t iterations, convergence rate is typically m^(t/2)
# where m is the sketch size and t is the number of iterations
tau_m <- m^(iterations/2)
tau_b <- b^(iterations/2)

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

