# Example: General Inference with Sketch-and-Solve Least Squares
# Using Hadamard sketching to illustrate the general inference framework

# Load the package
library(randalgoinf)

# Set up the problem
set.seed(123)
n <- 1000  # Large sample size
p <- 10    # Number of features
m <- 300   # Large sketch size (main estimate)
b <- 150   # Small sketch size (reduced-size estimates)
K <- 50    # Number of reduced-size estimates

# Generate data
X <- matrix(rnorm(n * p), n, p)
true_beta <- runif(p, 0, 1)
y <- X %*% true_beta + rnorm(n, sd = 0.2)

# Linear combination of interest (e.g., sum of first 3 coefficients)
c_vec <- c(1, 1, 1, rep(0, 7))

# Calculate OLS solution and true target
ols_beta <- qr.solve(X, y)  # More numerically stable than solve(t(X) %*% X) %*% t(X) %*% y
true_target <- sum(c_vec * ols_beta)  # True target based on OLS

cat("=== Sketch-and-Solve with General Inference ===\n")
cat("True target parameter (OLS):", round(true_target, 4), "\n\n")

# Get main estimate (large sketch size m)
cat("Computing main estimate with m =", m, "\n")
main_sketch <- ske_hadamard(X, y, m, c_vec)
theta_m <- main_sketch$linear_combination

cat("Main estimate (theta_m):", round(theta_m, 4), "\n\n")

# Get reduced-size estimates (small sketch size b, repeated K times)
cat("Computing", K, "reduced-size estimates with b =", b, "\n")
theta_b_list <- list()

for (k in 1:K) {
  if (k %% 10 == 0) cat("  Estimate", k, "of", K, "\n")
  sub_sketch <- ske_hadamard(X, y, b, c_vec)
  theta_b_list[[k]] <- sub_sketch$linear_combination
}

cat("Reduced-size estimates range: [", round(min(unlist(theta_b_list)), 4), ",", 
    round(max(unlist(theta_b_list)), 4), "]\n")
cat("Reduced-size estimates mean:", round(mean(unlist(theta_b_list)), 4), "\n\n")

# Set convergence parameters
# For sketch-and-solve: convergence parameters depend on the sketching method
# Using formulas from inf_from_sketches function for Hadamard sketching
n_eff <- 2^ceiling(log2(n))  # Padded size for Hadamard transform
tau_m <- sqrt((m - p) * (n_eff - p) / (n_eff - m))  # Convergence parameter for main estimate
tau_b <- sqrt((b - p) * (n_eff - p) / (n_eff - b))  # Convergence parameter for reduced-size estimates

cat("Convergence parameters (Hadamard sketching)\n")
cat("Original n =", n, ", Effective n (padded) =", n_eff, "\n")
cat("tau_m =", round(tau_m, 4), ", tau_b =", round(tau_b, 4), "\n")
cat("Convergence ratio (tau_m/tau_b) =", round(tau_m/tau_b, 2), "\n\n")

# Apply general inference methods (90% CI)
cat("Applying general inference methods (90% CI)\n")

# Sub-randomization inference
result_subrand <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                   method = "subrand", alpha = 0.10)

cat("Sub-randomization:\n")
cat("  Point estimate:", round(result_subrand$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_subrand$confidence_interval[1], 4), ",", 
    round(result_subrand$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_subrand$confidence_interval), 4), "\n\n")

# Multi-run plug-in inference
result_plugin <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                  method = "plugin", alpha = 0.10)

cat("Multi-run plug-in:\n")
cat("  Point estimate:", round(result_plugin$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_plugin$confidence_interval[1], 4), ",", 
    round(result_plugin$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_plugin$confidence_interval), 4), "\n\n")

# Multi-run aggregation inference
result_multirun <- general_inference(theta_m, theta_b_list, tau_m, tau_b, 
                                    method = "multirun", alpha = 0.10)

cat("Multi-run aggregation:\n")
cat("  Point estimate:", round(result_multirun$point_estimate, 4), "\n")
cat("  90% CI: [", round(result_multirun$confidence_interval[1], 4), ",", 
    round(result_multirun$confidence_interval[2], 4), "]\n")
cat("  CI length:", round(diff(result_multirun$confidence_interval), 4), "\n\n")
