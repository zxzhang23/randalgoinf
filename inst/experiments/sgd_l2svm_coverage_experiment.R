# SGD L2-SVM Coverage Experiment
# Run 100 replications to check empirical coverage rates using general_inference

# Load the package
library(randalgoinf)

# Experiment parameters
n_replications <- 100
alpha <- 0.10  # 90% confidence intervals
target_coverage <- 1 - alpha  # Should be 0.90

# Problem setup
d <- 5  # Dimension of the parameter space
target_coef_index <- 1  # We'll perform inference on the first coefficient

# Algorithm parameters (following sgd_l2svm.R example values)
m <- 2000   # Large sample size (main estimate)
b <- 300    # Small sample size (reduced-size estimates)
K <- 30     # Number of reduced-size estimates
eta <- 0.4  # Learning rate parameter
a <- 0.7    # Learning rate decay parameter
lambda <- 0 # L2 regularization parameter

# Generate learning rate schedules
learning_rates_m <- eta * (1:m)^(-a)
learning_rates_b <- eta * (1:b)^(-a)

# Create data generator (fixed across all replications)
data_gen <- create_l2svm_generator()

# Find the underlying truth using gradient descent on large fixed dataset
cat("=== Finding True Target Parameter ===\n")
set.seed(123)  # Fixed seed for true parameter estimation

# Generate large fixed dataset
n_samples <- 500000
X <- matrix(0, n_samples, 5)
y <- numeric(n_samples)
for(i in 1:n_samples){
  new_sample <- data_gen()
  X[i,] <- new_sample$X[1,]
  y[i] <- new_sample$y[1]
}

# Gradient descent function for L2-SVM
gd_l2svm <- function(X, y, learning_rate, n_iter = 1000, lambda = 0) {
    n_samples <- nrow(X)
    n_features <- ncol(X)
    w <- rep(0, 5)
    w0 <- 0
    
    # Store solution path
    w_path <- matrix(0, nrow = n_iter, ncol = n_features)
    w0_path <- numeric(n_iter)
    loss_path <- numeric(n_iter)
    
    for(iter in 1:n_iter) {
      # Compute hinge loss gradient for the entire dataset
      margins <- y * (X %*% w + w0)
      indicator <- as.numeric(ifelse(margins < 1, 1, 0))
      # Compute gradient
      gradient_w <- -colSums((indicator * pmax(0, 1-margins) * y) * X) / n_samples + lambda * w
      gradient_w0 <- -sum(indicator * pmax(0, 1-margins) * y) / n_samples
      
      # Update weights and bias
      w <- w - learning_rate * gradient_w
      w0 <- w0 - learning_rate * gradient_w0
      
      # Store current solution
      w_path[iter,] <- w
      w0_path[iter] <- w0
      
      hinge_losses <- cbind(0, 1-margins)
      loss_path[iter] <- sum((apply(hinge_losses, 1, max))^2)/(2*n_samples) + (lambda/2) * sum(w^2)
    }
    
    list(w = w, w0 = w0, w_path = w_path, w0_path = w0_path, loss_path = loss_path)
}

# Find true parameter using gradient descent
model_deterministic <- gd_l2svm(X, y, learning_rate = 0.1, n_iter = 1000, lambda = 0)
true_minimizer <- model_deterministic$w
true_minimizer  <- c(0.604222258,  0.599488112, -0.004622991, -0.598398798, -0.600957343)

true_target <- true_minimizer[target_coef_index]

cat("True target parameter (gradient descent):", round(true_target, 6), "\n")

cat("=== SGD L2-SVM Coverage Experiment ===\n")
cat("Running", n_replications, "replications with 90% confidence intervals\n")
cat("Problem: d =", d, ", target coefficient =", target_coef_index, ", true target =", round(true_target, 4), "\n")
cat("Algorithm: m =", m, ", b =", b, ", K =", K, "\n")
cat("Learning rate: eta * t^(-a) with eta =", eta, ", a =", a, "\n\n")

# Storage for results
coverage_subrand <- logical(n_replications)
coverage_plugin <- logical(n_replications)
ci_length_subrand <- numeric(n_replications)
ci_length_plugin <- numeric(n_replications)
ci_center_subrand <- numeric(n_replications)
ci_center_plugin <- numeric(n_replications)

# Progress tracking
cat("Progress: ")

# Run the experiment
for (rep in 1:n_replications) {
  # Progress indicator
  if (rep %% 10 == 0) cat(rep, "")
  
  # Get main estimate (large sample size)
  # Momentum schedule: momentum_schedule[t] = pmax(1 - momentum_coef * learning_rates[t], 0)
  # With momentum_coef = 1 and learning_rates[t] = 0.4 * t^(-0.7):
  # momentum_schedule[t] = pmax(1 - 0.4 * t^(-0.7), 0)
  # 
  # Momentum update rule (Heavy Ball method):
  # v_w[t] = momentum_schedule[t] * v_w[t-1] + learning_rates[t] * gradient_w[t]
  # w[t] = w[t-1] - learning_rates[t] * v_w[t]
  # note that inside sgd_l2svm, the momentum sgd is not formalized as the standard momentum update rule in computer science literature,
  # but rather as a update rule used in statistics literature that gurantee the convergence rate to be square root of the step size.
  main_result <- sgd_l2svm(learning_rates_m, m, data_gen, d, lambda = lambda, use_momentum = TRUE, momentum_coef = 1)
  theta_m <- main_result$w[target_coef_index]
  
  # Get reduced-size estimates (small sample size, repeated K times)
  theta_b_list <- list()
  for (k in 1:K) {
    # Same momentum schedule as main estimate for consistency:
    # momentum_schedule[t] = pmax(1 - 0.4 * t^(-0.7), 0)
    # This ensures both main and reduced-size estimates use identical algorithmic setup
    sub_result <- sgd_l2svm(learning_rates_b, b, data_gen, d, lambda = lambda, use_momentum = TRUE, momentum_coef = 1)
    theta_b_list[[k]] <- sub_result$w[target_coef_index]
  }
  
  # Set convergence parameters for SGD L2-SVM
  # For SGD with polynomial learning rate eta * t^(-a), convergence rate is t^(a/2)
  tau_m <- m^(a/2)
  tau_b <- b^(a/2)
  
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
  
  # Store CI lengths and centers
  ci_length_subrand[rep] <- diff(result_subrand$confidence_interval)
  ci_length_plugin[rep] <- diff(result_plugin$confidence_interval)
  ci_center_subrand[rep] <- result_subrand$point_estimate
  ci_center_plugin[rep] <- result_plugin$point_estimate
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


