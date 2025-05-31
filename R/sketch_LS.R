#' I.I.D. Random Sketching Estimator
#'
#' Performs I.I.D. random sketching for least squares regression using various
#' random distributions.
#'
#' @param X Design matrix (n x p)
#' @param y Response vector (n x 1)
#' @param m Sketch size (number of rows in sketched matrix)
#' @param c Linear combination vector (p x 1)
#' @param type Distribution type: 1=Gaussian, 2=Three-point, 3=Sparse three-point (default=1)
#' @param partial Whether to use partial sketching (0=full sketching, 1=partial sketching)
#'
#' @return List containing:
#' \item{sketched_ls_sol}{Sketched least squares solution}
#' \item{linear_combination}{c'*sketched_ls_sol}
#' \item{sketched_data}{Sketched design matrix and response}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 100; p <- 5; m <- 30
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X %*% c(1, -0.5, 0.3, 0.8, -0.2) + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)  # sum of coordinates of solution
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Perform I.I.D. sketching
#' result <- ske_iid(X, y, m, c_vec)
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("Sketched estimate:", round(result$linear_combination, 3), "\\n")
#'
#' @export
ske_iid <- function(X, y, m, c, type = 1, partial = 0) {
  n <- nrow(X)
  
  # Generate sketching matrix S based on distribution type
  if (type == 1) {
    S <- matrix(rnorm(m * n), m) / sqrt(m)
  } else if (type == 2) {
    S <- matrix(sampleDist(m * n), m) / sqrt(m)
  } else if (type == 3) {
    S <- matrix(sampleDistsparse(m * n), m) / sqrt(m)
  }
  
  # Apply sketching
  SX <- S %*% X
  Sy <- S %*% y
  
  # Compute coefficients
  M <- solve(t(SX) %*% SX)
  
  if (partial == 0) {
    w <- t(SX) %*% Sy
    g <- M %*% w
  } else {
    g <- M %*% (t(X) %*% y)
  }
  
  # Return results
  return(list(
    sketched_ls_sol = as.vector(g), 
    linear_combination = sum(c * g), 
    sketched_data = cbind(SX, Sy)
  ))
}

#' Haar (Uniform Orthogonal) Sketching Estimator
#'
#' Performs Haar distributed orthogonal sketching for least squares regression.
#'
#' @param X Design matrix (n x p)
#' @param y Response vector (n x 1)
#' @param m Sketch size (number of rows in sketched matrix)
#' @param c Linear combination vector (p x 1)
#' @param partial Whether to use partial sketching (0=full sketching, 1=partial sketching)
#'
#' @return List containing:
#' \item{sketched_ls_sol}{Sketched least squares solution}
#' \item{linear_combination}{c'*sketched_ls_sol}
#' \item{sketched_data}{Sketched design matrix and response}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 100; p <- 5; m <- 30
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X %*% c(1, -0.5, 0.3, 0.8, -0.2) + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Perform Haar sketching
#' result <- ske_haar(X, y, m, c_vec)
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("Haar estimate:", round(result$linear_combination, 3), "\\n")
#'
#' @export
ske_haar <- function(X, y, m, c, partial = 0) {
  n <- nrow(X)
  
  # Generate Haar distributed orthogonal matrix
  S <- Generate_Haar(m, n)
  
  # Apply sketching
  SX <- S %*% X
  Sy <- S %*% y
  
  # Compute coefficients
  M <- solve(t(SX) %*% SX)
  
  if (partial == 0) {
    w <- t(SX) %*% Sy
    g <- M %*% w
  } else {
    g <- M %*% (t(X) %*% y)
  }
  
  # Return results
  return(list(
    sketched_ls_sol = as.vector(g), 
    linear_combination = sum(c * g), 
    sketched_data = cbind(SX, Sy)
  ))
}

#' Hadamard (SRHT) Sketching Estimator
#'
#' Performs Subsampled Randomized Hadamard Transform (SRHT) sketching for 
#' least squares regression.
#'
#' @param X Design matrix (n x p)
#' @param y Response vector (n x 1)
#' @param m Sketch size (number of rows in sketched matrix)
#' @param c Linear combination vector (p x 1)
#' @param partial Whether to use partial sketching (0=full sketching, 1=partial sketching)
#'
#' @return List containing:
#' \item{sketched_ls_sol}{Sketched least squares solution}
#' \item{linear_combination}{c'*sketched_ls_sol}
#' \item{sketched_data}{Sketched design matrix and response}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 200; p <- 5; m <- 50
#' X <- matrix(rnorm(n * p), n, p)
#' true_beta <- c(1, -0.5, 0.3, 0.8, -0.2)
#' y <- X %*% true_beta + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Perform Hadamard sketching
#' result <- ske_hadamard(X, y, m, c_vec)
#' 
#' # Compare results
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("SRHT estimate:", round(result$linear_combination, 3), "\\n")
#'
#' @export
ske_hadamard <- function(X, y, m, c, partial = 0) {
  tryCatch({
    p <- ncol(X)
    pad <- padding(X, y)
    X1 <- pad$padX; y1 <- pad$pady
    n1 <- nrow(X1)
    gamma <- m / n1
    
    # Apply Hadamard transform and random sampling
    # Retry if no rows are selected (to avoid empty matrix)
    max_tries <- 3
    for (attempt in 1:max_tries) {
      indices <- which(rbinom(n1, 1, gamma) != 0)
      if (length(indices) >= p) {  # Need at least p rows for solution
        break
      }
      if (attempt == max_tries) {
        # Fallback: select exactly m rows deterministically
        indices <- sample(n1, min(m, n1))
      }
    }
    
    SXy <- apply(sample(c(1, -1), n1, replace = TRUE, prob = c(0.5, 0.5)) * cbind(X1, y1), 2, phangorn::fhm)[indices, ] / sqrt(m)
    
    if (partial == 0) {
      g <- qr.solve(SXy[, 1:p], SXy[, p + 1])
    } else {
      SX <- SXy[, 1:p]; M <- solve(t(SX) %*% SX); g <- M %*% (t(X) %*% y)
    }
    
    return(list(
      sketched_ls_sol = as.vector(g), 
      linear_combination = sum(c * g), 
      sketched_data = SXy
    ))
  }, error = function(e) {
    # If anything fails, print error and return NULL values
    cat("Error in ske_hadamard:", e$message, "\n")
    return(list(
      sketched_ls_sol = rep(NA, ncol(X)), 
      linear_combination = NA, 
      sketched_data = matrix(NA, 1, ncol(X) + 1)
    ))
  })
}


#' Iterative Sketching Algorithm
#'
#' Performs iterative sketching for least squares regression, which can be
#' useful for improving accuracy through multiple iterations.
#'
#' @param X Design matrix (n x p)
#' @param y Response vector (n x 1)
#' @param c Linear combination vector (p x 1)
#' @param m Sketch size per iteration
#' @param iterations Number of iterations (default = 5)
#' @param refresh_sketch Whether to refresh sketching matrix each iteration (default = TRUE)
#'
#' @return List containing:
#' \item{final_sketched_ls_sol}{Final sketched solution}
#' \item{final_linear_combination}{Final c'*beta estimate}
#' \item{iteration_estimates}{Estimates from each iteration}
#' \item{linear_combinations}{Linear combinations from each iteration}
#'
#' @examples
#' # Generate sample data
#' # TARGET: True OLS solution (what we'd get with full data)
#' set.seed(123)
#' n <- 5000; p <- 10; m <- 500
#' X <- matrix(rnorm(n * p), n, p)
#' true_beta <- rnorm(p)
#' y <- X %*% true_beta + rnorm(n, sd = 0.1)
#' c_vec <- rep(1, p)
#'
#' # Calculate true target: OLS solution with full data
#' ols_beta <- qr.solve(X, y)
#' true_target <- sum(c_vec * ols_beta)
#'
#' # Iterative sketching
#' result <- ite_ske(X, y, c_vec, m, iterations = 5)
#' cat("True target (OLS):", round(true_target, 3), "\\n")
#' cat("Final estimate:", round(result$final_linear_combination, 3), "\\n")
#' cat("Iteration estimates:", round(result$linear_combinations, 3), "\\n")
#'
#' @references
#' Pilanci, M. & Wainwright, M. J. (2016). Iterative Hessian sketch: Fast and 
#' accurate solution approximation for constrained least-squares. Journal of 
#' Machine Learning Research, 17(53), 1-38.
#'
#' @export
ite_ske <- function(X, y, c, m, iterations = 5, refresh_sketch = TRUE) {
  n <- nrow(X)
  p <- ncol(X)
  
  Xty <- t(X) %*% y
  beta_history <- matrix(0, p, iterations + 1)
  
  if (!refresh_sketch) {
    # Generate sketching matrix once
    S <- matrix(rnorm(m * n), m) / sqrt(m)
    SX <- S %*% X
    M <- solve(t(SX) %*% SX)
  }
  
  for (i in 2:(iterations + 1)) {
    # Current estimate
    X_beta <- X %*% beta_history[, i - 1]
    Xt_X_beta <- t(X) %*% X_beta
    
    if (refresh_sketch) {
      # Generate new sketching matrix
      S <- matrix(rnorm(m * n), m) / sqrt(m)
      SX <- S %*% X
      M <- solve(t(SX) %*% SX)
    }
    
    # Update estimate
    beta_history[, i] <- M %*% (Xty - Xt_X_beta) + beta_history[, i - 1]
  }
  
  # Return results
  final_estimates <- beta_history[, 2:(iterations + 1)]
  linear_combinations <- as.numeric(t(c) %*% final_estimates)
  
  return(list(
    final_sketched_ls_sol = as.vector(final_estimates[, iterations]),
    final_linear_combination = linear_combinations[iterations],
    iteration_estimates = final_estimates,
    linear_combinations = linear_combinations
  ))
}




# Helper functions

sampleDist <- function(n) {
  sample(c(-1, 0, 1), n, replace = TRUE, prob = c(1/6, 2/3, 1/6)) * sqrt(3)
}

sampleDistsparse <- function(n) {
  sample(c(-1, 0, 1), n, replace = TRUE, prob = c(1/20, 9/10, 1/20)) * sqrt(10)
}

Generate_Haar <- function(m, n) {
  O <- matrix(rnorm(m * n), m, n)
  S <- t(svd(O)$v) * sqrt(n / m)
  return(S)
}

padding <- function(X, y) {
  m <- nrow(X)
  if (ceiling(log(m, 2)) > log(m, 2)) {
    m1 <- floor(log(m, 2)) + 1
    padX <- rbind(X, matrix(0, 2^m1 - m, ncol(X)))
    pady <- append(y, rep(0, 2^m1 - m))
  } else {
    padX <- X
    pady <- y
  }
  return(list(padX = padX, pady = pady))
} 