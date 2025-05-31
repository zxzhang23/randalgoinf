#' Generates (X, y) pairs for L2-SVM where features come from a mixture of two
#' multivariate normal distributions with specified means and covariance.
#'
#' @param mean_1 Mean vector for first component (default = c(1, 1, 1, 0, 0))
#' @param mean_2 Mean vector for second component (default = c(0, 0, 1, 1, 1))
#' @param mixture_weights Mixture weights (default = c(0.2, 0.8))
#' @param Sigma Covariance matrix (default = diag(rep(0.5, 5)))
#'
#' @return List containing:
#' \item{X}{Design matrix (1 x d)}
#' \item{y}{Binary response (scalar) with value in {-1, +1}}
#'
#' @examples
#' # Generate L2-SVM data point
#' set.seed(123)
#' data <- generate_l2svm_data()
#' print(cbind(data$X, data$y))
#'
#' @keywords internal
generate_l2svm_data <- function(mean_1 = c(1, 1, 1, 0, 0),
                               mean_2 = c(0, 0, 1, 1, 1),
                               mixture_weights = c(0.2, 0.8),
                               Sigma = diag(rep(0.5, 5))) {
  
  d <- length(mean_1)
  component <- sample(1:2, 1, prob = mixture_weights)
  
  if (component == 1) {
    X <- mvtnorm::rmvnorm(1, mean = mean_1, sigma = Sigma)
    y <- 1
  } else {
    X <- mvtnorm::rmvnorm(1, mean = mean_2, sigma = Sigma)
    y <- -1
  }
  
  return(list(X = X, y = y))
}

#' Create Data Generator for L2-SVM
#'
#' Creates a data generation function for L2-SVM based on specified mixture
#' distribution parameters. This is a convenience function for creating data
#' generators to use with SGD functions.
#'
#' @param mean_1 Mean vector for first component
#' @param mean_2 Mean vector for second component
#' @param mixture_weights Mixture weights
#' @param Sigma Covariance matrix
#'
#' @return Function that generates single data points when called
#'
#' @examples
#' # Create a data generator
#' data_gen <- create_l2svm_generator()
#' 
#' # Generate some data points
#' point1 <- data_gen()
#' point2 <- data_gen()
#' 
#' @keywords internal
create_l2svm_generator <- function(mean_1 = c(1, 1, 1, 0, 0),
                                  mean_2 = c(0, 0, 1, 1, 1),
                                  mixture_weights = c(0.2, 0.8),
                                  Sigma = diag(rep(0.5, 5))) {
  function() generate_l2svm_data(mean_1, mean_2, mixture_weights, Sigma)
}

#' SGD for L2-SVM with Optional Momentum
#'
#' Performs stochastic gradient descent for L2-Support Vector Machine with
#' optional momentum (heavy ball method). Uses online data generation.
#'
#' @param learning_rates Vector of learning rates for each iteration
#' @param num_iterations Number of SGD iterations
#' @param data_generator Function that generates data points (should return list with X and y)
#' @param dimension Dimension of the parameter space
#' @param lambda L2 regularization parameter (default = 0)
#' @param use_momentum Whether to use momentum (default = FALSE)
#' @param momentum_schedule Vector of momentum coefficients (if use_momentum = TRUE and not NULL, uses this; otherwise calculates from momentum_coef)
#' @param momentum_coef Momentum coefficient for automatic schedule calculation (default = 1)
#' @param w_start Starting point for weights (default = NULL, uses zero)
#' @param w0_start Starting point for bias (default = NULL, uses zero)
#' @param burnin Number of burn-in iterations (default = 0)
#'
#' @return List containing:
#' \item{w}{Final weight vector}
#' \item{w0}{Final bias term}
#' \item{w_path}{Weight trajectory (num_iterations x dimension)}
#' \item{w0_path}{Bias trajectory (num_iterations x 1)}
#' \item{loss_path}{Loss trajectory}
#'
#' @examples
#' # Set up L2-SVM problem
#' set.seed(123)
#' data_gen <- create_l2svm_generator()
#' 
#' num_iter <- 1000
#' learning_rates <- 0.4 * (1:num_iter)^(-0.6)
#'
#' # Run SGD without momentum
#' result <- sgd_l2svm(learning_rates, num_iter, data_gen, 5)
#' cat("Final weights:", round(result$w, 3), "\n")
#' cat("Final bias:", round(result$w0, 3), "\n")
#'
#' # Run SGD with momentum (automatic schedule)
#' result_momentum <- sgd_l2svm(learning_rates, num_iter, data_gen, 5,
#'                             use_momentum = TRUE, momentum_coef = 1)
#' cat("Final weights (momentum):", round(result_momentum$w, 3), "\n")
#'
#' @export
sgd_l2svm <- function(learning_rates, num_iterations, data_generator, dimension,
                     lambda = 0, use_momentum = FALSE, momentum_schedule = NULL,
                     momentum_coef = 1, w_start = NULL, w0_start = NULL, burnin = 0) {
  
  d <- dimension
  
  # Initialize starting points
  if (is.null(w_start)) {
    w_start <- rep(0, d)
  }
  if (is.null(w0_start)) {
    w0_start <- 0
  }
  
  w <- w_start
  w0 <- w0_start
  
  # Momentum variables
  if (use_momentum) {
    v_w <- rep(0, d)
    v_w0 <- 0
    if (is.null(momentum_schedule)) {
      # Calculate momentum schedule using the user's formula: pmax(1 - momentum_coef * learning_rates, 0)
      momentum_schedule <- pmax(1 - momentum_coef * learning_rates, 0)
    }
  }
  
  # Burn-in phase
  if (burnin > 0) {
    for (iter in 1:burnin) {
      data_point <- data_generator()
      x_i <- data_point$X[1, ]
      y_i <- data_point$y[1]
      
      margin <- y_i * (sum(w * x_i) + w0)
      indicator <- as.numeric(margin < 1)
      
      gradient_w <- -indicator * pmax(0, 1 - margin) * y_i * x_i + lambda * w
      gradient_w0 <- -indicator * pmax(0, 1 - margin) * y_i
      
      current_lr <- if (length(learning_rates) >= iter) learning_rates[iter] else learning_rates[1]
      w <- w - current_lr * gradient_w
      w0 <- w0 - current_lr * gradient_w0
    }
  }
  
  # Storage for trajectories
  w_path <- matrix(0, nrow = num_iterations, ncol = d)
  w0_path <- numeric(num_iterations)
  loss_path <- numeric(num_iterations)
  accum_hinge_loss <- 0
  
  # Main SGD iterations
  for (iter in 1:num_iterations) {
    current_lr <- if (length(learning_rates) >= iter) learning_rates[iter] else learning_rates[length(learning_rates)]
    
    # Generate data point
    data_point <- data_generator()
    x_i <- data_point$X[1, ]
    y_i <- data_point$y[1]
    
    # Compute margin and gradient
    margin <- y_i * (sum(w * x_i) + w0)
    indicator <- as.numeric(margin < 1)
    
    gradient_w <- -indicator * pmax(0, 1 - margin) * y_i * x_i + lambda * w
    gradient_w0 <- -indicator * pmax(0, 1 - margin) * y_i
    
    if (use_momentum) {
      # Update velocities for momentum  
      current_momentum <- momentum_schedule[iter]
      v_w <- current_momentum * v_w + current_lr * gradient_w
      v_w0 <- current_momentum * v_w0 + current_lr * gradient_w0    
      
      # Update parameters
      w <- w - current_lr * v_w
      w0 <- w0 - current_lr * v_w0
    } else {
      # Standard SGD update
      w <- w - current_lr * gradient_w
      w0 <- w0 - current_lr * gradient_w0
    }
    
    # Track loss
    accum_hinge_loss <- accum_hinge_loss + (max(0, 1 - margin))^2
    loss_path[iter] <- accum_hinge_loss / (2 * iter) + (lambda / 2) * sum(w^2)
    
    # Store current solution
    w_path[iter, ] <- w
    w0_path[iter] <- w0
  }
  
  return(list(
    w = as.vector(w),
    w0 = w0,
    w_path = w_path,
    w0_path = w0_path,
    loss_path = loss_path
  ))
}