#' Sigmoid Function
#'
#' Computes the sigmoid function.
#'
#' @param z Input vector or scalar
#'
#' @return Sigmoid transformation of z
#' @export
#' @keywords internal
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

#' Generate Logistic Regression Data
#'
#' Generates (X, y) pairs for logistic regression where y follows Bernoulli distribution
#' with probability determined by the logistic function of X'*theta.
#'
#' @param theta True coefficient vector (parameter of the logistic model)
#' @param d Dimension of features (default = length(theta))
#'
#' @return List containing:
#' \item{X}{Design vector (length d)}
#' \item{y}{Binary response (scalar)}
#'
#' @examples
#' # Generate logistic regression data point
#' set.seed(123)
#' theta <- c(1, -0.5, 0.3)
#' data <- generate_logistic_data(theta)
#' print(c(data$X, data$y))
#'
#' @keywords internal
generate_logistic_data <- function(theta, d = length(theta)) {
  # Generate design vector from standard normal (faster vector approach)
  X <- rnorm(d)
  
  # Generate binary response using logistic model (faster vector operations)
  y <- rbinom(1, 1, sigmoid(sum(X * theta)))
  
  return(list(X = X, y = y))
}

#' Create Data Generator for Logistic Regression
#'
#' Creates a data generation function for logistic regression based on specified
#' true coefficients theta. This is a convenience function for creating data
#' generators to use with inference functions.
#'
#' @param theta True coefficient vector (parameter of the logistic model)
#'
#' @return Function that generates single data points when called
#'
#' @examples
#' # Create a data generator
#' theta <- c(1, -0.5, 0.3)
#' data_gen <- create_logistic_generator(theta)
#' 
#' # Generate some data points
#' point1 <- data_gen()
#' point2 <- data_gen()
#' 
#' @keywords internal
create_logistic_generator <- function(theta) {
  function() generate_logistic_data(theta, d = length(theta))
}

#' PRA Stochastic Gradient Descent for Logistic Regression
#'
#' Performs stochastic gradient descent with Polyak-Ruppert averaging for 
#' logistic regression. This function uses a provided data generation function to 
#' simulate streaming logistic regression.
#'
#' @param learning_rate Vector of learning rates for each iteration
#' @param num_iterations Number of SGD iterations
#' @param data_generator Function that generates data points (should return list with X and y)
#' @param dimension Dimension of the parameter space
#' @param average Whether to use PRA (default = TRUE)
#' @param burnin Number of burn-in iterations (default = round(num_iterations / 1000))
#' @param start_point Starting point for optimization (default = NULL, uses zero)
#'
#' @return List containing:
#' \item{final_weights}{Final weight vector}
#' \item{averaged_weights}{PRA weight vector (if average = TRUE)}
#' \item{iterations}{Number of iterations performed}
#'
#' @examples
#' # Define a data generation function based on true population parameters
#' # TARGET: True regression parameter (population parameter)
#' set.seed(123)
#' d <- 5
#' theta_true <- seq(0, d, d/4) / d  # True population parameters
#' data_gen <- create_logistic_generator(theta_true)
#' 
#' num_iter <- 1000
#' learning_rates <- 0.5 * (1:num_iter)^(-0.505)
#'
#' # Run PRA SGD
#' result <- PRA_logistic(learning_rates, num_iter, data_gen, d)
#' cat("True target (population param):", round(theta_true, 3), "\n")
#' cat("Estimated coefficients:", round(result$averaged_weights, 3), "\n")
#'
#' @export
PRA_logistic <- function(learning_rate, num_iterations, data_generator, dimension,
                        average = TRUE, burnin = NULL, start_point = NULL) {
  
  d <- dimension
  
  # Set default burn-in
  if (is.null(burnin)) {
    burnin <- round(num_iterations / 1000)
  }
  
  # Initialize starting point
  if (is.null(start_point)) {
    start_point <- rep(0, d)
    
    # Burn-in phase
    if (burnin > 0) {
      for (i in 1:burnin) {
        data_point <- data_generator()
        X_batch <- data_point$X
        y_batch <- data_point$y
        z <- sum(X_batch * start_point)  # Vector dot product (faster)
        p <- sigmoid(z)
        grad <- X_batch * (p - y_batch)  # Element-wise multiplication (faster)
        lr <- if (length(learning_rate) >= i) learning_rate[i] else learning_rate[1]
        start_point <- start_point - lr * grad
      }
    }
  }
  
  w <- start_point
  avg_w <- start_point
  
  # Main SGD iterations
  for (i in 1:num_iterations) {
    # Generate mini-batch data online
    data_point <- data_generator()
    X_batch <- data_point$X
    y_batch <- data_point$y
    
    # Compute gradient using vector operations (faster)
    z <- sum(X_batch * w)  # Vector dot product
    p <- sigmoid(z)
    grad <- X_batch * (p - y_batch)  # Element-wise multiplication
    
    # Update weights
    lr <- if (length(learning_rate) >= i) learning_rate[i] else learning_rate[length(learning_rate)]
    w <- w - lr * grad
    
    # PRA (Polyak-Ruppert averaging)
    if (average) {
      avg_w <- (avg_w * (i - 1) + w) / i
    }
  }
  
  return(list(
    final_weights = as.vector(w),
    averaged_weights = as.vector(avg_w),
    iterations = num_iterations
  ))
} 
