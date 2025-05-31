# Utility Functions for randalginf Development Examples
# This file contains helper functions and memory efficient sketching
# used across various examples and coverage experiments.

# ============================================================================
# Memory Efficient Sketching
# ============================================================================

#' Memory Efficient Sketching
#'
#' Performs memory-efficient sketching for large datasets that cannot be fully 
#' loaded into RAM. This function is designed specifically for situations where 
#' the dataset is too large to fit in memory, and processes data in blocks or 
#' streaming fashion by reading directly from disk.
#'
#' @param file_path Path to CSV file containing data (last column should be y)
#' @param n Number of rows in the dataset
#' @param p Number of features (excluding response)
#' @param m Main sketch size
#' @param method Processing method: "full" or "block"
#' @param num_blocks Number of blocks for block processing (default = 10)
#' @param K Number of reduced-size sketches to generate (default = 0, no reduced sketches)
#' @param b Size of each reduced sketch (required if K > 0)
#'
#' @return List containing:
#' \item{sketch_X}{Main sketched design matrix (size m)}
#' \item{sketch_y}{Main sketched response vector (size m)}
#' \item{reduced_sketches_X}{List of K reduced sketched design matrices (each size b)}
#' \item{reduced_sketches_y}{List of K reduced sketched response vectors (each size b)}
#' \item{processing_time}{Time taken for processing}
#' \item{loading_time}{Time taken for data loading}
#'
#' @examples
#' \dontrun{
#' # Create test data
#' n <- 10000; p <- 100; m <- 500; K <- 10; b <- 200
#' set.seed(123)
#' X <- matrix(rnorm(n * p), n, p)
#' true_beta <- rnorm(p)
#' y <- X %*% true_beta + rnorm(n, sd = 0.1)
#' data <- cbind(X, y)
#' colnames(data) <- c(paste0("X", 1:p), "y")
#' write.csv(data, "test_data.csv", row.names = FALSE)
#' 
#' # Test mem_eff_ske with main sketch + reduced sketches
#' result <- mem_eff_ske("test_data.csv", n, p, m, K = K, b = b)
#' 
#' # Check results
#' cat("Main sketch dimensions:", dim(result$sketch_X), "\n")
#' cat("Number of reduced sketches:", length(result$reduced_sketches_X), "\n")
#' cat("Reduced sketch dimensions:", dim(result$reduced_sketches_X[[1]]), "\n")
#' 
#' # Use with general inference
#' c_vec <- rep(1, p)  # sum of coordinates
#' theta_m <- sum(c_vec * qr.solve(result$sketch_X, result$sketch_y))
#' theta_b_list <- lapply(1:K, function(k) {
#'   sum(c_vec * qr.solve(result$reduced_sketches_X[[k]], result$reduced_sketches_y[[k]]))
#' })
#' 
#' # Apply general inference
#' # inference_result <- general_inference(theta_m, theta_b_list, tau_m, tau_b, method = "subrand")
#' 
#' # Cleanup
#' file.remove("test_data.csv")
#' }
#'
#' @export
mem_eff_ske <- function(file_path, n, p, m, method = "block", num_blocks = 10, K = 0, b = NULL) {
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("data.table package is required for memory efficient sketching")
  }
  if (!requireNamespace("phangorn", quietly = TRUE)) {
    stop("phangorn package is required for Hadamard transform")
  }
  
  # Validate parameters
  if (K > 0 && is.null(b)) {
    stop("Parameter 'b' must be specified when K > 0")
  }
  if (K > 0 && b >= m) {
    stop("Reduced sketch size 'b' must be smaller than main sketch size 'm'")
  }
  
  gc()
  start_time_total <- Sys.time()
  loading_time <- 0
  
  # Generate selection lists and diagonal matrices for main sketch
  select_list_m <- which(rbinom(n, 1, m/n) != 0)
  D_list_m <- sample(c(1,-1), n, replace=TRUE)
  sketch_m <- matrix(0, length(select_list_m), p)
  sketch_y_m <- numeric(length(select_list_m))
  
  # Generate selection lists and diagonal matrices for reduced sketches
  reduced_sketches_X <- list()
  reduced_sketches_y <- list()
  select_lists_b <- list()
  D_lists_b <- list()
  
  if (K > 0) {
    for (k in 1:K) {
      select_lists_b[[k]] <- which(rbinom(n, 1, b/n) != 0)
      D_lists_b[[k]] <- sample(c(1,-1), n, replace=TRUE)
      reduced_sketches_X[[k]] <- matrix(0, length(select_lists_b[[k]]), p)
      reduced_sketches_y[[k]] <- numeric(length(select_lists_b[[k]]))
    }
  }
  
  # Load y vector once (last column)
  start_load <- Sys.time()
  y <- as.numeric(data.table::fread(file_path, select = p + 1, header = TRUE)[[1]])
  end_load <- Sys.time()
  loading_time <- loading_time + as.numeric(difftime(end_load, start_load, units = "secs"))
  
  # Sketch y vector for main sketch
  sketch_y_m <- SRHT(select_list_m, D_list_m, y)
  
  # Sketch y vector for reduced sketches
  if (K > 0) {
    for (k in 1:K) {
      reduced_sketches_y[[k]] <- SRHT(select_lists_b[[k]], D_lists_b[[k]], y)
    }
  }
  
  if (method == "full") {
    start_load <- Sys.time()
    X <- as.matrix(data.table::fread(file_path, select = 1:p))
    end_load <- Sys.time()
    loading_time <- loading_time + as.numeric(difftime(end_load, start_load, units = "secs"))
    
    # Process main sketch
    for(j in 1:p) {
      sketch_m[,j] <- SRHT(select_list_m, D_list_m, X[,j])
    }
    
    # Process reduced sketches
    if (K > 0) {
      for (k in 1:K) {
        for(j in 1:p) {
          reduced_sketches_X[[k]][,j] <- SRHT(select_lists_b[[k]], D_lists_b[[k]], X[,j])
        }
      }
    }
    
    rm(X)
    gc()
  } else if (method == "block") {
    # Calculate block size for X columns only
    block_size <- ceiling(p / num_blocks)
    
    for(i in 1:num_blocks) {
      start_col <- ((i-1) * block_size + 1)
      end_col <- min(i * block_size, p)
      cols <- start_col:end_col
      
      start_load <- Sys.time()
      block_data <- data.table::fread(file_path, 
                        select = cols,
                        header = TRUE,
                        data.table = FALSE)
      end_load <- Sys.time()
      loading_time <- loading_time + as.numeric(difftime(end_load, start_load, units = "secs"))
      
      # Process each column in the block for main sketch
      for(j in 1:ncol(block_data)) {
        col_idx <- start_col + j - 1
        sketch_m[, col_idx] <- SRHT(select_list_m, D_list_m, block_data[,j])
      }
      
      # Process each column in the block for reduced sketches
      if (K > 0) {
        for (k in 1:K) {
          for(j in 1:ncol(block_data)) {
            col_idx <- start_col + j - 1
            reduced_sketches_X[[k]][, col_idx] <- SRHT(select_lists_b[[k]], D_lists_b[[k]], block_data[,j])
          }
        }
      }
      
      rm(block_data)
      gc()
    }
  }
  
  end_time_total <- Sys.time() 
  
  result <- list(
    sketch_X = sketch_m,
    sketch_y = sketch_y_m,
    processing_time = as.numeric(difftime(end_time_total, start_time_total, units = "secs")),
    loading_time = loading_time
  )
  
  # Add reduced sketches to result if K > 0
  if (K > 0) {
    result$reduced_sketches_X <- reduced_sketches_X
    result$reduced_sketches_y <- reduced_sketches_y
  }
  
  return(result)
}

#' Helper function for SRHT (Subsampled Randomized Hadamard Transform)
#'
#' @param select Selection indices
#' @param D Diagonal matrix entries
#' @param a Input vector
#'
#' @return Sketched vector
SRHT <- function(select, D, a) {
  Da <- D * a
  Sa <- phangorn::fhm(Da)[select]  
  return(Sa)
} 
