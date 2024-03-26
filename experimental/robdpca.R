robdpca <- function(X, gamma, k, max_iter = 100, tol = 1e-6) {
  n <- nrow(X)
  d <- ncol(X)
  
  # 1. Initialize s according to the value of gamma
  s <- rep(1, n)
  #s[1:gamma] <- 0
  
  # 2. Initialize P, zeta, and prev_objective_value to zeros
  P <- matrix(0, d, k)
  zeta <- rep(0, d)
  prev_objective_value <- 0
  
  # 3. Set iteration_count to 0
  iteration_count <- 0
  converged <- FALSE
  
  while (!converged && iteration_count < max_iter) {
    # 4. Perform SVD on X(diag(s) - (s * s^T) / (1^T * s))
    SVD <- svd(X %*% (diag(s) - (s %*% t(s) / sum(s))))
    
    # 5. Update the columns of P with the k right singular vectors
    P <- SVD$v[, 1:k]
    
    # 6. Update zeta using problem 17: zeta = (X * s) / (1^T * s)
    zeta <- (X %*% s) / sum(s)
    
    # 7. Compute reconstruction errors e_i for each data point
    E <- X - zeta %*% t(rep(1, n))
    errors <- colSums((E %*% (diag(d) - P %*% t(P)))^2)
    
    # 8. Update s using problem 21:
    e_gamma <- sort(errors, decreasing = TRUE)[gamma]
    s[errors <= e_gamma] <- 1
    s[errors > e_gamma] <- 0
    
    # 9. Compute the current_objective_value using the updated P, zeta, and s
    current_objective_value <- sum(s * errors)
    
    # 10. Check convergence:
    if (abs(current_objective_value - prev_objective_value) < tol) {
      converged <- TRUE
    } else {
      prev_objective_value <- current_objective_value
    }
    
    # 11. Increment iteration_count
    iteration_count <- iteration_count + 1
  }
  
  return(list(P = P, zeta = zeta))
}
