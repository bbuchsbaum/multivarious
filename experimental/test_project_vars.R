# Test project_vars mathematical relationship
set.seed(123)
n <- 10
p <- 15
X <- matrix(rnorm(n*p), n, p)

# SVD decomposition
svd_res <- svd(X)
U <- svd_res$u
D <- diag(svd_res$d)
V <- svd_res$v

# Verify SVD
cat("SVD reconstruction error:", max(abs(X - U %*% D %*% t(V))), "\n")

# For centered data
X_centered <- scale(X, center=TRUE, scale=FALSE)
svd_centered <- svd(X_centered)

# The scores in PCA are U*D
scores <- svd_centered$u %*% diag(svd_centered$d)

# The loadings are V
loadings <- svd_centered$v

# Mathematical relationship:
# X_centered = U * D * V^T
# X_centered^T * U * D = V * D^2
# X_centered^T * scores = V * D^2

# So: V = X_centered^T * scores / D^2

# Verify this
V_computed <- t(X_centered) %*% scores %*% diag(1/svd_centered$d^2)
cat("V reconstruction error:", max(abs(V_computed - loadings)), "\n")

# Now what is project_vars computing?
# t(new_data) %*% sc %*% diag(1/variance)
# where variance = sdev^2 = d^2

# So project_vars computes:
# X^T * scores / d^2 = X^T * U * D / d^2 = X^T * U / d

# For centered X, this gives V
# But the test expects: project_vars * (n-1) = V

# This means project_vars = V / (n-1)
# Which implies: X^T * scores / d^2 = V / (n-1)

# Let's check the variance of scores
score_var <- apply(scores, 2, var)
cat("Score variances:", score_var[1:5], "\n")
cat("d^2 / (n-1):", svd_centered$d[1:5]^2 / (n-1), "\n")

# So scores have variance d^2/(n-1)
# Therefore: X^T * scores = V * d^2
# And: X^T * scores / d^2 = V

# But in PCA with centering, we need to account for degrees of freedom
# The correct relationship is:
# cov(X_centered, scores) = V * d / sqrt(n-1)
# Because scores = U * d have variance d^2/(n-1)

cov_computed <- cov(X_centered, scores) * (n-1) 
cat("Cov * (n-1) vs V*d error:", max(abs(cov_computed - V %*% diag(svd_centered$d))), "\n")

# So project_vars should compute:
# cov(X, scores) = t(X_centered) %*% scores / (n-1)
# = V * d^2 / (n-1)

# Therefore: project_vars = V * d^2 / (n-1) / d^2 = V / (n-1)
# And: project_vars * (n-1) = V ✓