# Calculate Principal Angles Between Subspaces

Computes the principal angles between two subspaces defined by the
columns of two orthonormal matrices Q1 and Q2.

## Usage

``` r
prinang(Q1, Q2)
```

## Arguments

- Q1:

  An n x p matrix whose columns form an orthonormal basis for the first
  subspace.

- Q2:

  An n x q matrix whose columns form an orthonormal basis for the second
  subspace.

## Value

A numeric vector containing the principal angles in radians, sorted in
ascending order. The number of angles is `min(p, q)`.

## Examples

``` r
# Example: Angle between xy-plane and a plane rotated 45 degrees around x-axis
Q1 <- cbind(c(1,0,0), c(0,1,0)) # xy-plane basis
theta <- pi/4
R <- matrix(c(1, 0, 0, 0, cos(theta), sin(theta), 0, -sin(theta), cos(theta)), 3, 3)
Q2 <- R %*% Q1 # Rotated basis
angles_rad <- prinang(Q1, Q2)
angles_deg <- angles_rad * 180 / pi
print(angles_deg) # Should be approximately 0 and 45 degrees
#> [1]  0 45

# Example with PCA loadings (after ensuring orthonormality if needed)
# Assuming pca1$v and pca2$v are loading matrices (variables x components)
# Orthonormalize them first if they are not already (e.g., from standard SVD)
# Q1 <- qr.Q(qr(pca1$v[, 1:3]))
# Q2 <- qr.Q(qr(pca2$v[, 1:3]))
# prinang(Q1, Q2)
```
