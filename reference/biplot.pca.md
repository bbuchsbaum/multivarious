# Biplot for PCA Objects (Enhanced with ggrepel)

Creates a 2D biplot for a `pca` object, using ggplot2 and ggrepel to
show both sample scores (observations) and variable loadings (arrows).

## Usage

``` r
# S3 method for class 'pca'
biplot(
  x,
  y = NULL,
  dims = c(1, 2),
  scale_arrows = 2,
  alpha_points = 0.6,
  point_size = 2,
  point_labels = NULL,
  var_labels = NULL,
  arrow_color = "red",
  text_color = "red",
  repel_points = TRUE,
  repel_vars = FALSE,
  ...
)
```

## Arguments

- x:

  A `pca` object returned by
  [`pca`](https://bbuchsbaum.github.io/multivarious/reference/pca.md).

- y:

  (ignored) Placeholder to match `biplot(x, y, ...)` signature.

- dims:

  A length-2 integer vector specifying which principal components to
  plot on the x and y axes. Defaults to `c(1, 2)`.

- scale_arrows:

  A numeric factor to scale the variable loadings (arrows). Default is
  2.

- alpha_points:

  Transparency level for the sample points. Default is 0.6.

- point_size:

  Size for the sample points. Default is 2.

- point_labels:

  Optional character vector of labels for the sample points. If `NULL`,
  rownames of the scores matrix are used if available; otherwise numeric
  indices.

- var_labels:

  Optional character vector of variable names (columns in the original
  data). If `NULL`, rownames of `x\$v` are used if available; otherwise
  "Var1", "Var2", etc.

- arrow_color:

  Color for the loading arrows. Default is "red".

- text_color:

  Color for the variable label text. Default is "red".

- repel_points:

  Logical; if TRUE, repel sample labels using `geom_text_repel`. Default
  is `TRUE`.

- repel_vars:

  Logical; if TRUE, repel variable labels using `geom_text_repel`.
  Default is `FALSE`.

- ...:

  Additional arguments passed on to `ggplot2` or `ggrepel` functions (if
  needed).

## Value

A `ggplot` object.

## Details

This function constructs a scatterplot of the PCA scores (observations)
on two chosen components and overlays arrows for the loadings
(variables). The arrow length and direction indicate how each variable
contributes to those principal components. You can control arrow scaling
with `scale_arrows`.

If your `pca` object includes an `$explained_variance` field (e.g.,
proportion of variance per component), those values will appear in the
axis labels. Otherwise, the axes are labeled simply as "PC1", "PC2",
etc.

**Note**: If you do not have ggrepel installed, you can set
`repel_points=FALSE` and `repel_vars=FALSE`, or install ggrepel.

## Examples

``` r
# \donttest{
data(iris)
X <- as.matrix(iris[,1:4])
pca_res <- pca(X, ncomp=2)

# Enhanced biplot with repelled text
biplot(pca_res, repel_points=TRUE, repel_vars=TRUE)

# }
```
