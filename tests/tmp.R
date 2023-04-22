
# gen_preproc <- function(n, m) {
#   X <- matrix(rnorm(m*n), n,m)
#   p <- center()
#   proc <- prep(p)
#   print(pryr::object_size(proc))
#   X0 <- init_transform(proc, X)
#   print(pryr::object_size(proc))
#   rm(X)
#   rm(X0)
#   proc
# }
# 
# library(pryr)
# 
# p1 <- gen_preproc(100,1000)
# p2 <- gen_preproc(10,1000)
# p3 <- gen_preproc(1,1000)
# 
# pryr::object_size(p1[[2]])
# pryr::object_size(p2[[2]])
# pryr::object_size(p3[[2]])
# 
# 
# X <- matrix(rnorm(100*1000), 100,1000)
# pcres1 <- pca(X, ncomp=50)
# 
# X <- matrix(rnorm(10*1000), 10,1000)
# pcres1 <- pca(X, ncomp=50)
