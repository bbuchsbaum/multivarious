
#' generalized eigenvalue decomposition
#' 
#' compute generalized eigenvalues and eigenvectors using one of two methods
#' 
#' @param A the left hand side matrix
#' @param B the right hand side matrix
#' @param ncomp number of components to return
#' @param ordering the ordering of the eigenvalues/vectors (LM = largest magnitude first, SM = smallest magnitude first)
#' @param method robust or lapack (using `geigen` package)
#' 
#' @return `geneig` instance whcih is a subclass of `projector` with added slot for eigenvalues called `values`
#' @export
#' @examples 
#' 
#' A <- matrix(c(14, 10, 12,
#'               10, 12, 13,
#'               12, 13, 14), nrow=3, byrow=TRUE)

#' B <- matrix(c(48, 17, 26,
#'               17, 33, 32,
#'               26, 32, 34), nrow=3, byrow=TRUE)
#'               
#' @importFrom Matrix isDiagonal            
geneig <- function(A, B, ncomp=2, method=c("robust", "sdiag", "lapack")) {
  method <- match.arg(method)
  #which <- match.arg(ordering)
  
  ret <- if (method == "robust") {
    if (isDiagonal(B)) {
      Sinv <- Matrix::Diagonal(1/sqrt(diag(B)))
      W <- Matrix::Diagonal(x=Sinv) %*% A  %*% Matrix::Diagonal(x=Sinv)
    } else {
      decomp <- eigen(B)
      S <- decomp$values
      U <- decomp$vectors
      S[S < 1e-8] = Inf
      Sinv = 1 /sqrt(S)
      W = Matrix::Diagonal(x=Sinv) %*% crossprod(U, (A %*% U)) %*% Matrix::Diagonal(x=Sinv)
    }
    
    decomp2 = RSpectra::eigs(W, k=ncomp)
    
    vecs <- if (!isDiagonal(B)) {
      U %*% Matrix::Diagonal(x=Sinv) %*% decomp2$vectors
    } else {
      decomp2$vectors
    }
    list(vectors = vecs, values=decomp2$values)
  } else if (method == "sdiag") {
    B_decomp <- eigen(B)
    Bp <- B_decomp$vectors %*% diag(1/sqrt(B_decomp$values))
    Ap <- t(Bp) %*% A %*% Bp
    A_decomp <- RSpectra::eigs(Ap, k=ncomp)
    vecs <- Bp %*% A_decomp$vectors
    list(vectors=vecs, values=A_decomp$values)
  } else {
    res <- geigen(as.matrix(A),as.matrix(B))
   
    vec <- res$vectors[, nrow(res$vectors):(nrow(res$vectors)-(ncomp-1))]
    list(vectors=vec, values=rev(res$values)[1:ncomp])
  }
  
  projector(v=ret$vectors, classes="geneig", ordering=ordering, values=ret$values)
  
}



