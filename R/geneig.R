
#' #' generalized eigenvalue decomposition
#' #' 
#' #' compute generalized eigenvalues and eigenvectors using one of two methods
#' #' 
#' #' @param A the left hand side matrix
#' #' @param B the right hand side matrix
#' #' @param ncomp number of components to return
#' #' @param method robust or lapack (using `geigen` package)
#' #' @param ... extra args sent to underlying engine
#' #' 
#' #' @return `geneig` instance which is a subclass of `projector` with added slot for eigenvalues called `values`
#' #' @export
#' #' @examples 
#' #' 
#' #' A <- matrix(c(14, 10, 12,
#' #'               10, 12, 13,
#' #'               12, 13, 14), nrow=3, byrow=TRUE)
#' 
#' #' B <- matrix(c(48, 17, 26,
#' #'               17, 33, 32,
#' #'               26, 32, 34), nrow=3, byrow=TRUE)
#' #'               
#' #' @importFrom Matrix isDiagonal   
#' #' @import Matrix         
#' geneig <- function(A, B, ncomp, method=c("robust", "sdiag", "geigen", "primme"), ...) {
#'   method <- match.arg(method)
#'   
#'   chk::chk_equal(nrow(A), ncol(A))
#'   chk::chk_equal(nrow(B), ncol(B))
#'   chk::chk_equal(nrow(A), nrow(B))
#'  
#'   if (missing(ncomp)) {
#'     ncomp <- nrow(A)
#'   }
#'   #which <- match.arg(ordering)
#'   
#'   ret <- if (method == "robust") {
#'     if (isDiagonal(B)) {
#'       Sinv <- Matrix::Diagonal(x=1/sqrt(diag(B)))
#'       W <- Matrix::Diagonal(x=Sinv) %*% A  %*% Matrix::Diagonal(x=Sinv)
#'     } else {
#'       decomp <- eigen(B)
#'       S <- decomp$values
#'       U <- decomp$vectors
#'       S[S < 1e-8] = Inf
#'       Sinv = 1 /sqrt(S)
#'       W = Matrix::Diagonal(x=Sinv) %*% crossprod(U, (A %*% U)) %*% Matrix::Diagonal(x=Sinv)
#'     }
#'     
#'     decomp2 = eigen(W)
#'     
#'     vecs <- if (!isDiagonal(B)) {
#'       U %*% Matrix::Diagonal(x=Sinv) %*% Re(decomp2$vectors)
#'     } else {
#'       decomp2$vectors
#'     }
#'     list(vectors = vecs, values=decomp2$values)
#'   } else if (method == "sdiag") {
#'     B_decomp <- eigen(B)
#'     keep <- Re(B_decomp$values) > 1e-8
#'     Bp <- B_decomp$vectors[,keep,drop=FALSE] %*% diag(1/sqrt(B_decomp$values[keep]))
#'     Ap <- t(Bp) %*% A %*% Bp
#'     A_decomp <- eigen(Ap)
#'     keep <- Re(A_decomp$values) > 1e-8
#'     vecs <- Bp %*% A_decomp$vectors[,keep,drop=FALSE]
#'     list(vectors=vecs, values=A_decomp$values)
#'   } else if (method == "geigen") {
#'     res <- geigen(as.matrix(A),as.matrix(B))
#'     vec <- res$vectors[, nrow(res$vectors):(nrow(res$vectors)-(ncomp-1))]
#'     list(vectors=vec, values=rev(res$values)[1:ncomp])
#'   } else if (method == "primme") {
#'     res <- PRIMME::eigs_sym(A=A, B=B, NEig=ncomp,...)
#'     list(vectors=res$vectors, values=res$values)
#'   }
#'   
#'   projector(v=ret$vectors, classes="geneig",values=ret$values)
#'   
#' }



