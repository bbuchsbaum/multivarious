# tests/testthat/test_discriminant_projector.R
context("discriminant_projector")

library(multivarious)
library(MASS)          # for lda()

# -------------------------------------------------------------------------
# helper : two–class toy data ‒ clearly separated signal  +  pure noise
# -------------------------------------------------------------------------
set.seed(42)
n_per  <- 25L                      # per–class observations
p_sig  <- 2L                       # informative dimensions
p_noise<- 8L                       # noise dimensions
p      <- p_sig + p_noise

X_signal <- rbind(
  cbind(matrix(rnorm(n_per * p_sig ,  mean = -3), n_per, p_sig),
        matrix(rnorm(n_per * p_noise), n_per) ),
  cbind(matrix(rnorm(n_per * p_sig ,  mean =  3), n_per, p_sig),
        matrix(rnorm(n_per * p_noise), n_per) )
)
Y_signal <- factor(rep(c("A", "B"), each = n_per))

X_noise  <- X_signal                       # same features
Y_noise  <- factor(sample(Y_signal))       # labels unrelated to X

# -------------------------------------------------------------------------
# 1. Constructor integrity -------------------------------------------------
# -------------------------------------------------------------------------
test_that("discriminant_projector constructor stores consistent state", {

  lda_fit  <- lda(X_signal, grouping = Y_signal)
  # Initialize the default preprocessor (pass() does nothing, but follows pattern)
  preproc <- prep(pass())
  Xp <- init_transform(preproc, X_signal)
  
  dp       <- discriminant_projector(
                v      = lda_fit$scaling,
                s      = X_signal %*% lda_fit$scaling,
                sdev   = lda_fit$svd,
                preproc = preproc, # <-- Pass initialized preproc
                labels = Y_signal,
                Sigma  = lda_fit$covariance)

  # class & counts
  expect_s3_class(dp, c("discriminant_projector", "bi_projector"))
  expect_equal(dp$counts, table(Y_signal, dnn = NULL))

  # shape coherence
  expect_equal(dim(dp$v)[2], length(dp$sdev))
  expect_equal(dim(dp$s), c(length(Y_signal), dim(dp$v)[2]))
})

test_that("discriminant_projector preserves factor level order", {

  labs_custom <- factor(Y_signal, levels = c("B", "A"))
  lda_fit  <- lda(X_signal, grouping = labs_custom)
  preproc <- prep(pass())
  Xp <- init_transform(preproc, X_signal)

  dp <- discriminant_projector(
          v      = lda_fit$scaling,
          s      = X_signal %*% lda_fit$scaling,
          sdev   = lda_fit$svd,
          preproc = preproc,
          labels = labs_custom,
          Sigma  = lda_fit$covariance)

  expect_equal(levels(dp$labels), levels(labs_custom))
})

# -------------------------------------------------------------------------
# 2. Prediction engine (LDA & Euclidean) -----------------------------------
# -------------------------------------------------------------------------
test_that("predict.discriminant_projector produces sensible classes & probabilities", {

  # Initialize the default preprocessor
  preproc <- prep(pass())
  Xp <- init_transform(preproc, X_signal)
  
  dp <- {
    lda_fit <- lda(X_signal, grouping = Y_signal)
    discriminant_projector(v      = lda_fit$scaling, 
                           s      = X_signal %*% lda_fit$scaling,
                           sdev   = lda_fit$svd, 
                           preproc = preproc, # <-- Pass initialized preproc
                           labels = Y_signal,
                           Sigma  = lda_fit$covariance)
  }

  ## --- LDA method --------------------------------------------------------
  preds_lda  <- predict(dp, X_signal, method = "lda", type = "class")
  acc_lda    <- mean(preds_lda == Y_signal)
  expect_true(acc_lda > .90)                     # > 90 % on clearly separated data

  probs_lda  <- predict(dp, X_signal, method = "lda", type = "prob")
  expect_equal(dim(probs_lda), c(length(Y_signal), nlevels(Y_signal)))
  expect_true(all(abs(rowSums(probs_lda) - 1) < 1e-10))

  ## --- Euclidean method --------------------------------------------------
  preds_euc  <- predict(dp, X_signal, method = "euclid", type = "class")
  acc_euc    <- mean(preds_euc == Y_signal)
  expect_true(acc_euc > .80)                     # slightly laxer threshold

})

# -------------------------------------------------------------------------
# 3. Permutation test reacts to signal vs noise ----------------------------
# -------------------------------------------------------------------------
test_that("perm_test.discriminant_projector yields small p for signal and large p for noise", {

  # Initialize the default preprocessor
  preproc1 <- prep(pass())
  preproc2 <- prep(pass())
  initialized_proc <- init_transform(preproc1, X_signal)
  initialized_proc_noise <- init_transform(preproc2, X_noise) # Need one for noise data too
  
  ## fit on signal ---------------------------------------------------------
  lda_sig <- lda(X_signal, grouping = Y_signal)
  dp_sig  <- discriminant_projector(v      = lda_sig$scaling, 
                                    s      = X_signal %*% lda_sig$scaling,
                                    sdev   = lda_sig$svd, 
                                    preproc = preproc1, # <-- Pass initialized preproc
                                    labels = Y_signal,
                                    Sigma  = lda_sig$covariance)

  pt_sig  <- perm_test(dp_sig, X_signal, nperm = 150)
  expect_true(pt_sig$p.value < 0.05)             # evidence of separation

  ## fit on pure‑noise labels ---------------------------------------------
  lda_noise <- lda(X_noise, grouping = Y_noise)
  dp_noise  <- discriminant_projector(v      = lda_noise$scaling, 
                                      s      = X_noise %*% lda_noise$scaling,
                                      sdev   = lda_noise$svd, 
                                      preproc = preproc2, # <-- Pass initialized preproc for noise
                                      labels = Y_noise,
                                      Sigma  = lda_noise$covariance)

  pt_noise <- perm_test(dp_noise, X_noise, nperm = 150)
  expect_true(pt_noise$p.value > 0.10)           # no real separation
})


# -------------------------------------------------------------------------
# 4. Rank deficient covariance is handled via pseudo-inverse ---------------
# -------------------------------------------------------------------------
test_that("predict works when covariance is rank deficient", {

  set.seed(123)
  # create perfectly collinear feature to ensure singular covariance
  X_rd <- matrix(rnorm(50 * 2), 50, 2)
  X_rd <- cbind(X_rd, X_rd[,1])
  Y_rd <- factor(rep(c("A", "B"), each = 25))

  # Suppress expected collinearity warning
  lda_fit <- suppressWarnings(lda(X_rd, grouping = Y_rd))

  # manually supply rank deficient Sigma
  Sigma_rd <- cov(X_rd)

  preproc <- prep(pass())
  init_transform(preproc, X_rd)

  dp <- discriminant_projector(v      = lda_fit$scaling,
                               s      = X_rd %*% lda_fit$scaling,
                               sdev   = lda_fit$svd,
                               preproc = preproc,
                               labels = Y_rd,
                               Sigma  = Sigma_rd)

  preds <- predict(dp, X_rd, method = "lda", type = "class")
  expect_length(preds, length(Y_rd))
  expect_true(all(levels(preds) == levels(Y_rd)))
})
