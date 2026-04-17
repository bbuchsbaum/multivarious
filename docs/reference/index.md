# Package index

## Model construction

Functions that compute multivariate decompositions

- [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  : Principal Components Analysis (PCA)

- [`plsc()`](https://bbuchsbaum.github.io/multivarious/reference/plsc.md)
  : Partial Least Squares Correlation (PLSC)

- [`cca()`](https://bbuchsbaum.github.io/multivarious/reference/cca.md)
  : Canonical Correlation Analysis (CCA)

- [`svd_wrapper()`](https://bbuchsbaum.github.io/multivarious/reference/svd_wrapper.md)
  : Singular Value Decomposition (SVD) Wrapper

- [`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
  : Multi-output linear regression

- [`mixed_regress()`](https://bbuchsbaum.github.io/multivarious/reference/mixed_regress.md)
  : Mixed-effect multivariate regression

- [`cPCAplus()`](https://bbuchsbaum.github.io/multivarious/reference/cPCAplus.md)
  :

  Contrastive PCA++ (cPCA++) Performs Contrastive PCA++ (cPCA++) to find
  directions that capture variation enriched in a "foreground" dataset
  relative to a "background" dataset. This implementation follows the
  cPCA++ approach which directly solves the generalized eigenvalue
  problem Rf v = lambda Rb v, where Rf and Rb are the covariance
  matrices of the foreground and background data, centered using the
  *background mean*.

- [`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
  : Generalized Eigenvalue Decomposition

- [`identity_basis()`](https://bbuchsbaum.github.io/multivarious/reference/identity_basis.md)
  : Identity basis specification

- [`shared_pca()`](https://bbuchsbaum.github.io/multivarious/reference/shared_pca.md)
  : Shared PCA basis specification

- [`supplied_basis()`](https://bbuchsbaum.github.io/multivarious/reference/supplied_basis.md)
  : Supplied basis specification

## Model classes for multivariate decompositions and extension

Generic S3 classes use to represented multivariate model fits

- [`projector()`](https://bbuchsbaum.github.io/multivarious/reference/projector.md)
  :

  Construct a `projector` instance

- [`bi_projector()`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector.md)
  : Construct a bi_projector instance

- [`effect_operator()`](https://bbuchsbaum.github.io/multivarious/reference/effect_operator.md)
  : Construct an effect operator

- [`bi_projector_union()`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector_union.md)
  :

  A Union of Concatenated `bi_projector` Fits

- [`discriminant_projector()`](https://bbuchsbaum.github.io/multivarious/reference/discriminant_projector.md)
  : Construct a Discriminant Projector

- [`cross_projector()`](https://bbuchsbaum.github.io/multivarious/reference/cross_projector.md)
  : Two-way (cross) projection to latent components

- [`multiblock_biprojector()`](https://bbuchsbaum.github.io/multivarious/reference/multiblock_biprojector.md)
  : Create a Multiblock Bi-Projector

- [`multiblock_projector()`](https://bbuchsbaum.github.io/multivarious/reference/multiblock_projector.md)
  : Create a Multiblock Projector

## Model Fitting and Projections

Functions for fitting models and applying projections.

- [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
  : New sample projection
- [`effect()`](https://bbuchsbaum.github.io/multivarious/reference/effect.md)
  : Extract a named effect from a fitted model
- [`residualize()`](https://bbuchsbaum.github.io/multivarious/reference/residualize.md)
  : Compute a regression model for each column in a matrix and return
  residual matrix
- [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
  : Partially project a new sample onto subspace
- [`partial_projector()`](https://bbuchsbaum.github.io/multivarious/reference/partial_projector.md)
  : Construct a partial projector
- [`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md)
  : Project a single "block" of data onto the subspace
- [`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md)
  : Project one or more variables onto a subspace
- [`transpose()`](https://bbuchsbaum.github.io/multivarious/reference/transpose.md)
  : Transpose a model
- [`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
  : Reconstruct the data
- [`inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.md)
  : Inverse of the Component Matrix
- [`partial_inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/partial_inverse_projection.md)
  : Partial Inverse Projection of a Columnwise Subset of Component
  Matrix
- [`compose_projector()`](https://bbuchsbaum.github.io/multivarious/reference/compose_projector.md)
  : Compose Two Projectors
- [`compose_partial_projector()`](https://bbuchsbaum.github.io/multivarious/reference/compose_partial_projector.md)
  [`` `%>>%` ``](https://bbuchsbaum.github.io/multivarious/reference/compose_partial_projector.md)
  : Compose Multiple Partial Projectors
- [`refit()`](https://bbuchsbaum.github.io/multivarious/reference/refit.md)
  : refit a model
- [`nystrom_approx()`](https://bbuchsbaum.github.io/multivarious/reference/nystrom_approx.md)
  : Nyström approximation for kernel-based decomposition (Unified
  Version)
- [`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md)
  : Fit a preprocessing pipeline
- [`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md)
  : Fit and transform data in one step
- [`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)
  : Transform data using a fitted preprocessing pipeline
- [`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)
  : Inverse transform data using a fitted preprocessing pipeline

## Cross Projection

Functions for creating and working with cross_projectors for two-way
projection between two sets of variables or features.

- [`project(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/project.cross_projector.md)
  : project a cross_projector instance
- [`coef(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/coef.cross_projector.md)
  : Extract coefficients from a cross_projector object
- [`reprocess(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reprocess.cross_projector.md)
  : reprocess a cross_projector instance
- [`shape(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/shape.cross_projector.md)
  : shape of a cross_projector instance
- [`transfer()`](https://bbuchsbaum.github.io/multivarious/reference/transfer.md)
  : Transfer data from one domain/block to another via a latent space
- [`inverse_projection(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.cross_projector.md)
  : Default inverse_projection method for cross_projector
- [`partial_inverse_projection(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_inverse_projection.cross_projector.md)
  : Partial Inverse Projection of a Subset of the Loading Matrix in
  cross_projector
- [`partial_project(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.cross_projector.md)
  : Partially project data for a cross_projector

## Model Components and Properties

Functions for working with model components and properties.

- [`components()`](https://bbuchsbaum.github.io/multivarious/reference/components.md)
  : get the components
- [`scores()`](https://bbuchsbaum.github.io/multivarious/reference/scores.md)
  : Retrieve the component scores
- [`scores(`*`<cca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/scores.cca.md)
  : Extract scores from a CCA fit
- [`scores(`*`<plsc>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/scores.plsc.md)
  : Extract scores from a PLSC fit
- [`std_scores()`](https://bbuchsbaum.github.io/multivarious/reference/std_scores.md)
  : Compute standardized component scores
- [`sdev()`](https://bbuchsbaum.github.io/multivarious/reference/sdev.md)
  : standard deviations
- [`ncomp()`](https://bbuchsbaum.github.io/multivarious/reference/ncomp.md)
  : Get the number of components
- [`shape()`](https://bbuchsbaum.github.io/multivarious/reference/shape.md)
  : Shape of the Projector
- [`is_orthogonal()`](https://bbuchsbaum.github.io/multivarious/reference/is_orthogonal.md)
  : is it orthogonal
- [`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md)
  : truncate a component fit
- [`block_lengths()`](https://bbuchsbaum.github.io/multivarious/reference/block_lengths.md)
  : get block_lengths
- [`block_indices()`](https://bbuchsbaum.github.io/multivarious/reference/block_indices.md)
  : get block_indices
- [`nblocks()`](https://bbuchsbaum.github.io/multivarious/reference/nblocks.md)
  : get the number of blocks
- [`prinang()`](https://bbuchsbaum.github.io/multivarious/reference/prinang.md)
  : Calculate Principal Angles Between Subspaces
- [`principal_angles()`](https://bbuchsbaum.github.io/multivarious/reference/principal_angles.md)
  : Principal angles (two sub‑spaces)
- [`variables_used()`](https://bbuchsbaum.github.io/multivarious/reference/variables_used.md)
  : Identify Original Variables Used by a Projector
- [`vars_for_component()`](https://bbuchsbaum.github.io/multivarious/reference/vars_for_component.md)
  : Identify Original Variables for a Specific Component

## Rotation and Transformation

Functions for rotating and transforming model components.

- [`rotate()`](https://bbuchsbaum.github.io/multivarious/reference/rotate.md)
  : Rotate a Component Solution
- [`apply_rotation()`](https://bbuchsbaum.github.io/multivarious/reference/apply_rotation.md)
  : Apply rotation

## Resampling and Confidence Intervals

Functions for bootstrapping and estimating confidence intervals.

- [`bootstrap()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.md)
  : Bootstrap Resampling for Multivariate Models

- [`bootstrap(`*`<effect_operator>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.effect_operator.md)
  : Bootstrap stability summaries for an effect operator

- [`bootstrap_pca()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap_pca.md)
  :

  Fast, Exact Bootstrap for PCA Results from `pca` function

- [`bootstrap_plsc()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap_plsc.md)
  : Bootstrap inference for PLSC loadings

- [`perm_ci()`](https://bbuchsbaum.github.io/multivarious/reference/perm_ci.md)
  : Permutation Confidence Intervals

- [`perm_test`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
  [`perm_test.pca`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
  [`perm_test.cross_projector`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
  [`perm_test.discriminant_projector`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
  [`perm_test.multiblock_biprojector`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
  : Generic Permutation-Based Test

- [`perm_test(`*`<effect_operator>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.effect_operator.md)
  : Permutation test for an effect operator

- [`perm_test(`*`<plsc>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.plsc.md)
  : Permutation test for PLSC latent variables

- [`cv()`](https://bbuchsbaum.github.io/multivarious/reference/cv.md) :
  Cross-validation Framework

- [`cv_generic()`](https://bbuchsbaum.github.io/multivarious/reference/cv_generic.md)
  : Generic cross-validation engine

## Classifier Construction

Functions for constructing classifiers.

- [`classifier()`](https://bbuchsbaum.github.io/multivarious/reference/classifier.md)
  : Construct a Classifier
- [`classifier(`*`<discriminant_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/classifier.discriminant_projector.md)
  : Create a k-NN classifier for a discriminant projector
- [`classifier(`*`<multiblock_biprojector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/classifier.multiblock_biprojector.md)
  : Multiblock Bi-Projector Classifier
- [`rf_classifier()`](https://bbuchsbaum.github.io/multivarious/reference/rf_classifier.md)
  : construct a random forest wrapper classifier
- [`rf_classifier(`*`<projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/rf_classifier.projector.md)
  : Create a random forest classifier
- [`predict(`*`<classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/predict.classifier.md)
  : Predict Class Labels using a Classifier Object
- [`feature_importance()`](https://bbuchsbaum.github.io/multivarious/reference/feature_importance.md)
  : Evaluate feature importance
- [`feature_importance(`*`<classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/feature_importance.classifier.md)
  : Evaluate Feature Importance for a Classifier

## Model Diagnostics and Residuals

Functions for evaluating model fit and residuals.

- [`residuals()`](https://bbuchsbaum.github.io/multivarious/reference/residuals.md)
  : Obtain residuals of a component model fit
- [`measure_reconstruction_error()`](https://bbuchsbaum.github.io/multivarious/reference/measure_reconstruction_error.md)
  : Compute reconstruction-based error metrics
- [`measure_interblock_transfer_error()`](https://bbuchsbaum.github.io/multivarious/reference/measure_interblock_transfer_error.md)
  : Compute inter-block transfer error metrics for a cross_projector
- [`pca_outliers()`](https://bbuchsbaum.github.io/multivarious/reference/pca_outliers.md)
  : PCA Outlier Diagnostics
- [`rank_score()`](https://bbuchsbaum.github.io/multivarious/reference/rank_score.md)
  : Calculate Rank Score for Predictions
- [`subspace_similarity()`](https://bbuchsbaum.github.io/multivarious/reference/subspace_similarity.md)
  : Compute subspace similarity

## Pre-processing

Functions for pre-processing data and managing pipelines.

- [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)
  : center a data matrix
- [`pass()`](https://bbuchsbaum.github.io/multivarious/reference/pass.md)
  : a no-op pre-processing step
- [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)
  : center and scale each vector of a matrix
- [`colscale()`](https://bbuchsbaum.github.io/multivarious/reference/colscale.md)
  : scale a data matrix
- [`prep()`](https://bbuchsbaum.github.io/multivarious/reference/prep.md)
  : prepare a dataset by applying a pre-processing pipeline
- [`fresh()`](https://bbuchsbaum.github.io/multivarious/reference/fresh.md)
  : Get a fresh pre-processing node cleared of any cached data
- [`reprocess()`](https://bbuchsbaum.github.io/multivarious/reference/reprocess.md)
  : apply pre-processing parameters to a new data matrix
- [`apply_transform()`](https://bbuchsbaum.github.io/multivarious/reference/apply_transform.md)
  : apply a pre-processing transform
- [`reverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/reverse_transform.md)
  : reverse a pre-processing transform
- [`init_transform()`](https://bbuchsbaum.github.io/multivarious/reference/init_transform.md)
  : initialize a transform
- [`concat_pre_processors()`](https://bbuchsbaum.github.io/multivarious/reference/concat_pre_processors.md)
  : bind together blockwise pre-processors
- [`add_node()`](https://bbuchsbaum.github.io/multivarious/reference/add_node.md)
  : add a pre-processing stage
- [`preprocess()`](https://bbuchsbaum.github.io/multivarious/reference/preprocess.md)
  : Convenience function for preprocessing workflow
- [`check_fitted()`](https://bbuchsbaum.github.io/multivarious/reference/check_fitted.md)
  : Check if preprocessor is fitted and error if not
- [`is_fitted()`](https://bbuchsbaum.github.io/multivarious/reference/is_fitted.md)
  : Check if a preprocessing object is fitted
- [`mark_fitted()`](https://bbuchsbaum.github.io/multivarious/reference/mark_fitted.md)
  : Enhanced fitted state tracking
- [`get_fitted_state()`](https://bbuchsbaum.github.io/multivarious/reference/get_fitted_state.md)
  : Get fitted state from attributes

## Visualization

Functions for plotting and visualization.

- [`biplot(`*`<pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/biplot.pca.md)
  : Biplot for PCA Objects (Enhanced with ggrepel)
- [`screeplot()`](https://bbuchsbaum.github.io/multivarious/reference/screeplot.md)
  : Screeplot for PCA
- [`screeplot(`*`<pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/screeplot.pca.md)
  : Screeplot for PCA

## Utilities

Utility functions.

- [`group_means()`](https://bbuchsbaum.github.io/multivarious/reference/group_means.md)
  : Compute column-wise mean in X for each factor level of Y
- [`robust_inv_vTv()`](https://bbuchsbaum.github.io/multivarious/reference/robust_inv_vTv.md)
  : Possibly use ridge-regularized inversion of crossprod(v)
- [`topk()`](https://bbuchsbaum.github.io/multivarious/reference/topk.md)
  : top-k accuracy indicator
- [`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md)
  : Reconstruct new data in a model's subspace

## Other

Other functions

- [`print(`*`<bi_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.bi_projector.md)
  : Pretty Print S3 Method for bi_projector Class

- [`print(`*`<classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.classifier.md)
  :

  Pretty Print Method for `classifier` Objects

- [`print(`*`<concat_pre_processor>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.concat_pre_processor.md)
  : Print a concat_pre_processor object

- [`print(`*`<multiblock_biprojector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.multiblock_biprojector.md)
  :

  Pretty Print Method for `multiblock_biprojector` Objects

- [`print(`*`<pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.pca.md)
  : Print Method for PCA Objects

- [`print(`*`<perm_test>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.perm_test.md)
  : Print Method for perm_test Objects

- [`print(`*`<perm_test_pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.perm_test_pca.md)
  : Print Method for perm_test_pca Objects

- [`print(`*`<pre_processor>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.pre_processor.md)
  : Print a pre_processor object

- [`print(`*`<prepper>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.prepper.md)
  : Print a prepper pipeline

- [`print(`*`<regress>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.regress.md)
  :

  Pretty Print Method for `regress` Objects

- [`print(`*`<rf_classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/print.rf_classifier.md)
  :

  Pretty Print Method for `rf_classifier` Objects

- [`summary(`*`<composed_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/summary.composed_projector.md)
  : Summarize a Composed Projector

- [`coef(`*`<composed_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/coef.composed_projector.md)
  : Get Coefficients of a Composed Projector

- [`coef(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/coef.cross_projector.md)
  : Extract coefficients from a cross_projector object

- [`coef(`*`<multiblock_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/coef.multiblock_projector.md)
  : Coefficients for a Multiblock Projector

- [`predict(`*`<classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/predict.classifier.md)
  : Predict Class Labels using a Classifier Object

- [`predict(`*`<discriminant_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/predict.discriminant_projector.md)
  : Predict method for a discriminant_projector, supporting LDA or
  Euclid

- [`predict(`*`<rf_classifier>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/predict.rf_classifier.md)
  : Predict Class Labels using a Random Forest Classifier Object

- [`reprocess()`](https://bbuchsbaum.github.io/multivarious/reference/reprocess.md)
  : apply pre-processing parameters to a new data matrix

- [`reprocess(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reprocess.cross_projector.md)
  : reprocess a cross_projector instance

- [`reprocess(`*`<nystrom_approx>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reprocess.nystrom_approx.md)
  : Reprocess data for Nyström approximation

- [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
  : New sample projection

- [`project(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/project.cross_projector.md)
  : project a cross_projector instance

- [`project(`*`<nystrom_approx>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/project.nystrom_approx.md)
  : Project new data using a Nyström approximation model

- [`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md)
  : Project a single "block" of data onto the subspace

- [`project_block(`*`<multiblock_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/project_block.multiblock_projector.md)
  : Project Data onto a Specific Block

- [`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md)
  : Project one or more variables onto a subspace

- [`projector()`](https://bbuchsbaum.github.io/multivarious/reference/projector.md)
  :

  Construct a `projector` instance

- [`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md)
  : truncate a component fit

- [`truncate(`*`<composed_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/truncate.composed_projector.md)
  : Truncate a Composed Projector

- [`block_indices()`](https://bbuchsbaum.github.io/multivarious/reference/block_indices.md)
  : get block_indices

- [`block_indices(`*`<multiblock_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/block_indices.multiblock_projector.md)
  : Extract the Block Indices from a Multiblock Projector

- [`inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.md)
  : Inverse of the Component Matrix

- [`inverse_projection(`*`<composed_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.composed_projector.md)
  : Compute the Inverse Projection for a Composed Projector

- [`inverse_projection(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.cross_projector.md)
  : Default inverse_projection method for cross_projector

- [`partial_inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/partial_inverse_projection.md)
  : Partial Inverse Projection of a Columnwise Subset of Component
  Matrix

- [`partial_inverse_projection(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_inverse_projection.cross_projector.md)
  : Partial Inverse Projection of a Subset of the Loading Matrix in
  cross_projector

- [`partial_inverse_projection(`*`<regress>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_inverse_projection.regress.md)
  :

  Partial Inverse Projection for a `regress` Object

- [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
  : Partially project a new sample onto subspace

- [`partial_project(`*`<composed_partial_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.composed_partial_projector.md)
  : Partial Project Through a Composed Partial Projector

- [`partial_project(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.cross_projector.md)
  : Partially project data for a cross_projector

- [`partial_projector()`](https://bbuchsbaum.github.io/multivarious/reference/partial_projector.md)
  : Construct a partial projector

- [`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
  : Reconstruct the data

- [`reconstruct(`*`<composed_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.composed_projector.md)
  : Reconstruct Data from Scores using a Composed Projector

- [`reconstruct(`*`<pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.pca.md)
  : Reconstruct Data from PCA Results

- [`reconstruct(`*`<regress>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.regress.md)
  :

  Reconstruct fitted or subsetted outputs for a `regress` object

- [`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md)
  : Reconstruct new data in a model's subspace

- [`rotate()`](https://bbuchsbaum.github.io/multivarious/reference/rotate.md)
  : Rotate a Component Solution

- [`rotate(`*`<pca>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/rotate.pca.md)
  : Rotate PCA Loadings

- [`std_scores()`](https://bbuchsbaum.github.io/multivarious/reference/std_scores.md)
  : Compute standardized component scores

- [`std_scores(`*`<svd>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/std_scores.svd.md)
  : Calculate Standardized Scores for SVD results

- [`is_orthogonal()`](https://bbuchsbaum.github.io/multivarious/reference/is_orthogonal.md)
  : is it orthogonal

- [`is_orthogonal(`*`<projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/is_orthogonal.projector.md)
  : Stricter check for true orthogonality

- [`add_node()`](https://bbuchsbaum.github.io/multivarious/reference/add_node.md)
  : add a pre-processing stage

- [`add_node(`*`<prepper>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/add_node.prepper.md)
  : Add a pre-processing node to a pipeline

- [`shape()`](https://bbuchsbaum.github.io/multivarious/reference/shape.md)
  : Shape of the Projector

- [`shape(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/shape.cross_projector.md)
  : shape of a cross_projector instance

- [`transfer()`](https://bbuchsbaum.github.io/multivarious/reference/transfer.md)
  : Transfer data from one domain/block to another via a latent space

- [`transfer(`*`<cross_projector>`*`)`](https://bbuchsbaum.github.io/multivarious/reference/transfer.cross_projector.md)
  : Transfer from X domain to Y domain (or vice versa) in a
  cross_projector
