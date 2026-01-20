# Changelog

## v0.4.1 (2026-01-20)

  * Add `Scholar.FeatureExtraction.CountVectorizer`
  * Add `Scholar.NaiveBayes.Categorical`
  * Add `Scholar.Optimize.Brent`
  * Add `Scholar.Optimize.GoldenSection`
  * Improve `Scholar.Cluster.DBSCAN`'s performance
  * General fixes to `Scholar.Linear.LinearRegression`

## v0.4.0 (2025-01-15)

  * Require Nx `~> 0.9`
  * Add batching to regression metrics
  * Add `Scholar.Cluster.OPTICS`
  * Add `Scholar.Covariance.LedoitWolf`
  * Add `Scholar.Covariance.ShrunkCovariance`
  * Add `Scholar.CrossDecomposition.PLSSVD`
  * Add `Scholar.Decomposition.TruncatedSVD`
  * Add `Scholar.Impute.KNNImputter`
  * Add `Scholar.NaiveBayes.Bernoulli`
  * Add `Scholar.Preprocessing.Binarizer`
  * Add `Scholar.Preprocessing.RobustScaler`
  * Add `partial_fit/2` and `incremental_fit/2` to PCA
  * Split `RNN` into `Scholar.Neighbors.RadiusNNClassifier` and `Scholar.Neighbors.RadiusNNRegressor`
  * Unify shape checks across all APIs

## v0.3.1 (2024-06-18)

### Enhancements

  * Add a notebook about manifold learning
  * Make knn algorithm configurable on Trimap
  * Add `d2_pinball_score` and `d2_absolute_error_score`

## v0.3.0 (2024-05-29)

### Enhancements

  * Add LargeVis for visualization of large-scale and high-dimensional data in a low-dimensional (typically 2D or 3D) space
  * Add `Scholar.Neighbors.KDTree` and `Scholar.Neighbors.RandomProjectionForest`
  * Add `Scholar.Metrics.Neighbors`
  * Add `Scholar.Linear.BayesianRidgeRegression`
  * Add `Scholar.Cluster.Hierarchical`
  * Add `Scholar.Manifold.Trimap`
  * Add Mean Pinball Loss function
  * Add Matthews Correlation Coefficient function
  * Add D2 Tweedie Score function
  * Add Mean Tweedie Deviance function
  * Add Discounted Cumulative Gain function
  * Add Precision Recall f-score function
  * Add f-beta score function
  * Add convergence check to AffinityPropagation
  * Default Affinity Propagation preference to `reduce_min` and make it customizable
  * Move preprocessing functionality to their own modules with `fit` and `fit_transform` callbacks

### Breaking changes

  * Split `KNearestNeighbors` into `KNNClassifier` and `KNNRegressor` with custom algorithm support

## v0.2.1 (2023-08-30)

### Enhancements

  * Remove `VegaLite.Data` in favour of future use of `Tucan`
  * Do not use EXLA at compile time in `Metrics`

## v0.2.0 (2023-08-29)

This version requires Elixir v1.14+.

### Enhancements

  * Update notebooks
  * Add support for `:f16` and `:bf16` types in `SVD`
  * Add `Affinity Propagation`
  * Add `t-SNE`
  * Add `Polynomial Regression`
  * Replace seeds with `Random.key`
  * Add 'unrolling loops' option
  * Add support for custom optimizers in `Logistic Regression`
  * Add `Trapezoidal Integration`
  * Add `AUC-ROC`, `AUC`, and `ROC Curve`
  * Add `Simpson rule integration`
  * Unify tests
  * Add `Radius Nearest Neighbors`
  * Add `DBSCAN`
  * Add classification metrics: `Average Precision Score`, `Balanced Accuracy Score`,
  `Cohen Kappa Score`, `Brier Score Loss`, `Zero-One Loss`, `Top-k Accuracy Score`
  * Add regression metrics: `R2 Score`, `MSLE`, `MAPE`, `Maximum Residual Error`
  * Add support for axes in `Confusion Matrix`
  * Add support for broadcasting in `Metrics.Distances`
  * Update CI
  * Add `Gaussian Mixtures`
  * Add Model selection functionalities: `K-fold`, `K-fold Cross Validation`, `Grid Search`
  * Change structure of metrics in `Scholar`
  * Add a guide with `Cross-Validation` and `Grid Search`

## v0.1.0 (2023-03-29)

First release.
