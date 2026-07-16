# ML Playground: Technical Decisions

## Purpose

ML Playground is a code-first learning package for core machine-learning algorithms implemented with NumPy. The package should make the mathematics readable, provide predictable APIs, and prove correctness with small tests.

The project favors clarity over performance. It will not add a web interface or production serving layer.

## Package design

The repository will use a standard Python layout:

```text
src/ml_playground/
tests/
examples/
pyproject.toml
README.md
```

NumPy will remain the only runtime dependency. Pytest and development tools will stay in an optional development dependency group.

Every estimator will use a consistent interface:

- constructor parameters control training behavior
- `fit(X, y)` or `fit(X)` returns `self`
- prediction methods reject unfitted models
- learned attributes end with an underscore
- shape and parameter validation raises `ValueError` with a specific message
- random behavior accepts a seed

## Algorithms

The package will keep the six existing implementations:

- linear regression with gradient descent and optional L2 regularization
- binary logistic regression with gradient descent and optional L2 regularization
- perceptron classification
- k-nearest-neighbors classification
- k-means clustering
- principal component analysis

The implementations will expose useful learning state:

- linear and logistic regression: coefficients, intercept, and loss history
- perceptron: coefficients, intercept, and mistake history
- k-nearest neighbors: fitted training data
- k-means: cluster centers, labels, inertia, and iteration count
- PCA: components, explained variance, and explained-variance ratio

## Examples

One command will run a deterministic example covering regression, classification, clustering, and dimensionality reduction. The output will report a small set of metrics with fixed seeds so readers can compare changes.

The README will include short usage examples for each estimator and an algorithm table covering task type, training method, and the main limitation. It will state that the implementations support learning and experimentation rather than large datasets.

## Failure handling

Shared validation helpers will reject:

- non-numeric arrays
- empty datasets
- mismatched sample counts
- invalid target labels
- impossible component or cluster counts
- prediction arrays with the wrong feature count

Each algorithm will check its own parameters before training.

## Verification

Tests will cover mathematical behavior rather than implementation details:

- linear regression recovers a known linear relationship
- logistic regression separates a simple binary dataset and returns valid probabilities
- perceptron converges on linearly separable points
- k-nearest neighbors classifies known neighborhoods and handles ties deterministically
- k-means finds stable centers and reports finite inertia
- PCA produces orthonormal components and ordered explained variance
- utility functions preserve shapes and reproducibility
- unfitted and invalid-input paths raise clear errors

Continuous integration will install the package and run tests, formatting checks, and type-aware static checks on a supported Python version.
