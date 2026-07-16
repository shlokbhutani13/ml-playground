# ML Playground

[![CI](https://github.com/shlokbhutani13/ml-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/ml-playground/actions/workflows/ci.yml)

I built this package to understand how common machine-learning algorithms work beneath library APIs. Each estimator uses NumPy and keeps the training math close to the code.

The implementations favor readability and small datasets. They are useful for study, experiments, and interviews, but they are not replacements for scikit-learn in production work.

## Algorithms

| Estimator | Task | Training approach | Main limitation |
| --- | --- | --- | --- |
| `LinearRegressionGD` | Regression | Batch gradient descent | Requires learning-rate tuning |
| `LogisticRegressionGD` | Binary classification | Cross-entropy gradient descent | Supports two classes |
| `Perceptron` | Binary classification | Online mistake updates | Converges only on separable data |
| `KNNClassifier` | Classification | Stores the training set | Prediction cost grows with the dataset |
| `KMeans` | Clustering | Lloyd's algorithm | Sensitive to initialization |
| `PCA` | Dimensionality reduction | Covariance eigendecomposition | Uses a dense covariance matrix |

## Install

Use Python 3.11 or newer:

```bash
git clone https://github.com/shlokbhutani13/ml-playground.git
cd ml-playground
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Example

```python
import numpy as np

from ml_playground import LinearRegressionGD

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

model = LinearRegressionGD(lr=0.1, epochs=2_000).fit(X, y)
print(model.predict([[4.0]]))
```

Run the full example:

```bash
python examples/demo.py
```

Typical output:

```text
Logistic accuracy: 1.000
KNN accuracy: 1.000
Perceptron accuracy: 1.000
Linear regression MSE: 0.509
K-means inertia: 391.295
PCA explained variance: 0.946
```

The example uses fixed random seeds, so the output stays stable across runs with the supported dependency range.

## Learned attributes

The fitted estimators expose the state that helps explain each algorithm:

- gradient-descent models provide coefficients, intercepts, and loss history
- the perceptron records mistakes per epoch
- k-means reports centers, labels, inertia, and iteration count
- PCA reports components and explained variance

All learned attributes use a trailing underscore, following the convention used by common Python ML libraries.

## Verification

```bash
ruff check .
pytest
python -m build
```

The tests check known mathematical relationships, probability behavior, convergence, deterministic seeds, input validation, and unfitted-model errors.

## Project structure

```text
src/ml_playground/   algorithm implementations
tests/               mathematical and validation tests
examples/demo.py     deterministic end-to-end example
docs/DECISIONS.md    API and scope decisions
```

## License

[MIT](LICENSE)
