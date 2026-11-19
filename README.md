# ML Playground

[![CI](https://github.com/shlokbhutani13/ml-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokbhutani13/ml-playground/actions/workflows/ci.yml)

Six classic machine learning algorithms written from scratch in NumPy. I built it to understand what actually happens inside `model.fit()`, instead of only calling scikit-learn.

The code favors readability over speed. It's good for learning and interview prep, not for production.

## The algorithms

| Algorithm | Used for | How it learns |
| --- | --- | --- |
| Linear regression | Predicting numbers | Gradient descent |
| Logistic regression | Yes/no classification | Gradient descent on cross-entropy |
| Perceptron | Yes/no classification | Updates on each mistake |
| K-nearest neighbors | Classification | Votes from the closest examples |
| K-means | Clustering | Lloyd's algorithm |
| PCA | Shrinking dimensions | Eigenvectors of the covariance matrix |

Each model keeps what it learned so you can inspect it: weights and loss history for the gradient descent models, mistakes per epoch for the perceptron, centers and inertia for k-means, and explained variance for PCA.

## Try it

You need Python 3.11 or newer.

```bash
git clone https://github.com/shlokbhutani13/ml-playground.git
cd ml-playground
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/demo.py
```

The demo runs every algorithm on fixed random data, so you get the same output each time:

```text
Logistic accuracy: 1.000
KNN accuracy: 1.000
Perceptron accuracy: 1.000
Linear regression MSE: 0.509
K-means inertia: 391.295
PCA explained variance: 0.946
```

Or use one model directly:

```python
import numpy as np
from ml_playground import LinearRegressionGD

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])

model = LinearRegressionGD(lr=0.1, epochs=2_000).fit(X, y)
print(model.predict([[4.0]]))  # close to 9
```

## Tests

```bash
ruff check .
pytest
```

The tests check the math against known answers, make sure probabilities stay between 0 and 1, and confirm that bad inputs and unfitted models raise clear errors.

## License

[MIT](LICENSE)
