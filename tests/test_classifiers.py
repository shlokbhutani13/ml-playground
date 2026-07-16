import numpy as np
import pytest

from ml_playground import LogisticRegressionGD, Perceptron
from ml_playground.utils import accuracy_score


def classification_fixture():
    rng = np.random.default_rng(7)
    negative = rng.normal(loc=(-2.0, -2.0), scale=0.45, size=(100, 2))
    positive = rng.normal(loc=(2.0, 2.0), scale=0.45, size=(100, 2))
    X = np.vstack([negative, positive])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


def test_logistic_regression_returns_valid_probabilities():
    X, y = classification_fixture()
    model = LogisticRegressionGD(lr=0.1, epochs=2500, tol=1e-10, seed=2).fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (200, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert accuracy_score(y, model.predict(X)) > 0.95
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_regression_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="0 and 1"):
        LogisticRegressionGD().fit([[0.0], [1.0], [2.0]], [0, 1, 2])


def test_perceptron_converges_on_separable_data():
    X, y_zero_one = classification_fixture()
    y = np.where(y_zero_one == 1, 1, -1)

    model = Perceptron(lr=0.1, epochs=30, seed=5).fit(X, y)

    assert accuracy_score(y, model.predict(X)) == 1.0
    assert model.mistakes_history_[-1] == 0


def test_perceptron_rejects_invalid_labels():
    with pytest.raises(ValueError, match="-1 and 1"):
        Perceptron().fit([[0.0], [1.0]], [0, 1])
