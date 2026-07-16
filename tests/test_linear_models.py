import numpy as np
import pytest

from ml_playground import LinearRegressionGD


def test_linear_regression_recovers_known_relationship():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5

    model = LinearRegressionGD(lr=0.08, epochs=4000, tol=1e-12, seed=1).fit(X, y)

    np.testing.assert_allclose(model.coef_, [2.0, -3.0], atol=0.03)
    assert model.intercept_ == pytest.approx(0.5, abs=0.03)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_linear_regression_rejects_prediction_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearRegressionGD().predict([[1.0, 2.0]])
