import numpy as np
import pytest

from ml_playground.utils import (
    StandardScaler,
    accuracy_score,
    mean_squared_error,
    train_test_split,
)


def test_train_test_split_is_reproducible():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    first = train_test_split(X, y, test_size=0.3, seed=8)
    second = train_test_split(X, y, test_size=0.3, seed=8)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_train_test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        train_test_split([[1.0], [2.0]], [0, 1], test_size=1.0)


def test_standard_scaler_handles_constant_columns():
    X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    scaled = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(scaled[:, 0], 0.0)
    np.testing.assert_allclose(scaled[:, 1].mean(), 0.0)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        accuracy_score([0, 1], [0])
    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error([0.0, 1.0], [0.0])
