from __future__ import annotations

import numpy as np

from ._validation import as_1d_array, as_2d_float_array, check_feature_count, check_X_y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    X_array, y_array = check_X_y(X, y)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_array.shape[0])
    test_count = max(1, int(round(X_array.shape[0] * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    if train_indices.size == 0:
        raise ValueError("test_size leaves no training samples.")
    return (
        X_array[train_indices],
        X_array[test_indices],
        y_array[train_indices],
        y_array[test_indices],
    )


class StandardScaler:
    """Column-wise standardization using training-set statistics."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> StandardScaler:
        X_array = as_2d_float_array(X, name="X")
        self.mean_ = X_array.mean(axis=0)
        scale = X_array.std(axis=0)
        self.scale_ = np.where(scale == 0, 1.0, scale)
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.n_features_in_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=self.n_features_in_)
        return (X_array - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = as_1d_array(y_true, name="y_true")
    prediction = as_1d_array(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(truth == prediction))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = as_1d_array(y_true, name="y_true").astype(float)
    prediction = as_1d_array(y_pred, name="y_pred").astype(float)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean((truth - prediction) ** 2))
