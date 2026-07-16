from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def as_2d_float_array(values: Iterable[object], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def as_1d_array(values: Iterable[object], *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    return array


def check_X_y(
    X: Iterable[object],
    y: Iterable[object],
) -> tuple[np.ndarray, np.ndarray]:
    X_array = as_2d_float_array(X, name="X")
    y_array = as_1d_array(y, name="y")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    return X_array, y_array


def check_feature_count(X: np.ndarray, *, expected: int) -> None:
    if X.shape[1] != expected:
        raise ValueError(f"X must contain {expected} features; received {X.shape[1]}.")


def require_fitted(estimator: object, attributes: tuple[str, ...]) -> None:
    if any(getattr(estimator, attribute, None) is None for attribute in attributes):
        raise RuntimeError(f"{estimator.__class__.__name__} is not fitted.")


def validate_positive_int(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
