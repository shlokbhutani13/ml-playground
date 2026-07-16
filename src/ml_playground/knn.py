from __future__ import annotations

import numpy as np

from ._validation import (
    as_2d_float_array,
    check_feature_count,
    check_X_y,
    require_fitted,
    validate_positive_int,
)


class KNNClassifier:
    """Brute-force k-nearest-neighbors classifier."""

    def __init__(self, k: int = 5) -> None:
        validate_positive_int(k, name="k")
        self.k = k
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        X_array, y_array = check_X_y(X, y)
        if self.k > X_array.shape[0]:
            raise ValueError("k cannot exceed the number of training samples.")
        self.X_train_ = X_array.copy()
        self.y_train_ = y_array.copy()
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X_array.shape[1]
        return self

    def _neighbor_labels(self, X: np.ndarray) -> np.ndarray:
        require_fitted(self, ("X_train_", "y_train_", "classes_", "n_features_in_"))
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        squared_distances = ((X_array[:, None, :] - self.X_train_[None, :, :]) ** 2).sum(axis=2)
        indices = np.argsort(squared_distances, axis=1, kind="stable")[:, : self.k]
        return self.y_train_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        labels = self._neighbor_labels(X)
        return np.column_stack([(labels == value).mean(axis=1) for value in self.classes_])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
