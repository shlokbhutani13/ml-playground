from __future__ import annotations

import numpy as np

from ._validation import (
    as_2d_float_array,
    check_feature_count,
    require_fitted,
    validate_positive_int,
)


class KMeans:
    """Lloyd's k-means clustering algorithm."""

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 200,
        tol: float = 1e-4,
        seed: int = 42,
    ) -> None:
        validate_positive_int(k, name="k")
        validate_positive_int(max_iter, name="max_iter")
        if tol < 0:
            raise ValueError("tol must be zero or greater.")
        self.k = k
        self.max_iter = max_iter
        self.tol = float(tol)
        self.seed = int(seed)
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> KMeans:
        X_array = as_2d_float_array(X, name="X")
        if self.k > X_array.shape[0]:
            raise ValueError("k cannot exceed the number of samples.")
        rng = np.random.default_rng(self.seed)
        centers = X_array[rng.choice(X_array.shape[0], size=self.k, replace=False)].copy()

        for _iteration in range(1, self.max_iter + 1):
            distances = ((X_array[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for cluster in range(self.k):
                members = X_array[labels == cluster]
                if members.size:
                    new_centers[cluster] = members.mean(axis=0)
            shift = float(np.linalg.norm(new_centers - centers))
            centers = new_centers
            if shift <= self.tol:
                break

        distances = ((X_array[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(distances[np.arange(X_array.shape[0]), labels].sum())
        self.n_iter_ = _iteration
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        require_fitted(self, ("cluster_centers_", "n_features_in_"))
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        distances = ((X_array[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(distances, axis=1)
