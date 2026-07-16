from __future__ import annotations

import numpy as np

from ._validation import as_2d_float_array, check_feature_count, require_fitted


class PCA:
    """Principal component analysis using covariance eigendecomposition."""

    def __init__(self, n_components: int = 2) -> None:
        if not isinstance(n_components, int) or isinstance(n_components, bool):
            raise ValueError("n_components must be an integer.")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray) -> PCA:
        X_array = as_2d_float_array(X, name="X")
        upper = min(X_array.shape[0], X_array.shape[1])
        if not 1 <= self.n_components <= upper:
            raise ValueError(f"n_components must be between 1 and {upper}.")

        self.mean_ = X_array.mean(axis=0)
        centered = X_array - self.mean_
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = float(eigenvalues.sum())
        if total_variance == 0:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        else:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        require_fitted(
            self,
            ("mean_", "components_", "explained_variance_", "n_features_in_"),
        )
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        return (X_array - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
