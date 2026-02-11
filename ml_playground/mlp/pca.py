from __future__ import annotations
import numpy as np

class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        X = X.astype(float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # covariance eigen-decomposition
        cov = np.cov(Xc, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        self.components_ = eigvecs[:, : self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA not fitted.")
        Xc = X.astype(float) - self.mean_
        return Xc @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
