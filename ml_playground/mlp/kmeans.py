from __future__ import annotations
import numpy as np

class KMeans:
    def __init__(self, k: int = 3, max_iter: int = 200, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.centroids: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        X = X.astype(float)

        # init centroids from random samples
        idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iter):
            labels = self.predict(X)
            new_centroids = np.vstack([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Model not fitted.")
        X = X.astype(float)
        d = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)  # squared distances
        return np.argmin(d, axis=1)
