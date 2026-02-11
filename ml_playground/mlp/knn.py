from __future__ import annotations
import numpy as np

class KNNClassifier:
    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >=1")
        self.k = k
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(float)
        self.y = y.astype(int)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise RuntimeError("Model not fitted.")
        Xq = X.astype(float)

        # distances: (n_query, n_train)
        dists = np.sqrt(((Xq[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2))
        nn_idx = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
        votes = self.y[nn_idx]

        # majority vote
        preds = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds, dtype=int)
