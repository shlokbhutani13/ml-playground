from __future__ import annotations
import numpy as np
from .utils import add_bias, sigmoid

class LogisticRegressionGD:
    """
    Binary logistic regression from scratch with gradient descent.
    y must be 0/1.
    """
    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 0.0, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = add_bias(X)
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(scale=0.01, size=(Xb.shape[1],))

        n = Xb.shape[0]
        for _ in range(self.epochs):
            p = sigmoid(Xb @ self.w)
            grad = (Xb.T @ (p - y)) / n

            if self.l2 > 0:
                reg = self.l2 * np.r_[0.0, self.w[1:]]
                grad += reg

            self.w -= self.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        Xb = add_bias(X)
        return sigmoid(Xb @ self.w)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
