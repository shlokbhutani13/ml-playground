from __future__ import annotations
import numpy as np
from .utils import add_bias

class LinearRegressionGD:
    """
    Linear regression trained with gradient descent.
    Supports optional L2 regularization (ridge) via l2>0.
    """
    def __init__(self, lr: float = 0.05, epochs: int = 2000, l2: float = 0.0, seed: int = 42):
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
            pred = Xb @ self.w
            err = pred - y
            grad = (Xb.T @ err) / n

            # L2 regularization (exclude bias term)
            if self.l2 > 0:
                reg = self.l2 * np.r_[0.0, self.w[1:]]
                grad += reg

            self.w -= self.lr * grad
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        Xb = add_bias(X)
        return Xb @ self.w
