from __future__ import annotations
import numpy as np
from .utils import add_bias

class Perceptron:
    def __init__(self, lr: float = 0.1, epochs: int = 50, seed: int = 42):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # y must be -1/+1
        Xb = add_bias(X)
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(scale=0.01, size=(Xb.shape[1],))

        for _ in range(self.epochs):
            for i in range(len(Xb)):
                if y[i] * (Xb[i] @ self.w) <= 0:
                    self.w += self.lr * y[i] * Xb[i]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not fitted.")
        Xb = add_bias(X)
        return np.where((Xb @ self.w) >= 0, 1, -1)
