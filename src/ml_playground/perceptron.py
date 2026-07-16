from __future__ import annotations

import numpy as np

from ._validation import as_2d_float_array, check_feature_count, check_X_y, require_fitted


class Perceptron:
    """Online binary perceptron for labels -1 and 1."""

    def __init__(self, lr: float = 0.1, epochs: int = 50, seed: int = 42) -> None:
        if lr <= 0:
            raise ValueError("lr must be greater than zero.")
        if epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.mistakes_history_: list[int] = []
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        X_array, y_array = check_X_y(X, y)
        if not set(np.unique(y_array)).issubset({-1, 1}):
            raise ValueError("y must contain only labels -1 and 1.")

        self.coef_ = np.zeros(X_array.shape[1], dtype=float)
        self.intercept_ = 0.0
        self.n_features_in_ = X_array.shape[1]
        self.mistakes_history_ = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            mistakes = 0
            for index in rng.permutation(X_array.shape[0]):
                score = X_array[index] @ self.coef_ + self.intercept_
                if y_array[index] * score <= 0:
                    self.coef_ += self.lr * y_array[index] * X_array[index]
                    self.intercept_ += self.lr * float(y_array[index])
                    mistakes += 1
            self.mistakes_history_.append(mistakes)
            if mistakes == 0:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        require_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        scores = X_array @ self.coef_ + float(self.intercept_)
        return np.where(scores >= 0, 1, -1)
