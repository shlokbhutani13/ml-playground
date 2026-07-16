from __future__ import annotations

import numpy as np

from ._validation import as_2d_float_array, check_feature_count, check_X_y, require_fitted


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 2_000,
        l2: float = 0.0,
        tol: float = 1e-9,
        seed: int = 42,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be greater than zero.")
        if epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if l2 < 0:
            raise ValueError("l2 must be zero or greater.")
        if tol < 0:
            raise ValueError("tol must be zero or greater.")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.tol = float(tol)
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.classes_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionGD:
        X_array, y_array = check_X_y(X, y)
        unique = np.unique(y_array)
        if not np.array_equal(unique, np.array([0, 1])):
            raise ValueError("y must contain both binary labels 0 and 1.")
        y_float = y_array.astype(float)

        rng = np.random.default_rng(self.seed)
        self.coef_ = rng.normal(scale=0.01, size=X_array.shape[1])
        self.intercept_ = 0.0
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X_array.shape[1]
        self.loss_history_ = []

        for _ in range(self.epochs):
            probabilities = _sigmoid(X_array @ self.coef_ + self.intercept_)
            errors = probabilities - y_float
            gradient = (X_array.T @ errors) / X_array.shape[0] + self.l2 * self.coef_
            intercept_gradient = float(errors.mean())
            self.coef_ -= self.lr * gradient
            self.intercept_ -= self.lr * intercept_gradient

            updated = np.clip(
                _sigmoid(X_array @ self.coef_ + self.intercept_),
                1e-12,
                1 - 1e-12,
            )
            loss = -float(np.mean(y_float * np.log(updated) + (1 - y_float) * np.log(1 - updated)))
            loss += 0.5 * self.l2 * float(self.coef_ @ self.coef_)
            self.loss_history_.append(loss)
            if len(self.loss_history_) > 1:
                if abs(self.loss_history_[-2] - self.loss_history_[-1]) <= self.tol:
                    break
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        require_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        positive = _sigmoid(X_array @ self.coef_ + float(self.intercept_))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
