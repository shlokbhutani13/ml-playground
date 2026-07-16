from __future__ import annotations

import numpy as np

from ._validation import as_2d_float_array, check_feature_count, check_X_y, require_fitted


class LinearRegressionGD:
    """Linear regression trained with batch gradient descent."""

    def __init__(
        self,
        lr: float = 0.05,
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
        self.loss_history_: list[float] = []
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        X_array, y_array = check_X_y(X, y)
        y_float = np.asarray(y_array, dtype=float)
        if not np.isfinite(y_float).all():
            raise ValueError("y must contain only finite values.")

        rng = np.random.default_rng(self.seed)
        self.coef_ = rng.normal(scale=0.01, size=X_array.shape[1])
        self.intercept_ = 0.0
        self.n_features_in_ = X_array.shape[1]
        self.loss_history_ = []

        for _ in range(self.epochs):
            predictions = X_array @ self.coef_ + self.intercept_
            errors = predictions - y_float
            gradient = (X_array.T @ errors) / X_array.shape[0] + self.l2 * self.coef_
            intercept_gradient = float(errors.mean())
            self.coef_ -= self.lr * gradient
            self.intercept_ -= self.lr * intercept_gradient

            updated_errors = X_array @ self.coef_ + self.intercept_ - y_float
            loss = 0.5 * float(np.mean(updated_errors**2))
            loss += 0.5 * self.l2 * float(self.coef_ @ self.coef_)
            self.loss_history_.append(loss)
            if len(self.loss_history_) > 1:
                if abs(self.loss_history_[-2] - self.loss_history_[-1]) <= self.tol:
                    break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        require_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        X_array = as_2d_float_array(X, name="X")
        check_feature_count(X_array, expected=int(self.n_features_in_))
        return X_array @ self.coef_ + float(self.intercept_)
