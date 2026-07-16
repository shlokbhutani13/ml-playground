from __future__ import annotations

import numpy as np

from ml_playground import (
    PCA,
    KMeans,
    KNNClassifier,
    LinearRegressionGD,
    LogisticRegressionGD,
    Perceptron,
)
from ml_playground.utils import (
    StandardScaler,
    accuracy_score,
    mean_squared_error,
    train_test_split,
)


def make_classification(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    negative = rng.normal(loc=(-2.0, -2.0), scale=0.7, size=(200, 2))
    positive = rng.normal(loc=(2.0, 2.0), scale=0.7, size=(200, 2))
    return np.vstack([negative, positive]), np.array([0] * 200 + [1] * 200)


def make_regression(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(400, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.5 + rng.normal(0, 0.7, size=400)
    return X, y


def main() -> None:
    X_class, y_class = make_classification()
    X_train, X_test, y_train, y_test = train_test_split(
        X_class,
        y_class,
        test_size=0.2,
        seed=1,
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic = LogisticRegressionGD(lr=0.1, epochs=2_000, l2=0.001).fit(
        X_train_scaled,
        y_train,
    )
    knn = KNNClassifier(k=7).fit(X_train_scaled, y_train)
    perceptron = Perceptron(lr=0.1, epochs=30).fit(
        X_train_scaled,
        np.where(y_train == 1, 1, -1),
    )

    X_regression, y_regression = make_regression()
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_regression,
        y_regression,
        test_size=0.2,
        seed=2,
    )
    regression_scaler = StandardScaler().fit(X_train_r)
    linear = LinearRegressionGD(lr=0.05, epochs=3_000, l2=0.01).fit(
        regression_scaler.transform(X_train_r),
        y_train_r,
    )

    clusters = KMeans(k=2, seed=3).fit(X_class)
    pca = PCA(n_components=1).fit(X_class)

    print(f"Logistic accuracy: {accuracy_score(y_test, logistic.predict(X_test_scaled)):.3f}")
    print(f"KNN accuracy: {accuracy_score(y_test, knn.predict(X_test_scaled)):.3f}")
    print(
        "Perceptron accuracy: "
        f"{accuracy_score(np.where(y_test == 1, 1, -1), perceptron.predict(X_test_scaled)):.3f}"
    )
    print(
        "Linear regression MSE: "
        f"{mean_squared_error(y_test_r, linear.predict(regression_scaler.transform(X_test_r))):.3f}"
    )
    print(f"K-means inertia: {clusters.inertia_:.3f}")
    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}")


if __name__ == "__main__":
    main()
