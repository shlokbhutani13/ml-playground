import numpy as np
from mlp.utils import train_test_split, standardize_fit, standardize_transform, accuracy, mse
from mlp.linear_models import LinearRegressionGD
from mlp.logistic import LogisticRegressionGD
from mlp.knn import KNNClassifier
from mlp.kmeans import KMeans
from mlp.pca import PCA
from mlp.perceptron import Perceptron

def make_classification(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n//2, 2))
    X1 = rng.normal(loc=(2, 2), scale=1.0, size=(n//2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

def make_regression(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    y = 2.0 * X[:,0] - 3.0 * X[:,1] + 0.5 + rng.normal(0, 0.7, size=n)
    return X, y

def main():
    # ===== Classification demo =====
    X, y = make_classification()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=1)
    mu, sig = standardize_fit(Xtr)
    Xtr_s = standardize_transform(Xtr, mu, sig)
    Xte_s = standardize_transform(Xte, mu, sig)

    logreg = LogisticRegressionGD(lr=0.1, epochs=2000, l2=0.001).fit(Xtr_s, ytr)
    preds = logreg.predict(Xte_s)
    print("LogReg accuracy:", accuracy(yte, preds))

    knn = KNNClassifier(k=7).fit(Xtr_s, ytr)
    print("KNN accuracy:", accuracy(yte, knn.predict(Xte_s)))

    perc = Perceptron(lr=0.1, epochs=30).fit(Xtr_s, (ytr*2-1))
    print("Perceptron accuracy:", accuracy((yte*2-1), perc.predict(Xte_s)))

    # ===== Regression demo =====
    Xr, yr = make_regression()
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, seed=2)
    mu, sig = standardize_fit(Xtr)
    Xtr_s = standardize_transform(Xtr, mu, sig)
    Xte_s = standardize_transform(Xte, mu, sig)

    lin = LinearRegressionGD(lr=0.05, epochs=3000, l2=0.01).fit(Xtr_s, ytr)
    yhat = lin.predict(Xte_s)
    print("LinearReg MSE:", mse(yte, yhat))

    # ===== KMeans + PCA demo =====
    Xc, _ = make_classification(n=500, seed=3)
    km = KMeans(k=2, seed=3).fit(Xc)
    labels = km.predict(Xc)
    print("KMeans clusters:", np.bincount(labels))

    pca = PCA(n_components=1).fit_transform(Xc)
    print("PCA reduced shape:", pca.shape)

if __name__ == "__main__":
    main()
