import numpy as np
import pytest

from ml_playground import KMeans


def test_kmeans_finds_stable_centers_and_reports_inertia():
    rng = np.random.default_rng(9)
    X = np.vstack(
        [
            rng.normal(loc=(-3.0, -3.0), scale=0.2, size=(80, 2)),
            rng.normal(loc=(3.0, 3.0), scale=0.2, size=(80, 2)),
        ]
    )

    model = KMeans(k=2, seed=4, max_iter=100, tol=1e-6).fit(X)

    centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]
    np.testing.assert_allclose(centers, [[-3.0, -3.0], [3.0, 3.0]], atol=0.15)
    assert np.isfinite(model.inertia_)
    assert 1 <= model.n_iter_ <= 100


def test_kmeans_is_deterministic_for_same_seed():
    X = np.array([[0.0], [0.1], [3.0], [3.1]])
    first = KMeans(k=2, seed=2).fit(X)
    second = KMeans(k=2, seed=2).fit(X)
    np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)


def test_kmeans_rejects_more_clusters_than_samples():
    with pytest.raises(ValueError, match="number of samples"):
        KMeans(k=3).fit([[0.0], [1.0]])
