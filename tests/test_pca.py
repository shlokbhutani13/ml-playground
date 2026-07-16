import numpy as np
import pytest

from ml_playground import PCA


def test_pca_reports_orthonormal_components_and_ordered_variance():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 3))
    X[:, 1] = 2.5 * X[:, 0] + rng.normal(scale=0.1, size=400)

    model = PCA(n_components=2).fit(X)
    transformed = model.transform(X)

    assert transformed.shape == (400, 2)
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(2), atol=1e-7)
    assert model.explained_variance_[0] >= model.explained_variance_[1]
    assert np.all(model.explained_variance_ratio_ >= 0)
    assert model.explained_variance_ratio_.sum() <= 1.0 + 1e-12
    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0], atol=1e-12)


def test_pca_rejects_too_many_components():
    with pytest.raises(ValueError, match="between 1 and 2"):
        PCA(n_components=3).fit([[0.0, 1.0], [1.0, 0.0]])
