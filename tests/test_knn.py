import numpy as np
import pytest

from ml_playground import KNNClassifier


def test_knn_classifies_known_neighborhoods_and_probabilities():
    X = np.array([[0.0], [0.2], [1.0], [1.2]])
    y = np.array([0, 0, 1, 1])
    model = KNNClassifier(k=3).fit(X, y)

    np.testing.assert_array_equal(model.predict([[0.1], [1.1]]), [0, 1])
    probabilities = model.predict_proba([[0.1], [1.1]])
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_knn_breaks_ties_using_smallest_label():
    model = KNNClassifier(k=2).fit([[0.0], [2.0]], [2, 1])
    assert model.predict([[1.0]])[0] == 1


def test_knn_rejects_k_larger_than_training_set():
    with pytest.raises(ValueError, match="training samples"):
        KNNClassifier(k=3).fit([[0.0], [1.0]], [0, 1])
