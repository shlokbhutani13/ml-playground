import numpy as np
import pytest

from ml_playground._validation import as_2d_float_array, check_feature_count, check_X_y


def test_as_2d_float_array_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one sample"):
        as_2d_float_array([], name="X")


def test_as_2d_float_array_rejects_one_dimensional_features():
    with pytest.raises(ValueError, match="2-dimensional"):
        as_2d_float_array([1, 2, 3], name="X")


def test_as_2d_float_array_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        as_2d_float_array([[1.0, np.nan]], name="X")


def test_check_x_y_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="same number of samples"):
        check_X_y([[1.0], [2.0]], [1])


def test_check_feature_count_rejects_wrong_width():
    with pytest.raises(ValueError, match="2 features"):
        check_feature_count(np.ones((3, 1)), expected=2)
