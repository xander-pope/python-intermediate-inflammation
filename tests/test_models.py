"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise


@pytest.mark.parametrize(
        ("test", "expected"),
        [
            [[[0, 0], [0, 0], [0, 0]], [0, 0]],
            [[[1, 2], [3, 4], [5, 6]], [3, 4]],
            [[[7, 2], [3, 8], [2, -7]], [4, 1]],
        ],
        ids=["zeros", "positive integers", "negative integers"]
    )
def test_daily_mean_integers(test, expected):
    """Test that mean function works for an array of integers."""
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
        ("test", "expected"),
        [
            [[[0, 0], [0, 0], [0, 0]], [0, 0]],
            [[[1, 2], [3, 4], [5, 6]], [5, 6]],
            [[[5, 2], [3, 6], [-1, 7]], [5, 7]],
        ],
        ids=["zeros", "positive integers", "negative integers"]
    )
def test_daily_max_integers(test, expected):
    """Test that max function works for an array of integers."""
    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
        ("test", "expected"),
        [
            [[[0, 0], [0, 0], [0, 0]], [0, 0]],
            [[[1, 2], [3, 4], [5, 6]], [1, 2]],
            [[[5, 2], [3, 6], [-1, 7]], [-1, 2]],
        ],
        ids=["zeros", "positive integers", "negative integers"]
    )
def test_daily_min_integers(test, expected):
    """Test that min function works for an array of integers."""
    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_error_string():
    """Test that min function doesn't accept strings."""
    with pytest.raises(TypeError):
        daily_min([["Hello", "There"], ["Goodbye", "Now"]])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         None
         ),
        ([[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         ValueError
         ),
        ([[[1, 2], [3, 4]], [[4, 5], [6, 7]]],
         [[[0.5, 1], [0.75, 1]], [[0.8, 1], [6/7, 1]]],
         ValueError
         ),
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(np.array(test))
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
