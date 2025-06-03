"""Tests for statistics functions within the Model layer."""

import numpy as np
import pytest
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min


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
    with pytest.raises(TypeError):
        daily_min([["Hello", "There"], ["Goodbye", "Now"]])