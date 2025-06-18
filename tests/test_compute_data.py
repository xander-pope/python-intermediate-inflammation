"""Tests for functions within the compute_data layer."""

import numpy as np
import numpy.testing as npt
import math
import pytest
from inflammation.compute_data import *


@pytest.mark.parametrize(
        ("test, expect_raises"),
        [
            ['/home/xj3041/training/python-intermediate-inflammation/data', None],
            ['/home/xj3041/training/python-intermediate-inflammation/inflammation', ValueError],
        ],
        ids=["existing data directory", "non-existing data directory"]
    )
def test_load_inflammation_data(test, expect_raises):
    """Test for loading inflammation csv files"""
    csv_data = CSVDataSource(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            csv_data.load_inflammation_data()
    else:
        csv_data.load_inflammation_data()


@pytest.mark.parametrize(
        ("test, expected"),
        [
            ['/home/xj3041/training/python-intermediate-inflammation/data',
             ([ 0.        , 0.22510286, 0.18157299, 0.1264423 , 0.9495481 ,
                0.27118211, 0.25104719, 0.22330897, 0.89680503, 0.21573875,
                1.24235548, 0.63042094, 1.57511696, 2.18850242, 0.3729574 ,
                0.69395538, 2.52365162, 0.3179312 , 1.22850657, 1.63149639,
                2.45861227, 1.55556052, 2.8214853 , 0.92117578, 0.76176979,
                2.18346188, 0.55368435, 1.78441632, 0.26549221, 1.43938417,
                0.78959769, 0.64913879, 1.16078544, 0.42417995, 0.36019114,
                0.80801707, 0.50323031, 0.47574665, 0.45197398, 0.22070227])]
        ]
    )
def test_analyse_data(test, expected):
    """Test for correct graph data"""
    csv_data = CSVDataSource(test)
    data = csv_data.load_inflammation_data()
    result = analyse_data(data)
    npt.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    ("data, expected"),
    [
        ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
        ([[[0, 1, 0]], [[0, 2, 0]]], [0, math.sqrt(0.25), 0]),
        ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
    ],
    ids=[
            'Two patients in same file',
            'Two patients in different files',
            'Two identical patients in two different files'
        ]
)
def test_standard_deviation(data, expected):
    """Test for calculating the standard deviation function"""
    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected)
