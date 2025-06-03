import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean

test_input = np.array([[2, 0], [4,0]])
test_result = np.array([2, 0])
npt.assert_array_equal(daily_mean(test_input), test_result)

test_input = np.array([[3, 0], [4,0]])
test_result = np.array([3, 0])
npt.assert_array_equal(daily_mean(test_input), test_result)