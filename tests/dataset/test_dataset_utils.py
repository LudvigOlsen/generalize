from numpy.testing import assert_raises  # , assert_array_almost_equal
from generalize.dataset.utils import all_values_same


def assert_all_values_same(d):
    assert all_values_same(d)


def test_all_values_same():
    d = {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
        "c": [1, 2, 3, 4],
    }
    assert all_values_same(d)

    d = {
        "a": [1, 2, 3, 4],
        "b": list(reversed([1, 2, 3, 4])),
        "c": [1, 2, 3, 4],
    }

    assert_raises(AssertionError, assert_all_values_same, d)
