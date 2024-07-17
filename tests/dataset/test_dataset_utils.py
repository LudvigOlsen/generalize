from numpy.testing import assert_raises  # , assert_array_almost_equal


def _all_values_same(d):
    """
    Check that all values are the same for all keys of a dict.
    """
    if not d:
        return True  # An empty dictionary is considered to have all values the same
    iterator = iter(d.values())
    first_value = next(iterator)
    return all(value == first_value for value in iterator)


def assert_all_values_same(d):
    assert _all_values_same(d)


def test_all_values_same():
    d = {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
        "c": [1, 2, 3, 4],
    }
    assert _all_values_same(d)

    d = {
        "a": [1, 2, 3, 4],
        "b": list(reversed([1, 2, 3, 4])),
        "c": [1, 2, 3, 4],
    }

    assert_raises(AssertionError, assert_all_values_same, d)
