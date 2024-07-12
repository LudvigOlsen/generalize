
import numbers
import numpy as np


def scalar_safe_div(x, y):
    """
    Safe division of two scalars.
    When `y` is 0, it returns `np.nan`.
    """
    assert isinstance(x, numbers.Number)
    assert isinstance(y, numbers.Number)
    return np.nan if y == 0 else x / y
