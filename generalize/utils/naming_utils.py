

from typing import Union


def format_rank(x: Union[int, str], max_digits: int) -> str:
    """
    Add 0s in front of x until it has `max_digits` digits.

    :raises: `ValueError` when `x` has more digits than `max_digits`.
    :param x: A wholenumber. Either as string or integer.
    :param max_digits: How many digits the final string should have.
    :return: `x` as string with 0s prefixed. Length (number of digits/characters) is `max_digits`.
    """
    current_digits = len(str(x))
    if current_digits > max_digits:
        raise ValueError("`x` has more than `max_digits` digits.")
    if current_digits == max_digits:
        return x
    return "".join(["0"] * (max_digits - current_digits)) + str(x)
