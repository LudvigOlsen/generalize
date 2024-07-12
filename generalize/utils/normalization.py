import warnings
import numpy as np


# TODO Add log-space normalization and scaling constant
# TODO add column / row normalization options
def normalize_sum_to_one(
    x, missing_val_fn=np.nanmedian, sum_to=1.0, handle_negatives="raise"
):
    """
    Normalize `x` to sum to one, while scaling for missing data.

    NaN scaling:
        If 40% of our data is NaN, and we normalize the rest
        to sum to 1, the normalized values will not be on the
        same scale as if we only had 10% NaNs.
        Hence, we treat each NaN as if it had some (e.g. the median)
        value during scaling.

    :warns: Throws warning when all elements in `x` are NaN.
        In which case, `x` is returned as is without any normalization.
    :raises: `ValueError` when encountering negative numbers and
        `handle_negatives` is 'raise'.
    :param x: nD `numpy.ndarray` that should sum to 1.
        Elements must be nonnegative.
    :param missing_val_fn: Function for finding the value
        that NaN elements should have during scaling.
        When `None`, the nansum of `x` is used alone
        (i.e. no correction for NaNs).
    :param sum_to: The value that elements should sum to.
        This can be useful when summing to 1 would lead to very small
        values (underflow).
    :param handle_negatives: How to handle negative numbers.
        One of:
            'raise': Raises a `ValueError`.
            'abs': Normalize by the sum of the absolute values.
                Negative numbers will still be negative
                and `x` will *not* sum to one.
            'shift': Adds the absolute minimum value to all elements
                before the normalization.
            'truncate': Set negative numbers to 0.0.
    :returns: `x` divided by its (estimated) sum.
    """

    assert handle_negatives in ["raise", "abs", "shift", "truncate"]

    # Handle negative numbers
    x_signs = None
    if np.isnan(x).all():
        warnings.warn(
            "Normalization: All elements of `x` were NaN. Returning `x` as is."
        )
        return x
    if np.nanmin(x) < 0:
        if handle_negatives == "raise":
            raise ValueError(
                "By default, this normalization does not support negative numbers."
            )
        if handle_negatives == "abs":
            # Find whether a number is negative (-1),
            # zero (0), or positive (1)
            x_signs = np.sign(x)
            x = np.abs(x)
        if handle_negatives == "shift":
            # Add the smallest value
            # It is below 0, so we need to subtract to add the absolute value
            x += np.abs(np.nanmin(x))
        if handle_negatives == "truncate":
            # Set negative numbers to 0
            x[x < 0] = 0.0

    # Calculate estimated sum if values had not been missing
    estimated_sum = _estimate_sum_with_missing_data(x=x, missing_val_fn=missing_val_fn)

    if x_signs is not None:
        # Restore the sign of the negative values
        x *= x_signs

    # Avoid zero-division
    if estimated_sum == 0:
        return np.asarray(x)

    # Divide by the number we need the array to sum to
    estimated_sum /= sum_to

    return np.asarray(x) / estimated_sum


def _estimate_sum_with_missing_data(x, missing_val_fn):
    if missing_val_fn is not None:
        missing_val = missing_val_fn(x)
        _sum = np.nansum(x)
        num_nans = np.count_nonzero(np.isnan(x))
        estimated_sum = _sum + (num_nans * missing_val)
    else:
        estimated_sum = np.nansum(x)
    return estimated_sum


def standardize(x):
    """
    :param x: nD `numpy.ndarray` to standardize.
    :returns: `x` centered and divided by its standard deviations.
    """
    std = x.std()
    if std == 0:
        std = 1
    return (x - x.mean()) / std


def standardize_features_3D(data):
    """
    Standardize feature-wise.

    For each feature for each feature set,
    we subtract the mean and divide by the standard deviation.

    NOTE: NOT intended for images.

    :param data: 3D `numpy.ndarray` with shape (samples, feature sets, features).
    :returns: Standardized version of `data`.
    """
    assert data.ndim == 3
    out = np.zeros_like(data)
    for fs in range(data.shape[1]):
        for feature in range(data.shape[2]):
            out[:, fs, feature] = standardize(data[:, fs, feature])
    return out


def standardize_features_2D(data):
    """
    Standardize feature-wise.

    For each feature, we subtract the mean and divide by the standard deviation.

    NOTE: NOT intended for images.

    :param data: 2D `numpy.ndarray` with shape (samples, features).
    :returns: Standardized version of `data`.
    """
    assert data.ndim == 2
    out = np.zeros_like(data)
    for feature in range(data.shape[1]):
        out[:, feature] = standardize(data[:, feature])
    return out
