from typing import List, Optional, Tuple
import numpy as np


def remove_missing_data(
    arrays: List[Optional[np.ndarray]],
) -> Tuple[List[Optional[np.ndarray]], List[int]]:
    """
    Find the indices of values (1D) or rows (2D) of a set of 1- and 2D
    arrays with missing data (`numpy.nan`) and remove them from all
    the arrays.
    E.g. if you have a 1D sample target array and a 2D sample x feature array
    we find the samples with NaNs and remove them from both.

    Parameters
    ----------
    arrays : List of 1- and 2D `numpy.ndarray`s or `None`s
        Any `None` elements are passed through.

    Returns
    -------
    List of possibly smaller 1- and 2D `numpy.ndarray`s without `numpy.nan`s.
        Any `None` elements are passed through.
    List of removed indices.
    """
    assert isinstance(arrays, list)
    index_arrays = []
    for arr in arrays:
        if arr is not None:
            index_arrays.append(missing_data_indices(arr, axis=0))
    missing_indices = np.unique(np.concatenate(index_arrays, axis=-1))
    out_arrays = []
    for arr in arrays:
        if arr is None:
            out_arrays.append(None)
        else:
            out_arrays.append(np.delete(arr, missing_indices, axis=0))
    return out_arrays, list(missing_indices)


def missing_data_indices_1D(x):
    assert x.ndim == 1
    if np.issubdtype(x.dtype, np.number):
        return np.argwhere(np.isnan(x)).flatten()
    if np.issubdtype(x.dtype, np.str_):
        return np.argwhere(x == "nan").flatten()
    raise RuntimeError(f"Could not check NaNs in `x` with dtype {x.dtype}.")


def missing_data_indices_2D(x, axis=0):
    """
    Get indices with either columns or rows with `numpy.nan` in them.

    param axis: 0=rows, 1=columns.
    """
    assert x.ndim == 2
    if np.issubdtype(x.dtype, np.number):
        return np.argwhere(np.isnan(x))[:, axis].flatten()
    if np.issubdtype(x.dtype, np.str_):
        return np.argwhere(x == "nan")[:, axis].flatten()
    raise RuntimeError(f"Could not check NaNs in `x` with dtype {x.dtype}.")


def missing_data_indices(x, axis=0):
    if x.ndim == 1:
        return missing_data_indices_1D(x)
    if x.ndim == 2:
        return missing_data_indices_2D(x, axis=axis)
    raise ValueError("Only supports 1- and 2D arrays for now.")
