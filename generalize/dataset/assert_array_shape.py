from typing import List, Optional, Union, Dict
import numpy as np


def assert_shape(
    x: np.ndarray,
    expected_n_dims: Optional[Union[int, List[int]]] = None,
    expected_dim_sizes: Optional[Dict[int, int]] = None,
    x_name: str = "`x`",
) -> None:
    """
    Checks that specified dimensions of a numpy array have the right size.

    Checks the number of dimensions first (when specified),
    then the expected dimension sizes (when specified).

    Parameters
    ----------
    x
        Numpy array to check dimensions of.
    expected_n_dims
        The number of dimensions `x` should have.
        Can be a list of allowed dimensions.
    expected_dim_sizes
        A dict mapping a dimension index to an expected shape.

    x_name
        Name of `x` to use in error message.

    Raises
    ------
    `ValueError`
        When shape expectations are not met or function is used wrongly.
    """

    if expected_n_dims is not None:
        max_expected_n_dims = (
            expected_n_dims
            if isinstance(expected_n_dims, int)
            else max(expected_n_dims)
        )

        if expected_dim_sizes is not None and max_expected_n_dims <= max(
            expected_dim_sizes.keys()
        ):
            raise ValueError(
                "`assert_shape`: `expected_dim_sizes` contained check of "
                "dimension index that's not allowed by `expected_n_dims`: "
                f"{max_expected_n_dims} <= {max(expected_dim_sizes.keys())}."
            )

    # Check shape assertions
    if expected_n_dims is not None:
        if isinstance(expected_n_dims, list):
            if x.ndim not in expected_n_dims:
                raise ValueError(
                    f"Dimension mismatch: {x_name} did not have the expected number of dimensions. "
                    f"Expected one of the following numbers of dimensions: {', '.join([str(n) for n in expected_n_dims])} "
                    f"but got: `{x.ndim}`."
                )
        elif isinstance(expected_n_dims, int):
            if x.ndim != expected_n_dims:
                raise ValueError(
                    f"Dimension mismatch: {x_name} did not have the expected number of dimensions. "
                    f"Expected {expected_n_dims} dimensions but got: `{x.ndim}`."
                )
        else:
            raise ValueError(
                f"`assert_shape`: `expected_n_dims` had wrong type: {type(expected_n_dims)}"
            )

    if expected_dim_sizes is not None and expected_dim_sizes:
        for dim_idx, dim_exp in expected_dim_sizes.items():
            if not x.shape[dim_idx] == dim_exp:
                raise ValueError(
                    f"Shape mismatch: {x_name} did not match the "
                    f"expected size in dimension {dim_idx}. "
                    f"Expected `dataset.shape[{dim_idx}] == {dim_exp}` "
                    f"but got: `{x.shape[dim_idx]}`."
                )
