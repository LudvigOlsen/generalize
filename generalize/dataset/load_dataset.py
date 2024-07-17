import pathlib
from typing import Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
from utipy import Messenger, check_messenger

from generalize.dataset.subset_dataset import select_feature_sets, select_indices
from generalize.dataset.assert_array_shape import assert_shape


def load_dataset(
    path: Union[pathlib.Path, str],
    indices: Optional[List[Union[Tuple[int, int], int]]] = None,
    feature_sets: Optional[List[int]] = None,
    flatten_feature_sets: bool = True,
    as_type: Callable = np.float32,
    name: Optional[str] = None,
    allow_pickle: bool = True,
    expected_shape: Optional[Dict[int, int]] = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> np.ndarray:
    """
    Loads dataset and gets the specified feature sets (when given 3 dimensions).

    When `flatten_feature_sets` is enabled, the chosen feature sets are flattened such that the
    output has shape (samples * feature_sets, features).

    Parameters
    ----------
    path
        Path to dataset file with the extension `.npy`.
        The dataset must have either 2 or 3 dimensions.
        When two dimensions, the expected shape is (samples, features).
        When three dimensions, the expected shape is (samples, feature sets, features).
    indices
        List of indices to get.
        Dataset is 3D: Tuples of (feature set, feature) indices.
        Dataset is 2D: Feature indices.
    feature_sets
        List of indices in the `feature sets` dimension to get.
        Only used when the dataset has 3 dimensions.
        When both this and `indices` are `None`, all feature sets are used.
        When both this and `indices` are set, the feature sets are *additional* to indices.
            Hence, `feature_sets` can get one or more entire feature sets
            and `indices` a few extra features from the other feature sets.
        When this is `None` but `indices` are set, only the values specified
        by `indices` are returned.
    flatten_feature_sets
        Whether to concatenate feature sets or
        keep the 2nd dimension, *when the dataset has 3 dimensions*.
        When `indices` is specified, this usually must be `True`,
        unless the indices selected are the same for all feature sets.
    as_type
        The dtype to set the dataset `numpy.ndarray` as.
    param name
        Name of the dataset. Purely for messaging (printing/logging) purposes.
    allow_pickle
        Whether to allow un-pickling when loading the dataset. See `numpy.load()`.
    expected_shape
        An optional dict mapping a dimension index to an expected shape.
        A `ValueError` is thrown when the dimension expectations are not met.
    messenger
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g. `print` or `log.info`)
        and potential indentation.

    Returns
    -------
    `numpy.ndarray`
        2D array with shape (samples, selected combinations of feature_sets * features).
    """
    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Load dataset
    dataset = np.load(path, allow_pickle=allow_pickle).astype(as_type)

    # Check shape is correct
    assert_shape(
        x=dataset,
        expected_n_dims=[2, 3],
        expected_dim_sizes=expected_shape,
        x_name="Loaded dataset",
    )

    assert (
        feature_sets is None or isinstance(feature_sets, list) and len(feature_sets) > 0
    ), "`feature_sets` must be either `None` or a non-empty list of integers."

    name_err_string = f"({name}) " if name is not None else ""
    messenger(f"{name}:" if name is not None else "Loaded dataset:")
    with messenger.indentation(add_indent=2):
        messenger(f"Dataset has shape: {dataset.shape}")
        if indices is not None and not isinstance(indices, list):
            raise ValueError(
                f"{name_err_string}`indices` can be either a list or `None`, but was {type(indices)}."
            )

        if dataset.ndim == 2:
            if feature_sets is not None:
                raise ValueError(
                    f"{name_err_string}When dataset is 2D, `feature_sets` must be `None`."
                )
            if indices is not None:
                if not isinstance(indices[0], int):
                    raise ValueError(
                        (
                            f"{name_err_string}When dataset is 2D, `indices` should be a list "
                            "that contains integers (or be `None`)."
                        )
                    )
                # Subset dataset
                dataset = dataset[:, indices]
        elif dataset.ndim == 3:
            if indices is not None:
                [dataset], indices = select_indices(
                    datasets=[dataset],
                    indices=indices,
                    add_feature_sets=feature_sets,
                    flatten_feature_sets=flatten_feature_sets,
                )
            else:
                [dataset] = select_feature_sets(
                    datasets=[dataset],
                    indices=feature_sets,
                    flatten_feature_sets=flatten_feature_sets,
                )
            feature_sets_string = (
                f" ({feature_sets})" if feature_sets is not None else ""
            )
            flattening_string = " and flattened" if flatten_feature_sets else ""
            messenger(
                f"Selected{flattening_string} feature sets{feature_sets_string}. "
                f"New shape: {dataset.shape}"
            )

    return dataset


# TODO Improve docs and test code
def load_dataset_index_file(path: Union[pathlib.Path, str]):
    """
    Load txt file with 1/2 columns containing (feature set (optional), feature) indices.
    The file should have no header.
    """
    # Load feature indices
    index_df = pd.read_csv(path, header=None)
    if len(index_df.columns) == 2:
        index_df.columns = ["feature_set", "feature"]
        return list(
            zip(index_df["feature_set"].to_list(), index_df["feature"].to_list())
        )
    else:
        index_df.columns = ["feature"]
        return index_df["feature"].to_list()
