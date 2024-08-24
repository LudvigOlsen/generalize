from typing import Callable, List, Optional, Tuple, Union
import re
import numpy as np
from utipy import Messenger, check_messenger

from generalize.dataset.utils import all_values_same


def select_samples(
    datasets: List[Optional[np.ndarray]],
    labels: List[str],
    labels_to_use: Optional[Union[set, list]] = None,
    collapse_map: Optional[dict] = None,
    positive_label: Optional[str] = None,
    downsample: bool = False,
    shuffle: bool = False,
    rm_prefixed_index: bool = False,
    seed: Optional[int] = 1,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[list, np.ndarray, dict, int]:
    """
    Selects samples (rows) in one or more datasets based from a set of labels. Optionally downsample to the smallest class.

    During downsampling, the same sample indices are selected from all datasets.

    Parameters
    ----------
    datasets
        List of 2D or 3D `numpy.ndarray`s. `None`s are passed through.
        Expected shapes: (samples, feature sets (optional), features).
        All datasets must have the same sample dimension, including the order,
        matching the `labels`.
    labels
        List of labels. One for each sample in the datasets.
    labels_to_use
        List of labels to select data for.
    collapse_map
        Dict with `new_label->labels to collapse`.
        E.g. `'cancer':['colon', 'rectal']`.
    positive_label
        The label that should be the positive class.
        The new class index for that label is returned.
    downsample
        Whether to downsample to the smallest class.
    shuffle
        Whether to shuffle the output. Otherwise the samples are ordered by their label.
    rm_prefixed_index
        Whether to remove prefixed indices (e.g. "0_<label>") from new label names.
    seed
        Random seed to use.
    messenger
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g. `print` or `log.info`)
        and potential indentation.

    Returns
    --------
    list of `numpy.ndarray`s
        New datasets.
        Elements of `datasets` that were `None`
        are passed through.
    numpy.ndarray of str
        New labels.
        In classification, these are label indexes (0,1,2,..).
    dict
        Dict mapping new labels to old labels.
    int
        The index for the new `positive_label`.
    """
    assert isinstance(datasets, list)
    assert isinstance(labels, (list, np.ndarray))
    if len(labels):
        assert isinstance(
            labels[0], str
        ), f"`labels` must be strings, got {type(labels[0])}."
    assert labels_to_use is None or isinstance(labels_to_use, (list, set))
    assert collapse_map is None or isinstance(collapse_map, dict)
    assert positive_label is None or isinstance(positive_label, (int, str))

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    num_input_datasets = len(datasets)
    none_dataset_indices = [i for i, d in enumerate(datasets) if d is None]
    datasets = [d for d in datasets if d is not None]
    if len(datasets) == 0:
        messenger("select_samples(): All datasets were `None`.")
        return [None for _ in range(len(none_dataset_indices))]

    # Ensure same number of samples in all datasets
    num_original_samples = len(datasets[0])
    for data in datasets:
        if len(data) != num_original_samples:
            raise ValueError("All datasets must have the same number of samples.")

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Use all labels when no subset is given
    if labels_to_use is None:
        labels_to_use = list(np.unique(labels))

    messenger(f"Creating dataset(s) with: {labels_to_use}")

    # Create mapping from the collapsed label names
    # to the original labels names
    if collapse_map is None or not collapse_map:
        new_label_to_old_labels = {label: [label] for label in labels_to_use}
    else:
        if messenger.verbose:
            coll_string = "; ".join(
                [
                    f'{k} <- {", ".join([v for v in vals])}'
                    for k, vals in collapse_map.items()
                ]
            )
            messenger(f"Collapsing: {coll_string}")
        collapsed_labels = [v for vals in collapse_map.values() for v in vals]
        non_collapsed_labels = [
            label for label in labels_to_use if label not in collapsed_labels
        ]

        # Case: labels_to_use = [0,1,2,3]; collapse_map = {0:[2,3]}
        # Here the original 0 label will be wrongly discarded
        # So the user must add it to the collapsing ({0:[0,2,3]})
        # or change the key ({2:[2,3]})
        assert not set(collapse_map.keys()).intersection(non_collapsed_labels), (
            "Keys in `collapse_maps` cannot overlap with non-collapsed labels in `labels_to_use`. "
            "Either change the key or add the non-collapsed label to the collapsing. "
            f"Found the following overlaps: {set(collapse_map.keys()).intersection(non_collapsed_labels)}"
        )

        new_label_to_old_labels = {label: [label] for label in non_collapsed_labels}
        new_label_to_old_labels.update(collapse_map)

    # Order `new_label_to_old_labels` by key alphabetically
    # and remove potential prefixed indices (when specified)
    def remove_prefixed_index(s):
        if not rm_prefixed_index:
            return s
        # When present, remove the index followed by an underscore at the start of the string
        return re.sub(r"^\d+_", "", s)

    new_label_to_old_labels = {
        remove_prefixed_index(key): new_label_to_old_labels[key]
        for key in sorted(new_label_to_old_labels.keys())
    }

    # Create additional mappings between labels (old/new) and label indices
    old_label_to_new_label_idx = {}
    new_label_idx_to_new_label = {}
    for label_idx, (k, vals) in enumerate(new_label_to_old_labels.items()):
        for v in vals:
            old_label_to_new_label_idx[v] = label_idx
        new_label_idx_to_new_label[label_idx] = k
    num_new_labels = len(new_label_idx_to_new_label.keys())
    new_label_to_new_label_idx = {v: k for k, v in new_label_idx_to_new_label.items()}

    # Get updated positive label
    new_positive_label = None
    if positive_label is not None:
        positive_label = remove_prefixed_index(positive_label)
        if num_new_labels != 2:
            raise ValueError(
                (
                    f"`positive_label` should not be specified for multiclass targets "
                    f"(output will have {num_new_labels} unique labels, not 2)."
                )
            )
        new_positive_label = new_label_to_new_label_idx[positive_label]

    # Find indices of samples belonging to each of the new labels
    indices = {k: [] for k in new_label_idx_to_new_label.keys()}
    for lab, lab_idx in old_label_to_new_label_idx.items():
        indices[lab_idx] += [i for i, t in enumerate(labels) if t == lab]
    for lab_idx in indices.keys():
        indices[lab_idx] = np.asarray(indices[lab_idx], dtype=np.int32)

    # Downsample the indices to the smallest class
    if downsample:
        messenger("Original samples per label: ")
        for lab_idx in indices.keys():
            messenger(f"{new_label_idx_to_new_label[lab_idx]}: {len(indices[lab_idx])}")

        # Calculate number of samples to downsample to
        min_num_indices = min([len(idxs) for _, idxs in indices.items()])
        messenger(f"Downsampling to {min_num_indices} samples per class.")

        # Downsample indices
        for lab_idx in indices.keys():
            indices[lab_idx] = np.random.choice(
                indices[lab_idx], size=min_num_indices, replace=False
            )

    messenger("Output samples per label: ")

    # Get data and generate labels for chosen samples
    data_splits = [[] for _ in datasets]
    new_labels = []
    for lab_idx, indxs in indices.items():
        for data_idx, data in enumerate(datasets):
            data_splits[data_idx].append(data[indxs, :])
        new_labels += [np.repeat(lab_idx, len(indxs))]
        messenger(
            f"{new_label_idx_to_new_label[lab_idx]} ({lab_idx}): "
            f"{len(indxs)} samples",
            indent=2,
        )

    # Combine labels for the various classes
    new_labels = np.concatenate(new_labels)
    # Combine data for the various classes
    new_datasets = [np.concatenate(splits, axis=0) for splits in data_splits]

    # Test number of samples are the same for labels and all datasets
    if sum([len(new_labels) != len(dset) for dset in new_datasets]) > 0:
        new_dataset_lengths = ", ".join([str(len(d)) for d in new_datasets])
        if len(new_dataset_lengths) > 20:
            new_dataset_lengths = new_dataset_lengths[:20] + "..."
        raise RuntimeError(
            f"The output labels ({len(new_labels)}) and one of the "
            f"output datasets ({new_dataset_lengths}) "
            "had different number of samples."
        )

    # Shuffle arrays together
    if shuffle:
        new_order = np.random.choice(
            np.arange(len(new_labels)), size=len(new_labels), replace=False
        )
        new_labels = new_labels[new_order]
        for data_idx, data in enumerate(new_datasets):
            new_datasets[data_idx] = data[new_order, ...]

    # Add back the `None` datasets
    if none_dataset_indices:
        tmp_new_datasets = []
        next_idx = 0
        for i in range(num_input_datasets):
            if i in none_dataset_indices:
                tmp_new_datasets.append(None)
            else:
                tmp_new_datasets.append(new_datasets[next_idx])
                next_idx += 1
        new_datasets = tmp_new_datasets

    return new_datasets, new_labels, new_label_idx_to_new_label, new_positive_label


def remove_features(
    datasets: List[Optional[np.ndarray]],
    indices: Union[list, np.ndarray],
    copy: bool = True,
) -> List[np.ndarray]:
    """
    Remove features (last dimension) from a list of datasets.

    Parameters
    ----------
    datasets
        A list of `numpy.ndarray`s of which to
        remove the same indices from the last dimension.
        `None`s are passed through.
    indices
        1D list/array of indices to remove
        from the last dimension of each dataset.
    copy
        Whether to return a copy of the datasets
        to avoid altering the input datasets.

    Returns
    -------
    list of `numpy.ndarray`s
        List of (copies of) the same datasets
        with the specified elements removed from
        the last dimension.
        Elements of `datasets` that were `None`
        are passed through.
    """
    new_datasets = [0] * len(datasets)
    for data_idx, data in enumerate(datasets):
        if data is None:
            new_datasets[data_idx] = None
            continue
        if copy:
            data = data.copy()
        new_datasets[data_idx] = np.delete(data, indices, axis=-1)
    return new_datasets


def remove_nan_features(
    datasets: List[Optional[np.ndarray]], copy: bool = True
) -> List[np.ndarray]:
    """
    Remove features (last dimension) where the value is NaN in *ANY* of
    the datasets from all the datasets.

    Parameters
    ----------
    datasets
        A list of `numpy.ndarray`s of which to
        remove the indices from the last dimension that are
        NaN in any of the datasets.
        `None`s are passed through.
    copy
        Whether to return a copy of the datasets
        to avoid altering the input datasets.

    Returns
    -------
    list of `numpy.ndarray`s
        List of (copies of) the same datasets
        with the potentially removed elements from
        the last dimension.
        Elements of `datasets` that were `None`
        are passed through.
    """
    indices = []
    for data in datasets:
        if data is None:
            continue
        # TODO Check that axis should be 0 here? Does it give us
        # the indices for the last dimension?
        indices += list(np.argwhere((np.isnan(data).any(axis=0))))
    indices = sorted(list(np.unique(indices)))
    return remove_features(datasets, indices=indices, copy=copy)


def select_feature_sets(
    datasets: List[Optional[np.ndarray]],
    indices: Optional[List[int]] = None,
    flatten_feature_sets: bool = True,
    copy: bool = True,
) -> List[np.ndarray]:
    """
    Select the feature sets to use and (optionally) flatten them to 2D datasets.

    Parameters
    ----------
    datasets
        List of 3D `numpy.ndarray` with shape (samples, feature sets, features).
        `None`s are passed through.
    indices
        List feature set indices to get.
        When `None`, all feature sets are selected.
    flatten_feature_sets
        Whether to flatten the feature sets or
        keep the second dimension.
    copy
        Whether to return a copy of the datasets
        to avoid altering the input datasets.

    Returns
    -------
    list of `numpy.ndarray`s
        List of 2/3D `numpy.ndarray` with (optionally) combined, selected feature sets.
        2D: With shape (samples, features * selected feature sets).
        3D: With shape (samples, selected feature sets, features).
        Elements of `datasets` that were `None`
        are passed through.
    """
    new_datasets = [0] * len(datasets)
    for data_idx, data in enumerate(datasets):
        if data is None:
            new_datasets[data_idx] = None
            continue

        if copy:
            data = data.copy()

        if data.ndim != 3:
            raise ValueError(
                f"Dataset {data_idx}: Cannot extract feature sets from a {data.ndim}D array."
            )

        if indices is None:
            indices = range(data.shape[1])
        else:
            assert isinstance(indices, list)
            assert len(indices) > 0

        # Subset dataset
        data = data[:, indices, :]

        # Flatten to (samples, features * selected feature sets)
        if flatten_feature_sets and data.ndim == 3:
            data = data.reshape(data.shape[0], -1)

        # Add to list of new datasets
        new_datasets[data_idx] = data

    return new_datasets


def select_indices(
    datasets: List[Optional[np.ndarray]],
    indices: Optional[List[int]] = None,
    add_feature_sets: Optional[List[int]] = None,
    flatten_feature_sets: bool = True,
    copy: bool = True,
    return_updated_indices: bool = False,
) -> Tuple[List[np.ndarray], Optional[List[List[Tuple[int, int]]]]]:
    """
    Select the (feature set, feature) indices to use and flatten them to 2D datasets.

    NOTE: The flattening is usually required as we may have a different number of features
    per feature set. Set `flatten_feature_sets = False` when you know this is not the case!

    Parameters
    ----------
    datasets
        List of 3D `numpy.ndarray` with shape
        (samples, feature sets, features)
        to select features from.
        `None`s are passed through.
    indices
        List of tuples with indices of the feature sets and features to get.
        e.g. [(0, 1), (1,8), ...]
        When `None`, all feature sets and features are selected.
    add_feature_sets
        Additional features sets to get all features for.
    flatten_feature_sets
        Whether to flatten the 3D arrays to 2D arrays.
        Required when not selecting the same features
        for all feature sets. Only disable if
        you know that is the case.
    copy
        Whether to return a copy of the datasets
        to avoid altering the input datasets.
    return_updated_indices
        Whether to return a list of lists with index tuples
        (feature_set, feature) for the included indices.

    Returns
    -------
    list of `numpy.ndarray`s
        List of 2D `numpy.ndarray`s with selected (feature set, feature) values.
        Elements of `datasets` that were `None`
        are passed through.
    list (when `return_updated_indices`) or `None`
        List with updated tuple indices.
    """
    if indices is not None and (
        not isinstance(indices[0], tuple) or len(indices[0]) != 2
    ):
        raise ValueError(
            "When `indices` is specified, it must be a list of tuples of length 2."
        )
    new_datasets = [0] * len(datasets)
    if return_updated_indices:
        new_indices = [0] * len(datasets)
    for data_idx, data in enumerate(datasets):
        if data is None:
            new_datasets[data_idx] = None
            continue

        if copy:
            data = data.copy()

        if data.ndim != 3:
            raise ValueError(
                f"Dataset {data_idx}: Cannot extract feature sets from a {data.ndim}D array."
            )

        if indices is None:
            # Use all but flattened to 2D
            if return_updated_indices:
                new_indices[data_idx] = [
                    (fset, feat)
                    for fset in range(data.shape[1])
                    for feat in range(data.shape[2])
                ]
            new_datasets[data_idx] = (
                data.reshape(data.shape[0], -1) if flatten_feature_sets else data
            )
            continue
        else:
            assert isinstance(indices, list)
            assert len(indices) > 0

        feature_sets = []
        features = []

        # Get additional feature set indices
        if add_feature_sets is not None:
            for fset in add_feature_sets:
                feature_sets += [fset for _ in range(data.shape[-1])]
                features += list(range(data.shape[-1]))

        # Remove already selected indices
        if add_feature_sets is not None:
            indices = [(fs, f) for fs, f in indices if fs not in add_feature_sets]

        # Get indices from `indices`
        indices_feature_sets, indices_features = zip(*indices)
        feature_sets += indices_feature_sets
        features += indices_features

        # Subset data
        if flatten_feature_sets:
            data = data[:, feature_sets, features]
        else:
            # Without flattening
            # Requires subsetting per feature set then concatenating
            # Due to the tuple indexing
            feature_set_to_feature_indices = {fs: [] for fs in set(feature_sets)}
            for fs, fi in zip(feature_sets, features):
                feature_set_to_feature_indices[fs].append(fi)

            if not all_values_same(feature_set_to_feature_indices):
                raise ValueError(
                    "`select_indices`: When `flatten_feature_sets = False`, "
                    "all feature indices must be the same for all feature sets."
                )

            data_per_feature_set = []
            for fs, fis in feature_set_to_feature_indices.items():
                data_per_feature_set.append(data[:, fs, fis])

            try:
                data = np.stack(data_per_feature_set, axis=1)
            except:  # noqa: E722
                raise ValueError(
                    "`select_indices`: Can't stack selected features. Seems to require flattening."
                )

        # Flatten to (samples, selected (feature set, feature) coordinates)
        if data.ndim == 3 and flatten_feature_sets:
            data = data.reshape(data.shape[0], -1)

        new_datasets[data_idx] = data
        if return_updated_indices:
            new_indices[data_idx] = list(zip(feature_sets, features))

    if return_updated_indices:
        return new_datasets, new_indices
    return new_datasets, None
