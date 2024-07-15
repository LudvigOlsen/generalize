from typing import List, Tuple, Union
import numpy as np


def order_by_label(
    datasets: List[np.ndarray], labels: Union[np.ndarray, List[str]], copy: bool = True
) -> Tuple[List[np.ndarray], Union[List[str], np.ndarray]]:
    """
    Order a list of datasets and corresponding labels by the labels.

    Parameters
    ----------
    datasets
        A list of `numpy.ndarray`s that share the
        same labels. The first dimension will be ordered by `labels`.
    labels
        1D list/array of labels to sort (by).
    copy
        Whether to return a copy of the datasets and labels
        to avoid altering the input arrays/lists.

    Returns
    -------
    list of `numpy.ndarray`s
        List of (copies of) the same datasets ordered by the labels.
    list of str
        (Copy of) sorted labels.
    """
    if copy:
        labels = labels.copy()
    sort_indices = np.argsort(labels)
    labels_sorted = labels[sort_indices]
    new_datasets = [0] * len(datasets)
    for data_idx, data in enumerate(datasets):
        if copy:
            data = data.copy()
        new_datasets[data_idx] = data[sort_indices, :]
    return new_datasets, labels_sorted
