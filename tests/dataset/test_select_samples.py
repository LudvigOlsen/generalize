import numpy as np
import pytest

from generalize.dataset.subset_dataset import select_samples


def test_select_samples():
    seed = 1
    np.random.seed(seed)

    datasets = [np.random.normal(size=(30, 3, 3)), np.random.normal(size=(30, 3, 3))]

    labels = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5

    labels_to_use = ["A", "B", "C", "D"]

    collapse_map = {"1": ["A", "B"], "2": ["C", "D"]}

    positive_label = "1"

    new_datasets, new_labels, new_label_idx_to_new_label, new_positive_label = (
        select_samples(
            datasets=datasets,
            labels=labels,
            labels_to_use=labels_to_use,
            collapse_map=collapse_map,
            positive_label=positive_label,
            seed=seed,
        )
    )

    assert new_datasets[0].shape == (20, 3, 3)
    assert new_datasets[0].shape == new_datasets[1].shape
    assert all(new_labels == [0] * 10 + [1] * 10)
    assert new_label_idx_to_new_label == {0: "1", 1: "2"}
    assert new_positive_label == 0


def test_select_samples_collapses_to_self():
    # This tests what happens when the collapse map
    # has overlapping keys with the labels_to_use list

    seed = 1
    np.random.seed(seed)

    datasets = [np.random.normal(size=(30, 3, 3)), np.random.normal(size=(30, 3, 3))]

    labels = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5 + ["E"] * 5 + ["F"] * 5

    labels_to_use = ["A", "B", "C", "D"]

    collapse_map = {"A": ["A", "B"], "C": ["C", "D"]}

    positive_label = "A"

    new_datasets, new_labels, new_label_idx_to_new_label, new_positive_label = (
        select_samples(
            datasets=datasets,
            labels=labels,
            labels_to_use=labels_to_use,
            collapse_map=collapse_map,
            positive_label=positive_label,
            seed=seed,
        )
    )

    assert new_datasets[0].shape == (20, 3, 3)
    assert new_datasets[0].shape == new_datasets[1].shape
    assert all(new_labels == [0] * 10 + [1] * 10)
    assert new_label_idx_to_new_label == {0: "A", 1: "C"}
    assert new_positive_label == 0

    # Keys and vals are switched
    collapse_map = {"C": ["A", "B"], "A": ["C", "D"]}

    new_datasets, new_labels, new_label_idx_to_new_label, new_positive_label = (
        select_samples(
            datasets=datasets,
            labels=labels,
            labels_to_use=labels_to_use,
            collapse_map=collapse_map,
            positive_label=positive_label,
            seed=seed,
        )
    )

    assert new_datasets[0].shape == (20, 3, 3)
    assert new_datasets[0].shape == new_datasets[1].shape
    assert all(new_labels == [0] * 10 + [1] * 10)
    assert new_label_idx_to_new_label == {0: "C", 1: "A"}
    assert new_positive_label == 1

    # Keeping non-collapsed labels
    labels_to_use = ["A", "B", "C", "D", "E", "F"]

    new_datasets, new_labels, new_label_idx_to_new_label, new_positive_label = (
        select_samples(
            datasets=datasets,
            labels=labels,
            labels_to_use=labels_to_use,
            collapse_map=collapse_map,
            seed=seed,
        )
    )

    assert new_datasets[0].shape == (30, 3, 3)
    assert new_datasets[0].shape == new_datasets[1].shape
    assert all(new_labels == [0] * 5 + [1] * 5 + [2] * 10 + [3] * 10)
    assert new_label_idx_to_new_label == {0: "E", 1: "F", 2: "C", 3: "A"}

    # Collapse key is a non-collapsed label
    # Should raise error (as the original E would otherwise be discarded)
    collapse_map = {"E": ["A", "B"], "A": ["C", "D"]}

    with pytest.raises(Exception):
        select_samples(
            datasets=datasets,
            labels=labels,
            labels_to_use=labels_to_use,
            collapse_map=collapse_map,
            seed=seed,
        )
