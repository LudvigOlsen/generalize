import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedGroupKFold,
    GroupKFold,
)

from generalize.model.cross_validate.kfolder import KFolder, GroupSpecifiedFolder


def _test_splits(splits, groups_map, num_samples, expected_num_test_indices, by_group):
    all_test_indices = []
    for train, test in splits:
        all_test_indices += list(test)
        assert len(train) + len(test) == num_samples
        # Test grouping was (dis)respected
        train_groups = set([groups_map[i] for i in train])
        test_groups = set([groups_map[i] for i in test])
        if by_group:
            assert not train_groups.intersection(
                test_groups
            ), "Found the same group(s) in train and test."
        else:
            assert train_groups.intersection(
                test_groups
            ), "Did not find the same group(s) in train and test as expected."
    assert len(all_test_indices) == expected_num_test_indices
    assert len(np.unique(all_test_indices)) == expected_num_test_indices


def test_kfolder_strat_group(xy_binary_classification_xl):
    # Stratify=True, by_group=True

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = np.sort(xy_binary_classification_xl["y"])
    num_samples = xy_binary_classification_xl["num_samples"]

    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups_map = dict(enumerate(groups))

    # print([(t,g) for t,g in zip(y, groups)])

    folder = KFolder(
        n_splits=3, stratify=True, by_group=True, shuffle=True, random_state=0
    )

    assert folder.folder.__class__ == StratifiedGroupKFold

    # With all data points being part of testing
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples,
        by_group=True,
    )

    # Add train_only indicators
    groups[np.isin(groups, range(40, 50))] *= -1  # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # With 30 data points being "train only"
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples - 10 * 3,
        by_group=True,
    )


def test_kfolder_strat(xy_binary_classification_xl):
    # Stratify=True, by_group=False
    # NOTE: We don't currently test the stratification succes

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = np.sort(xy_binary_classification_xl["y"])
    num_samples = xy_binary_classification_xl["num_samples"]

    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups_map = dict(enumerate(groups))

    # print([(t,g) for t,g in zip(y, groups)])

    folder = KFolder(
        n_splits=3, stratify=True, by_group=False, shuffle=True, random_state=0
    )

    assert folder.folder.__class__ == StratifiedKFold

    # With all data points being part of testing
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples,
        by_group=False,
    )

    # Add train_only indicators
    groups[np.isin(groups, range(40, 50))] *= -1  # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # With 30 data points being "train only"
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples - 10 * 3,
        by_group=False,
    )


def test_kfolder_group(xy_binary_classification_xl):
    # Stratify=False, by_group=True

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = np.sort(xy_binary_classification_xl["y"])
    num_samples = xy_binary_classification_xl["num_samples"]

    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups_map = dict(enumerate(groups))

    # print([(t,g) for t,g in zip(y, groups)])

    folder = KFolder(
        n_splits=3, stratify=False, by_group=True, shuffle=True, random_state=0
    )

    assert folder.folder.__class__ == GroupKFold

    # With all data points being part of testing
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples,
        by_group=True,
    )

    # Add train_only indicators
    groups[np.isin(groups, range(40, 50))] *= -1  # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # With 30 data points being "train only"
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples - 10 * 3,
        by_group=True,
    )


def test_kfolder(xy_binary_classification_xl):
    # Stratify=False, by_group=False

    # KFold should NOT respect grouping (except the `train_only` part)

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = np.sort(xy_binary_classification_xl["y"])
    num_samples = xy_binary_classification_xl["num_samples"]

    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups_map = dict(enumerate(groups))

    # print([(t,g) for t,g in zip(y, groups)])

    folder = KFolder(
        n_splits=3, stratify=False, by_group=False, shuffle=True, random_state=0
    )

    assert folder.folder.__class__ == KFold

    # With all data points being part of testing
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples,
        by_group=False,
    )

    # Add train_only indicators
    groups[np.isin(groups, range(40, 50))] *= -1  # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # With 30 data points being "train only"
    _test_splits(
        splits=folder.split(X=x, y=y, groups=groups),
        groups_map=groups_map,
        num_samples=num_samples,
        expected_num_test_indices=num_samples - 10 * 3,
        by_group=False,
    )


def test_groups_specified_folder(xy_binary_classification_xl):

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = np.sort(xy_binary_classification_xl["y"])
    num_samples = xy_binary_classification_xl["num_samples"]

    groups = np.arange(len(y))
    splits = np.repeat(range(3), num_samples // 3)
    groups = [f"{g}_|_{f}" for g, f in zip(groups, splits)]

    folder = GroupSpecifiedFolder(
        n_splits=3,
    )

    for train_indices, test_indices in folder.split(X=x, y=y, groups=groups):
        assert len(train_indices) == num_samples / 3 * 2
        assert len(test_indices) == num_samples / 3

    # TODO: Add relevant tests
