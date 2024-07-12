from typing import Optional, Callable, Union
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedGroupKFold,
    GroupKFold,
)
from sklearn.utils import indexable, check_random_state


class KFolder(KFold):
    def __init__(
        self,
        n_splits: int,
        stratify: bool,
        by_group: bool,
        shuffle: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """
        Wrapper class for selecting the proper *KFold class and allowing
        group IDs wrapped in `train_only` to only be used for training.

        Parameters
        ----------
        n_splits: int,
        stratify: bool,
        by_group: bool,
        shuffle: bool,
        random_state: Optional[Union[int, np.random.RandomState]]

        n_splits : int
            Number of folds. Must be at least 2.
        shuffle : bool, default=False
            See shuffle for the relevant *KFold class
            (one of {KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold}).
            When `groups` contain `train_only`, the training set
            is shuffled after concatenating these "train_only"
            indices and the training indices from the split.
        random_state : int or RandomState instance, default=None
            When `shuffle` is True, `random_state` affects the ordering of the
            indices, which controls the randomness of each fold for each class.
            Otherwise, leave `random_state` as `None`.
            Pass an int for reproducible output across multiple function calls.

        """
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stratify = stratify
        self.by_group = by_group
        self.folder = self._get_folder()(
            n_splits=self.n_splits, shuffle=shuffle, random_state=random_state
        )

    def _get_folder(self) -> Callable:
        return {
            (True, True): StratifiedGroupKFold,
            (True, False): lambda n_splits, shuffle, random_state: GroupKFold(
                n_splits=n_splits
            ),
            (False, True): StratifiedKFold,
            (False, False): KFold,
        }[(self.by_group, self.stratify)]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Data points where `groups` contains `train_only`
        are always put in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            **Train only**: Wrap group ID in `"train_only(ID)"` (where ID is the group ID)
            for samples that should only be used in the training set
            (never tested on).

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is not None:
            groups = np.array([str(g) for g in groups])
        X, y, groups = indexable(X, y, groups)

        if groups is not None:
            is_train_groups = ["train_only" in g for g in groups]

        if groups is None or not any(is_train_groups):
            for train, test in self.folder.split(X, y, groups):
                yield train, test
        else:
            train_only_indices = np.where(is_train_groups)[0]
            rest_indices = np.delete(np.arange(len(X)), train_only_indices)
            indices_map = {
                i: i_rest for i, i_rest in zip(range(len(rest_indices)), rest_indices)
            }
            for train_indices, test_indices in self.folder.split(
                X=X[rest_indices],
                y=y[rest_indices],
                groups=groups[rest_indices] if groups is not None else None,
            ):
                train_indices = np.concatenate(
                    [
                        train_only_indices,
                        np.asarray([indices_map[i] for i in train_indices]),
                    ]
                )
                test_indices = np.asarray([indices_map[i] for i in test_indices])
                if self.shuffle:
                    random_state = check_random_state(self.random_state)
                    random_state.shuffle(train_indices)
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.
        """
        return self.n_splits

    def _iter_test_indices(self, X, y, groups):
        """Implemented to catch errors / unexpected behavior only."""
        raise NotImplementedError(
            "`_iter_test_indices()` should not be used by this class."
        )


def specified_folds_iterator(folds: np.ndarray, yield_names: bool = False):
    """
    An iterable yielding (train, test) splits as arrays of indices based on specified folds.

    Parameters
    ----------
    folds : np.ndarray
        An array of fold identifiers (numbers or strings) for the samples.
        Wrap fold ID with 'train_only(ID)' for samples that should always
        be in the training sets.
    yield_names : bool
        Whether to yield the fold IDs instead of indices.

    Yields
    ------
    tuple of np.ndarrays
        (train, test) splits as arrays of indices.
    OR
    str
        Name of split (the fold ID used for testing).
    """
    if not isinstance(folds, np.ndarray):
        raise TypeError("`folds` was not a `numpy.ndarray`.")
    for fold_id in np.unique(folds):
        if "train_only" in fold_id:
            continue
        if yield_names:
            yield fold_id
        else:
            yield (
                np.argwhere(folds != fold_id).flatten(),  # Train
                np.argwhere(folds == fold_id).flatten(),  # Test
            )


class GroupSpecifiedFolder(KFold):
    def __init__(self, n_splits: int) -> None:
        """
        Folder class that selects the fold based on the last term in `groups`.
        Specify fold names by adding "_|_<fold_ID>" at the end of the group names.
        The group names are split by "_|_" and the last result is considered
        the fold ID.
        Allows the `train_only` (str or int) group to only be used for training.

        Parameters
        ----------
        n_splits : int
            Number of folds. Must be at least 2.
        """
        super().__init__(n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Data points where `groups` contains `train_only`
        are always put in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
            **Train only**: Wrap group ID in `"train_only(ID)"` (where ID is the group ID)
            for samples that should only be used in the training set
            (never tested on).
            **NOTE**: Must be specified. Placed third to fit `sklearn` convention.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "`GroupSpecifiedFolder` requires `groups` to be passed but `groups` was `None`."
            )
        if not all(["_|_" in g for g in groups]):
            raise ValueError(
                "`GroupSpecifiedFolder`: `groups` must have '_|_<Fold_ID>' appended."
            )

        groups = np.array([str(g) for g in groups])
        X, y, groups = indexable(X, y, groups)

        fold_ids = [g.split("_|_")[-1] for g in groups]
        fold_ids = [
            f"train_only({f})" if "train_only" in g else f
            for f, g in zip(fold_ids, groups)
        ]

        yield from specified_folds_iterator(folds=np.asarray(fold_ids))

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.
        """
        return self.n_splits

    def _iter_test_indices(self, X, y, groups):
        """Implemented to catch errors / unexpected behavior only."""
        raise NotImplementedError(
            "`_iter_test_indices()` should not be used by this class."
        )
