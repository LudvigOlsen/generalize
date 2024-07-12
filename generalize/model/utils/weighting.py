from typing import Optional, Tuple, Dict
import numpy as np


def calculate_sample_weight(
    y: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    weight_loss_by_groups: bool,
    weight_loss_by_class: bool,
    weight_per_split: bool,
    weight_splits_equally: bool,
    split_weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sample weights based on labels and groups.

    Parameters
    ----------
    weight_per_split
        Whether to perform the weighting per split
        (defined by second term in `groups`, split by `_|_`).
        NOTE: When weighting by class, we first weight per split
        and then globally based on the per-split group and
        class weighting. When one split only contains a single class,
        the global weighting will be unbalanced when doing it per
        split. But by doing it first per split and then adjusting those
        weights to the global balance, we get the best of both worlds.
    """

    # Output array to multiply weightings on to
    sample_weights = np.ones(shape=(len(y)), dtype=np.float64)

    if split_weights is not None:
        _, split_ids, unique_slit_ids = split_groups_and_splits(groups)

        # Normalize split_weights
        total_split_weights = np.mean(
            [w for k, w in split_weights.items() if k in unique_slit_ids]
        )
        split_weights = {
            split_id: weight / total_split_weights
            for split_id, weight in split_weights.items()
        }

        # Weight `sample_weights` by the split weights
        for split_id in unique_slit_ids:
            sample_weights[split_ids == split_id] *= split_weights[split_id]

    # Ensure inputs are arrays
    if y is not None:
        y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
    if y is not None and groups is not None and len(y) != len(groups):
        raise ValueError(
            f"When both specified, `y` ({len(y)}) and `groups` ({len(groups)}) must have the same length."
        )

    if weight_loss_by_class:
        if y is None:
            raise ValueError(
                "Cannot weight loss by inverse class frequencies when `y` is `None`."
            )
        if np.issubdtype(y.dtype, np.floating):  #  and not np.mod(y, 1).sum() == 0
            raise ValueError(
                "`weight_loss_by_class` was enabled but `y` was floating point. "
                "Can only weight by classes in classification tasks."
            )

    if weight_per_split:
        if groups is None:
            raise ValueError("`weight_per_split` was enabled but `groups` was `None`.")

        # In this case, we have added the split IDs
        # to the groups (e.g. `group1_|_split1`)
        groups, split_ids, unique_slit_ids = split_groups_and_splits(groups)

        if weight_loss_by_groups:
            for split_id in unique_slit_ids:
                split_groups = groups[split_ids == split_id].copy()
                # Disable scaling here. If all groups have 10 data points,
                # we want them balanced overall (0.1) - not for all of them to
                # have the weight 1
                split_unique_groups, split_weights = calculate_weights(
                    split_groups, normalize=False
                )
                for group, weight in zip(split_unique_groups, split_weights):
                    sample_weights[
                        (split_ids == split_id) & (groups == group)
                    ] *= weight
            del split_weights, split_groups

        if weight_loss_by_class:
            for split_id in unique_slit_ids:
                split_labels = y[split_ids == split_id].copy()
                split_unique_labels, split_weights = calculate_weights(
                    split_labels,
                    # When weighting by groups, each group should only
                    # count 1 in the label frequencies
                    # So we weight the label frequencies by the group weight
                    # NOTE: Requires group weighting to come prior to class weighting
                    weights=sample_weights[split_ids == split_id],
                )
                for lab, weight in zip(split_unique_labels, split_weights):
                    sample_weights[(split_ids == split_id) & (y == lab)] *= weight
            del split_weights, split_labels

        if weight_splits_equally:
            for split_id in unique_slit_ids:
                sample_weights[split_ids == split_id] /= np.sum(
                    sample_weights[split_ids == split_id]
                )

    else:
        if weight_loss_by_groups:
            if groups is None:
                raise ValueError(
                    "`weight_loss_by_groups` was enabled but `groups` was `None`."
                )

            # Calculate sample weights based on group sizes
            unique_groups, group_weights = calculate_weights(groups, normalize=False)

            for group, weight in zip(unique_groups, group_weights):
                sample_weights[groups == group] *= weight

            del weight, unique_groups

    # Weight globally by class size
    # NOTE: In case one of the splits only contain
    # one class, the global weighting won't be balanced
    # when doing it per split, so we first weight per split
    # and then globally based on the per-split weights
    # this way, we get the best of both worlds
    if weight_loss_by_class:
        # Calculate sample weights based on class sizes
        unique_labels, label_weights = calculate_weights(
            y,
            # When weighting by groups, each group should only
            # count 1 in the label frequencies
            # So we weight the label frequencies by the group weight
            # NOTE: Requires group weighting to come prior to class weighting
            weights=sample_weights,
        )

        for label, weight in zip(unique_labels, label_weights):
            sample_weights[y == label] *= weight

        del weight, unique_labels

    # Sum-to-N normalization
    sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)

    # if groups is not None:
    #     print(list(zip(list(groups), list(sample_weights))))

    return sample_weights, groups


def split_groups_and_splits(groups):
    groups, split_ids = zip(*[g.split("_|_") for g in groups])
    groups = np.asarray(groups)
    split_ids = np.asarray(split_ids)
    unique_slit_ids = set(split_ids)
    return groups, split_ids, unique_slit_ids


def weighted_freqs(v, weights) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the sum of weights for each unique element.
    As opposed to each element counting 1, they will count their weight.
    """
    uniques = np.unique(v)
    weight_sums = np.asarray([weights[v == lab].sum() for lab in uniques])
    return uniques, weight_sums


def calculate_weights(
    v: np.ndarray, weights: Optional[np.ndarray] = None, normalize=True, scale=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a weight for each element in `v`
    based on the frequency of the unique elements in `v`.
    Supply `weights` to weight the frequencies. The
    frequencies becomes the sum of weights for each of
    the unique elements in `v`.

    normalize
        Whether to divide the inverse frequencies by the sum of inverse frequencies
        and (when `scale=True`) scale by the number of unique values.
    scale:
        Whether to multiply the sum-to-one weights by the number of unique elements in `v`
        to have them centered around 1. Note that it's the weights for
        the unique elements that are centered.
        Ignored when `normalize` is disabled.
    """
    if weights is not None:
        weights = np.asarray(weights)
        uniques, freqs = weighted_freqs(v=v, weights=weights)
    else:
        uniques, freqs = np.unique(v, return_counts=True)
    inv_freqs = 1.0 / freqs
    if normalize:
        inv_freqs = (inv_freqs / np.sum(inv_freqs)) * (len(inv_freqs) if scale else 1)
    return uniques, inv_freqs
