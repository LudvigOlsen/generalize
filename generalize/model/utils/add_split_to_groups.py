from typing import Optional, Union, List, Tuple
import numpy as np


def add_split_to_groups(
    groups: Optional[np.ndarray],
    split: Optional[Union[List[Union[int, str]], np.ndarray]],
    weight_loss_by_groups: bool,
    weight_loss_by_class: bool,
    weight_per_split: bool,
    k_inner: Optional[int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Add split IDs to group IDs. I.e. "group_|_split".
    Only when `split` is specified, `weight_per_split=True` and either
    `weight_loss_by_groups=True`, `weight_loss_by_class=True`, or `k_inner=None`.
    """

    if groups is None and split is None:
        return groups, split

    if split is not None and (
        k_inner is None
        or (weight_per_split and (weight_loss_by_groups or weight_loss_by_class))
    ):

        # Ensure groups exist if we are to add to them
        if groups is None:
            groups = np.asarray([str(g) for g in np.arange(len(split))], dtype=object)

        # Add the split ID to the groups to allow weighting loss per split
        if split.ndim == 1:
            groups = np.asarray(
                [group + "_|_" + spl for group, spl in zip(groups, split)]
            )
        else:
            # Groups becomes 2D to have one weighting per split column
            groups = np.asarray(
                [
                    [group + "_|_" + spl for group, spl in zip(groups, split_col)]
                    for split_col in range(split.shape[1])
                ]
            )

    return groups, split
