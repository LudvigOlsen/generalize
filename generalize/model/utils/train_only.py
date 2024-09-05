from typing import Optional, Union, List, Callable, Tuple
import numpy as np
from utipy import Messenger, check_messenger


def detect_train_only(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    split: Optional[Union[List[Union[int, str]], np.ndarray]],
    weight_per_split: bool,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[np.ndarray, np.ndarray]:
    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Ensure split is 2D array (will convert back to 1d when appropriate)
    if split is not None:
        if isinstance(split, list):
            split = np.expand_dims(np.asarray(split), axis=1)
        elif split.ndim == 1:
            split = np.expand_dims(split, axis=1)
        elif split.ndim != 2:
            raise ValueError("`split` must be 1- or 2D numpy array or a list.")

    messenger("Detecting train-only data points")
    if groups is not None:
        groups = np.asarray([str(g).strip() for g in groups], dtype=object)
        training_group_flags = ["train_only" in g for g in groups]
        training_groups = groups[training_group_flags]
        messenger(
            f"Detected {len(set(training_groups))} unique train-only groups "
            f"({sum(training_group_flags)} data points)",
            indent=2,
        )
        if any(["no_eval(" in g for g in training_groups]):
            raise ValueError(
                "A train-only group also contained the `no_eval` wrapper. "
                "Only one of these group wrappers can be applied per group."
            )
    elif weight_per_split and split is not None:
        groups = np.asarray([str(g) for g in np.arange(len(y))], dtype=object)
        training_group_flags = [False for _ in groups]  # All False

    if split is not None:
        train_only_bools = [
            ["train_only" in s for s in split[:, col_idx]]
            for col_idx in range(split.shape[1])
        ]
        assert _all_lists_equal(train_only_bools), (
            "When providing multiple splits, all splits must have "
            "`train_only` indicators for the same sample indices (or none at all)."
        )
        messenger(
            f"Detected {sum(['train_only' in s for s in split[:, 0]])} "
            "train-only data points in `split`",
            indent=2,
        )

        # Add train_only indicators from `groups` to `split`
        if groups is not None:
            if sum(training_group_flags) > 0:
                for col_idx in range(split.shape[1]):
                    split[training_group_flags, col_idx] = [
                        f"train_only({s})" if "train_only" not in s else s
                        for s in split[training_group_flags, col_idx]
                    ]

            messenger(
                "Total train-only data points: "
                f"{sum(['train_only' in s for s in split[:, 0]])}",
                indent=2,
            )

        training_split_flags = ["train_only" in s for s in split[:, 0]]
        if any(training_split_flags):
            # Specify the train_only data points in `groups`
            # When `groups` is None, give each of the other data points their own group
            if groups is None:
                groups = np.asarray([str(g) for g in np.arange(len(y))], dtype=object)
                training_group_flags = [False for _ in groups]  # All False
            else:
                # Ensure arbitrary length strings are allowed
                groups = np.asarray(groups, dtype=object)

            # Transfer split train-only statuses to groups
            groups[training_split_flags] = [
                f"train_only({g})" if "train_only" not in g else g
                for g in groups[training_split_flags]
            ]

        if split.shape[1] == 1:
            # If originally 1D split
            split = split[:, 0]

    return groups, split


def _all_lists_equal(lists: List[list]) -> bool:
    if not lists:
        return True
    reference_list = lists[0]
    return all(l == reference_list for l in lists)
