from typing import Optional, Callable, Tuple, List
import numpy as np
from utipy import Messenger, check_messenger


def detect_no_evaluation(
    groups: Optional[np.ndarray],
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[np.ndarray, List[int]]:
    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    messenger("Detecting no-evaluation data points")
    if groups is not None:
        groups = np.asarray([str(g).strip() for g in groups], dtype=object)
        no_eval_group_flags = ["no_eval" in g for g in groups]
        no_eval_group_indices = np.argwhere(no_eval_group_flags).flatten().tolist()
        no_eval_groups = groups[no_eval_group_flags]
        messenger(
            f"Detected {len(set(no_eval_groups))} unique no-evaluation groups "
            f"({sum(no_eval_group_flags)} data points)",
            indent=2,
        )
        if any(["train_only(" in g for g in no_eval_groups]):
            raise ValueError(
                "A no-evaluation group also contained the `train_only` wrapper. "
                "Only one of these group wrappers can be applied per group."
            )
        groups[no_eval_group_flags] = np.asarray(
            [_clean_group(g) for g in groups[no_eval_group_flags]]
        )
        return groups, no_eval_group_indices
    else:
        return groups, []


def _clean_group(group: str) -> str:
    """
    Checks the group and removes the
    `no_eval()` wrapper.
    """
    group = group.strip()
    tag = "no_eval("
    if group[: len(tag)] != tag:
        raise ValueError(
            f"When supplying the {tag}) wrapper to a group, "
            "it must be the first / most outer part of the group ID. "
            f"Got: {group}."
        )
    if group[-1] != ")":
        raise ValueError(
            f"When supplying the {tag}) wrapper to a group, the last "
            "character in the group must be a closing parenthesis ')'. "
            f"Got: {group}."
        )
    return group[:-1].replace(tag, "")
