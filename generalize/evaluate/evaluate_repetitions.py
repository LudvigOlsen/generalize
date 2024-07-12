from typing import Callable, Optional, List
import numpy as np
from utipy import Messenger, check_messenger

from generalize.evaluate.evaluate import Evaluator
from generalize.evaluate.evaluate_splits import (
    evaluate_existing_splits,
    evaluate_splits,
)
from generalize.evaluate.summarize_scores import summarize_data_frame


# TODO Add typing
def evaluate_repetitions(
    predictions_list,
    task,
    targets=None,
    targets_list=None,
    groups=None,
    groups_list=None,
    positive=None,
    splits_list=None,
    thresholds=None,
    target_labels=None,
    identifier_cols_dict=None,
    summarize_splits: bool = False,
    summarize_splits_allow_differing_members: bool = False,
    eval_names: Optional[List[str]] = None,
    eval_idx_colname: str = "Repetition",
    split_id_colname: str = "Fold",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """

    Parameters
    ----------

    splits_list : list of lists / 1D `numpy.ndarrays` or `None``
        Lists of data splits (e.g. folds) to evaluate by.
        When specified, evaluation is first performed per split
        and averaged per repetition, then averaged again.
        Else, all predictions in the repetition is evaluated together
        (often preferable).
    summarize_splits : bool
        Whether to create a summary for each split.
        This can be useful when a split is a dataset or
        in other ways a meaningful split that is always
        the same across the prediction lists.
        When enabled, all splits in `splits_list`
        must be the same.
    summarize_splits_allow_differing_members: bool
        Whether splits can differ in their memberships.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    if task == "binary_classification" and positive is None:
        raise ValueError(
            "When `task` is `'binary_classification'`, " "`positive` must be specified."
        )

    num_reps = len(predictions_list)
    messenger(
        f"Evaluating predictions from {num_reps} "
        f"repetition{'s' if num_reps > 1 else ''}"
    )

    # Ensure we work with `targets_list` for code simplicity
    if sum([targets is not None, targets_list is not None]) != 1:
        raise ValueError(
            "Exactly one of {`targets`, `targets_list`} should be specified."
        )
    if targets is not None:
        targets_list = [targets for _ in range(len(predictions_list))]
        targets = None

    # Ensure we work with `groups_list` for code simplicity
    if sum([groups is not None, groups_list is not None]) > 1:
        raise ValueError("Maximally one of {`groups`, `groups_list`} can be specified.")
    if groups is not None:
        groups_list = [groups for _ in range(len(predictions_list))]
        groups = None

    num_splits = None
    if splits_list is not None:
        num_splits = [len(np.unique(split)) for split in splits_list]
        if len(set(num_splits)) == 0:
            raise ValueError("`splits_list` was empty.")
        messenger(
            "Evaluation is performed by splits and averaged per repetition", indent=2
        )
        split_str = (
            f"All repetitions had {num_splits[0]} splits"
            if len(set(num_splits)) == 1
            else (
                "Repetitions had the following number of splits: "
                f"{', '.join([str(n) for n in num_splits])} splits"
            )
        )
        messenger(split_str, indent=2)
        if summarize_splits and not all(
            [all(arr == splits_list[0]) for arr in splits_list]
        ):
            if not summarize_splits_allow_differing_members:
                raise ValueError(
                    "`summarize_splits` was enabled but `splits_list` "
                    "contained different splits."
                )
            if not all([set(arr) == set(splits_list[0]) for arr in splits_list]):
                raise ValueError(
                    "`summarize_splits` and `summarize_splits_allow_differing_members` were "
                    "enabled but `splits_list` contained different split IDs in different splits."
                )
    elif summarize_splits:
        raise ValueError(
            "`summarize_splits` was enabled but no " "`splits_list` was specified."
        )

    else:
        messenger("Evaluation is performed on all predictions per repetition")

    if num_splits is not None:
        rep_summaries, split_evaluations = list(
            zip(
                *[
                    evaluate_splits(
                        predictions=predictions_list[rep_idx],
                        targets=targets_list[rep_idx],
                        groups=(
                            groups_list[rep_idx] if groups_list is not None else None
                        ),
                        splits=splits_list[rep_idx],
                        task=task,
                        positive=positive,
                        thresholds=thresholds,
                        target_labels=target_labels,
                        identifier_cols_dict=identifier_cols_dict,
                        split_id_colname=split_id_colname,
                        messenger=None,
                    )
                    for rep_idx in range(num_reps)
                ]
            )
        )

        if len(set(num_splits)) == 1:
            num_splits_str = f"{num_splits[0]}"
        else:
            num_splits_str = "(" + ", ".join([str(n) for n in num_splits[:4]])
            if len(num_splits) > 4:
                num_splits_str += ", ...)"

        threshold_version_str = ""
        if "Threshold Version" in rep_summaries[0]:
            num_thresholds = len(rep_summaries[0]["Threshold Version"])
            threshold_version_str = f"and {num_thresholds} threshold version{'s' if num_thresholds > 1 else ''} "

        out = {
            "What": (
                f"Evaluations and summaries from {num_reps} "
                f"repetitions with {num_splits_str} splits {threshold_version_str}"
                f"from {Evaluator.PRETTY_TASK_NAMES[task].lower()}."
            ),
            "Evaluations": {},
        }

        if "Threshold Versions" in split_evaluations[0]:
            out["Threshold Versions"] = split_evaluations[0]["Threshold Versions"]

        # Summarize summaries
        out["Summary"] = Evaluator.summarize_summaries(
            summaries=rep_summaries, task=task
        )

        # Combine split evaluations
        combined_split_evals = Evaluator.combine_combined_evaluations(
            evaluations=split_evaluations,
            eval_names=eval_names,
            eval_idx_colname=eval_idx_colname,
            identifier_cols_dict=identifier_cols_dict,
        )

        # Transfer select evaluation parts to output
        for eval_part_name in [
            "Scores",
            "Confusion Matrices",
            "Confusion Matrix",
            "ROC",
            "One-vs-All",
        ]:
            if eval_part_name in combined_split_evals:
                out["Evaluations"][eval_part_name] = combined_split_evals[
                    eval_part_name
                ]

        # TODO What happens when num reps are 1?
        if summarize_splits:
            out["Summary"]["Splits"] = summarize_by_splits(
                split_evaluations=split_evaluations,
                combined_split_evals=combined_split_evals,
                eval_idx_colname=eval_idx_colname,
                split_id_colname=split_id_colname,
                identifier_cols_dict=identifier_cols_dict,
                threshold_versions=out.get("Threshold Versions", None),
            )

        return out

    else:
        rep_summary, rep_evaluations = evaluate_existing_splits(
            predictions_list=predictions_list,
            targets_list=targets_list,
            groups_list=groups_list,
            split_names=eval_names,
            task=task,
            positive=positive,
            thresholds=thresholds,
            target_labels=target_labels,
            identifier_cols_dict=identifier_cols_dict,
            split_id_colname=eval_idx_colname,
            messenger=None,
        )

        threshold_version_str = ""
        if "Threshold Versions" in rep_evaluations:
            num_thresholds = len(rep_evaluations["Threshold Versions"])
            threshold_version_str = f"and {num_thresholds} threshold version{'s' if num_thresholds > 1 else ''} "

        out = {
            "What": (
                f"Evaluations and summaries from {num_reps} "
                f"repetitions {threshold_version_str}"
                f"from {Evaluator.PRETTY_TASK_NAMES[task].lower()}."
            ),
            "Summary": rep_summary,
            "Evaluations": rep_evaluations,
        }

        if "Threshold Versions" in rep_evaluations:
            out["Threshold Versions"] = rep_evaluations["Threshold Versions"]
            if rep_summary is not None and "Threshold Versions" in rep_summary:
                del rep_summary["Threshold Versions"]
            if "Threshold Versions" in rep_evaluations:
                del rep_evaluations["Threshold Versions"]

        return out


# Split evaluation summary
def summarize_by_splits(
    split_evaluations,
    combined_split_evals,
    eval_idx_colname,
    split_id_colname,
    identifier_cols_dict=None,
    threshold_versions=None,
):
    out = {}

    res = combined_split_evals["Scores"]

    group_cols = [split_id_colname]
    if threshold_versions is not None:
        group_cols.append("Threshold Version")

    cols_to_drop = ["Num Classes", eval_idx_colname, "Positive Class"]
    if identifier_cols_dict is not None:
        cols_to_drop += list(identifier_cols_dict.keys())
    cols_to_drop = [c for c in cols_to_drop if c in res.columns]

    out["Scores"] = summarize_data_frame(
        df=res,
        by=group_cols,
        threshold_versions=threshold_versions,
        drop_cols=cols_to_drop,
    )

    conf_matrix, conf_name = Evaluator._sum_confusion_matrices(
        evals_or_summaries=split_evaluations,
        note=f"Sum of confusion matrices from {len(split_evaluations)} split evaluations.",
    )
    if conf_name is not None:
        out[conf_name] = conf_matrix

    return out
