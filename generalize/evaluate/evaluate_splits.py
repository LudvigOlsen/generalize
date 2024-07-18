from typing import Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from utipy import Messenger, check_messenger

from generalize.evaluate.confusion_matrices import ConfusionMatrices
from generalize.evaluate.evaluate import Evaluator
from generalize.evaluate.prepare_inputs import (
    aggregate_classification_predictions_by_group,
)


def evaluate_existing_splits(
    predictions_list,
    task,
    positive,
    targets_list=None,
    targets=None,
    groups=None,
    groups_list=None,
    split_names=None,
    thresholds=None,
    target_labels=None,
    identifier_cols_dict=None,
    split_id_colname="Fold",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    multiset_evaluator = MultisetEvaluator(task=task, messenger=messenger)
    return multiset_evaluator.evaluate_sets(
        predictions_list=predictions_list,
        positive=positive,
        target_labels=target_labels,
        identifier_cols_dict=identifier_cols_dict,
        eval_idx_colname=split_id_colname,
        split_names=split_names,
        targets=targets,
        targets_list=targets_list,
        groups=groups,
        groups_list=groups_list,
        thresholds=thresholds,
    )


def evaluate_splits(
    predictions,
    targets,
    splits,
    task,
    positive,
    groups=None,
    thresholds=None,
    target_labels=None,
    identifier_cols_dict=None,
    split_id_colname="Fold",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[dict, dict]:
    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    num_splits = len(set(splits))
    messenger(
        f"Evaluating {num_splits} set{'s' if num_splits > 1 else ''} " f"of predictions"
    )
    predictions_list = _split_array_by_groups(predictions, splits)
    targets_list = _split_array_by_groups(targets, splits)
    groups_list = None
    if groups is not None:
        groups_list = _split_array_by_groups(np.asarray(groups), splits)
    split_names = list(np.unique(splits))

    return evaluate_existing_splits(
        predictions_list=predictions_list,
        task=task,
        positive=positive,
        targets_list=targets_list,
        targets=None,
        groups_list=groups_list,
        split_names=split_names,
        thresholds=thresholds,
        target_labels=target_labels,
        identifier_cols_dict=identifier_cols_dict,
        split_id_colname=split_id_colname,
        messenger=messenger,
    )


def _split_array_by_groups(
    x: np.ndarray, groups: Union[list, np.ndarray]
) -> List[np.ndarray]:
    """
    Split array by a group array.
    """
    assert len(x) == len(
        groups
    ), f"`x` ({len(x)}) and `groups` ({len(groups)}) must have the same length."
    return [x[groups == g] for g in np.unique(groups)]


class MultisetEvaluator:
    def __init__(
        self,
        task: str,
        messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
    ) -> None:
        self.task = task
        self.messenger = check_messenger(messenger)

        if self.task not in [
            "binary_classification",
            "multiclass_classification",
            "regression",
        ]:
            raise ValueError(f"`task` was not recognized: {self.task}")

    def evaluate_sets(
        self,
        predictions_list: List[np.ndarray],
        positive: Optional[int],
        target_labels: Optional[dict],
        identifier_cols_dict: dict,
        eval_idx_colname: str,
        split_names: Optional[List[str]] = None,
        targets: Optional[np.ndarray] = None,
        targets_list: Optional[List[np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
        groups_list: Optional[List[np.ndarray]] = None,
        thresholds: Union[float, List[float], Dict[str, float]] = None,
    ) -> Tuple[dict, dict]:
        if self.task == "binary_classification":
            return self._evaluate_binary_classification(
                predictions_list=predictions_list,
                positive=positive,
                target_labels=target_labels,
                identifier_cols_dict=identifier_cols_dict,
                eval_idx_colname=eval_idx_colname,
                split_names=split_names,
                targets=targets,
                targets_list=targets_list,
                groups=groups,
                groups_list=groups_list,
                thresholds=thresholds,
            )
        else:
            return self._evaluate_rest(
                predictions_list=predictions_list,
                target_labels=target_labels,
                identifier_cols_dict=identifier_cols_dict,
                eval_idx_colname=eval_idx_colname,
                split_names=split_names,
                targets=targets,
                targets_list=targets_list,
                groups=groups,
                groups_list=groups_list,
            )

    @staticmethod
    def _replace_split_names(
        df: pd.DataFrame, split_names: Optional[List[str]], colname: str
    ) -> pd.DataFrame:
        # Replace evaluation indices by the split names
        if split_names is not None:
            split_names_dict = dict(enumerate(split_names))
            df[colname] = [split_names_dict[i] for i in df[colname]]
        return df

    @staticmethod
    def _check_shared_inputs(
        predictions_list: List[np.ndarray],
        target_labels: Optional[dict],
        targets: Optional[np.ndarray] = None,
        targets_list: Optional[List[np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
        groups_list: Optional[List[np.ndarray]] = None,
        split_names: Optional[List[str]] = None,
    ) -> Tuple[int, list, list]:
        num_prediction_sets = len(predictions_list)
        if num_prediction_sets == 0:
            raise ValueError("`predictions_list` was empty.")
        if sum([targets is not None, targets_list is not None]) != 1:
            raise ValueError(
                "Exactly one of {`targets`, `targets_list`} should be specified."
            )
        if targets is not None:
            # Repeating `targets` simplifies downstream code
            targets_list = [targets for _ in range(num_prediction_sets)]
        if target_labels is not None and not isinstance(target_labels, dict):
            raise TypeError(
                "When specified, `target_labels` must be a dict "
                "mapping names to the target values."
            )

        if groups is not None:
            # Repeating `groups` simplifies downstream code
            groups_list = [groups for _ in range(num_prediction_sets)]
        elif groups_list is None:
            groups_list = [None for _ in range(num_prediction_sets)]

        if split_names is not None and len(split_names) != num_prediction_sets:
            raise ValueError(
                f"`split_names` ({len(split_names)}) had a different length "
                f"than `predictions_list` ({num_prediction_sets})"
            )

        return num_prediction_sets, targets_list, groups_list

    def _evaluate_binary_classification(
        self,
        predictions_list: List[np.ndarray],
        positive: Optional[int],
        target_labels: Optional[dict],
        identifier_cols_dict: dict,
        eval_idx_colname: str,
        split_names: Optional[List[str]] = None,
        targets: Optional[np.ndarray] = None,
        targets_list: Optional[List[np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
        groups_list: Optional[List[np.ndarray]] = None,
        thresholds: Union[float, List[float], Dict[str, float]] = None,
    ) -> Tuple[dict, dict]:
        (
            num_prediction_sets,
            targets_list,
            groups_list,
        ) = MultisetEvaluator._check_shared_inputs(
            predictions_list=predictions_list,
            target_labels=target_labels,
            targets=targets,
            targets_list=targets_list,
            groups=groups,
            groups_list=groups_list,
            split_names=split_names,
        )
        thresholds, threshold_names = MultisetEvaluator._get_thresholds(
            thresholds=thresholds
        )

        # Evaluate each prediction set with each threshold
        threshold_evaluations = list(
            zip(
                *[
                    MultisetEvaluator._run_evaluate_binary_classification(
                        predictions=preds,
                        targets=targs,
                        groups=grps,
                        positive=positive,
                        target_labels=target_labels,
                        thresholds=thresholds,
                    )
                    for preds, targs, grps in zip(
                        predictions_list, targets_list, groups_list
                    )
                ]
            )
        )

        def plural_s(n):
            return "s" if n > 1 else ""

        combined_evaluations_out = {
            "What": (
                f"Combined evaluations from {num_prediction_sets} "
                f"prediction set{plural_s(num_prediction_sets)} and "
                f"{len(threshold_names)} threshold{plural_s(len(threshold_names))} "
                "from binary classification."
            ),
            "Threshold Versions": threshold_names,
        }

        # Combine all evaluations
        combined_results = []
        all_confusion_matrices = {}
        for threshold_idx, set_evaluations in enumerate(threshold_evaluations):
            combined_evaluations_for_threshold = Evaluator.combine_evaluations(
                evaluations=set_evaluations,
                eval_names=split_names,
                eval_idx_colname=eval_idx_colname,
                identifier_cols_dict=identifier_cols_dict,
            )

            # Add evaluation version identifier
            combined_evaluations_for_threshold["Scores"]["Threshold Version"] = (
                threshold_names[threshold_idx]
            )

            # Append to lists
            combined_results.append(combined_evaluations_for_threshold["Scores"])
            all_confusion_matrices[threshold_names[threshold_idx].replace(".", "_")] = (
                combined_evaluations_for_threshold["Confusion Matrices"]
            )

            # ROC objects shouldn't be affected by the threshold
            # so we just get the first one
            if threshold_idx == 0:
                combined_evaluations_out["ROC"] = combined_evaluations_for_threshold[
                    "ROC"
                ]

        # Concatenate results from all thresholds
        combined_evaluations_out["Scores"] = pd.concat(
            combined_results, ignore_index=True
        )

        # Merge the confusion matrix collections
        combined_evaluations_out["Confusion Matrices"] = ConfusionMatrices.merge(
            all_confusion_matrices, path_prefix="Threshold Version"
        )

        # Summarize evaluations (when multiple)
        evaluation_summary = None
        if num_prediction_sets > 1:
            # Summarize evaluations
            self.messenger("Summarizing evaluations")
            evaluation_summaries = []

            evaluation_summary = {
                "What": f"Evaluation summar{'ies' if len(threshold_evaluations) > 1 else 'y'} from binary classification.",
                "Threshold Versions": threshold_names,
            }

            for threshold_idx, set_evaluations in enumerate(threshold_evaluations):
                summary = Evaluator.summarize_evaluations(
                    evaluations=set_evaluations, task="binary_classification"
                )
                summary["Scores"]["Threshold Version"] = threshold_names[threshold_idx]
                evaluation_summaries.append(summary)

                # Add confusion matrix to collection
                conf_mat = summary["Confusion Matrix"]
                if threshold_idx == 0:
                    evaluation_summary["Confusion Matrices"] = ConfusionMatrices(
                        class_roles=conf_mat.class_roles,
                        count_names=conf_mat.count_names,
                    )
                evaluation_summary["Confusion Matrices"].add(
                    path=f"Threshold Version.{threshold_names[threshold_idx].replace('.', '_')}",
                    matrix=conf_mat.get_counts(),
                )

            # Combine result summaries
            result_summaries = pd.concat(
                [summ["Scores"] for summ in evaluation_summaries], ignore_index=True
            )
            evaluation_summary["Scores"] = result_summaries

        return evaluation_summary, combined_evaluations_out

    def _evaluate_rest(
        self,
        predictions_list: List[np.ndarray],
        target_labels: Optional[dict],
        identifier_cols_dict: dict,
        eval_idx_colname: str,
        split_names: Optional[List[str]] = None,
        targets: Optional[np.ndarray] = None,
        targets_list: Optional[List[np.ndarray]] = None,
        groups: Optional[np.ndarray] = None,
        groups_list: Optional[List[np.ndarray]] = None,
    ) -> Tuple[dict, dict]:
        (
            num_prediction_sets,
            targets_list,
            groups_list,
        ) = MultisetEvaluator._check_shared_inputs(
            predictions_list=predictions_list,
            target_labels=target_labels,
            targets=targets,
            targets_list=targets_list,
            groups=groups,
            groups_list=groups_list,
            split_names=split_names,
        )

        set_evaluations = [
            MultisetEvaluator._run_evaluate_rest(
                predictions=preds,
                targets=targs,
                groups=grps,
                task=self.task,
                target_labels=target_labels,
            )
            for preds, targs, grps in zip(predictions_list, targets_list, groups_list)
        ]

        # Combine results
        combined_evaluations = Evaluator.combine_evaluations(
            evaluations=set_evaluations,
            eval_idx_colname=eval_idx_colname,
            identifier_cols_dict=identifier_cols_dict,
        )

        # Replace evaluation indices by the split names
        combined_evaluations["Scores"] = MultisetEvaluator._replace_split_names(
            df=combined_evaluations["Scores"],
            split_names=split_names,
            colname=eval_idx_colname,
        )
        if "One-vs-All" in combined_evaluations:
            combined_evaluations["One-vs-All"] = MultisetEvaluator._replace_split_names(
                df=combined_evaluations["One-vs-All"],
                split_names=split_names,
                colname=eval_idx_colname,
            )

        # Summarize evaluations
        evaluation_summary = None
        if num_prediction_sets > 1:
            self.messenger("Summarizing evaluations")
            evaluation_summary = Evaluator.summarize_evaluations(
                evaluations=set_evaluations, task=self.task
            )

        return evaluation_summary, combined_evaluations

    @staticmethod
    def _run_evaluate_binary_classification(
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray],
        positive: int,
        target_labels: Optional[dict],
        thresholds: Optional[List[float]],
    ):
        if thresholds is None:
            # Get default threshold values
            thresholds = MultisetEvaluator._calculate_default_thresholds(
                predictions=predictions,
                targets=targets,
                groups=groups,
                positive=positive,
            )

        return (
            Evaluator.evaluate(
                targets=targets,
                predictions=predictions,
                groups=groups,
                task="binary_classification",
                thresh=thresh,
                positive=positive,
                labels=target_labels,
            )
            for thresh in thresholds
        )

    @staticmethod
    def _run_evaluate_rest(
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray],
        task: str,
        target_labels: Optional[dict],
    ) -> dict:
        return Evaluator.evaluate(
            targets=targets,
            predictions=predictions,
            groups=groups,
            task=task,
            labels=target_labels,
        )

    @staticmethod
    def _get_thresholds(
        thresholds: Optional[Union[Dict[str, float], float, List[float]]],
    ) -> Tuple[Optional[List[float]], List[str]]:
        """
        Only relevant for binary classification.

        Returns
        -------
        List of floats or `None`
            The threshold values.
        List of str
            The names of the thresholds.
        """
        # Names of evaluations
        if thresholds is not None:
            if isinstance(thresholds, dict):
                threshold_names = list(thresholds.keys())
                thresholds = list(thresholds.values())
            elif isinstance(thresholds, float):
                threshold_names = [f"{thresholds} Threshold"]
                thresholds = [thresholds]
            elif isinstance(thresholds, list):
                threshold_names = [f"{thresh} Threshold" for thresh in thresholds]
        else:
            # Default thresholds
            threshold_names = [
                "Max. J Threshold",
                "High Specificity Threshold",
                "0.5 Threshold",
            ]

        return thresholds, threshold_names

    @staticmethod
    def _calculate_default_thresholds(
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: Optional[np.ndarray],
        positive: int,
    ) -> List[float]:
        """
        Calculate default threshold values from a *single set* of predictions.
        """
        # Aggregrate predicted probabilities per group (when groups are specified)
        targets, predictions, _ = aggregate_classification_predictions_by_group(
            targets=targets, probabilities=predictions, predictions=None, groups=groups
        )

        # Calculate Max J threshold (Youden's J statistic)
        max_j = Evaluator.get_threshold_at_max_j(
            targets=targets, predicted_probabilities=predictions, positive=positive
        )

        # Find first threshold above a specificity
        high_specificity = Evaluator.get_threshold_at_specificity(
            targets=targets,
            predicted_probabilities=predictions,
            above_specificity=0.95,
            positive=positive,
        )

        # Collect default thresholds
        thresholds = [
            max_j["Threshold"],
            high_specificity["Threshold"],
            0.5,
        ]

        return thresholds
