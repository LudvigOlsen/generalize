import numbers
import pathlib
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from utipy import mk_dir, move_column_inplace
from nattrs import nested_getattr

from generalize.evaluate.confusion_matrices import ConfusionMatrices
from generalize.evaluate.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from generalize.evaluate.evaluate_binary_classifier import BinaryEvaluator
from generalize.evaluate.evaluate_multiclass_classifier import MulticlassEvaluator
from generalize.evaluate.evaluate_regression import RegressionEvaluator
from generalize.evaluate.roc_curves import ROCCurves
from generalize.evaluate.summarize_scores import summarize_cols_by, summarize_data_frame


class Evaluator:
    TASKS = ["binary_classification", "multiclass_classification", "regression"]

    PRETTY_TASK_NAMES = {
        "binary_classification": "Binary Classification",
        "multiclass_classification": "Multiclass Classification",
        "regression": "Regression",
    }

    SUPPORTED_METRICS = {
        "binary_classification": BinaryEvaluator.METRICS,
        "multiclass_classification": MulticlassEvaluator.METRICS,
        "regression": RegressionEvaluator.METRICS,
    }

    @staticmethod
    def evaluate(
        targets: np.ndarray,
        predictions: np.ndarray,
        task: str,
        groups: Optional[np.ndarray] = None,
        positive: Optional[int] = None,
        thresh: float = 0.5,
        labels: Optional[dict] = None,
        ignore_missing: bool = False,
    ) -> dict:
        """
        Evaluate model predictions (classification/regression) on a wide range of metrics.

        Parameters
        ----------
        targets : 1/2D `numpy.ndarray`
            Target values.
        predictions : 1/2D `numpy.ndarray`
            Predicted values.
            In classification, pass the predicted probabilities.
        task : str
            The task to evaluate. One of:
                {'binary_classification', 'multiclass_classification', 'regression'}.
        groups: 1D `numpy.ndarray`
            Group identifiers. Same size as `targets` and `predictions`.
            Classification:
                Probabilities are averaged per group before evaluation.
                Class predictions are selected by majority vote before evaluation.
                    In case of tie, the smallest value (or first alphabetically)
                    is selected. I.e., `0` if `0` and `1` are tied.
            Regression:
                The value is averaged. TODO Not implemented yet
        positive : int or None
            The target value of the positive class *in binary classification*.
        thresh : float
            The probability cutoff *in binary classification*.
        labels: dict or None
            Dict mapping target value to label name.
            E.g. {0: "healthy", 1: "cancer"}
        ignore_missing : bool
            Whether to ignore missing (`numpy.nan`) values.
            NOTE: Currently only used in regression.

        Returns
        -------
        dict
            Shared:
                ``Scores`` : `pandas.DataFrame`
            Classification:
                ``Confusion Matrix`` : `numpy.ndarray` # TODO Correct type
                    Counts of prediction-target combinations.
                Binary:
                    ``ROC``: `ROCCurve` with `numpy.ndarray`s
                        Wrapped output of `scikit-learn`'s `roc_curve()`.
                Multiclass:
                    ``One-vs-All`` : `pandas.DataFrame`
                        Results from the one-vs-all binary evaluations.
        """
        Evaluator._check_task(task)
        if task == "binary_classification" and positive is None:
            raise ValueError(
                "When `task` is `binary_classification`, `positive` must be specified."
            )
        if task != "regression" and ignore_missing:
            raise NotImplementedError(
                "Currently `ignore_missing` is only used in regression tasks."
            )

        # Prepare output dict
        out = {}

        # Add explanation to output
        binary_thresh_what = (
            f" with a probability cutoff at {thresh}"
            if task == "binary_classification"
            else ""
        )
        out["What"] = (
            f"Evaluation of {len(predictions)} predictions from "
            f"{Evaluator.PRETTY_TASK_NAMES[task].lower()}{binary_thresh_what}."
        )

        if task == "binary_classification":
            (
                out["Scores"],
                out["Confusion Matrix"],
                out["ROC"],
            ) = BinaryEvaluator.evaluate(
                targets=targets,
                predicted_probabilities=predictions,
                groups=groups,
                positive=positive,
                thresh=thresh,
                labels=labels,
            )

        elif task == "multiclass_classification":
            (
                out["Scores"],
                out["One-vs-All"],
                out["Confusion Matrix"],
            ) = MulticlassEvaluator.evaluate(
                targets=targets,
                predicted_probabilities=predictions,
                groups=groups,
                labels=labels,
            )

        elif task == "regression":
            out["Scores"] = RegressionEvaluator.evaluate(
                targets=targets,
                predictions=predictions,
                groups=groups,
                ignore_missing=ignore_missing,
            )

        return out

    @staticmethod
    def combine_evaluations(
        evaluations: List[dict],
        eval_names: Optional[List[str]] = None,
        eval_idx_colname: str = "Repetition",
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> dict:
        """
        Combine a list of evaluations, e.g. from multiple repetitions.

        Parameters
        ----------
        evaluations : list
            List of evaluation dicts created with one or more calls to
            the `.evaluate()` method.
        eval_names: list of str or None
            Names of evaluations (repetitions).
            When `None`, a simple index (0 -> N-1) is used.
        eval_idx_colname: str
            Name of evaluation index column in the final, concatenated data frame.
        identifier_cols_dict : dict or None
            Dict mapping column name -> string
            to add to the ``Scores`` and ``One-vs-All`` data frames
            *after* repetitions are concatenated.

        Returns
        -------
        dict
            Shared:
                ``Scores`` : `pandas.DataFrame`
                    The concatenated scores data frames with an additional
                    "evaluation index" column (see `eval_idx_colname`).
            Classification:
                ``Confusion Matrices`` : `ConfusionMatrices`
                    Collection of confusion matrices from all evaluations.
                Binary:
                    ``ROC`` : `ROCCurves`
                        A collection of ROC curves from all evaluations.
                Multiclass:
                    ``One-vs-All Scores`` : `pandas.DataFrame`
                        The concatenated one-vs-all data frames with
                        an additional "evaluation index" column
                        (see `eval_idx_colname`).
        """
        assert isinstance(evaluations, (list, tuple)) and isinstance(
            evaluations[0], dict
        ), (
            "`evaluations` must be a list of dicts, where each dict "
            "is an evaluation made with `evaluate()`."
        )

        out = {"What": "Combined evaluations."}

        # Extract and combine results data frames
        all_results = [evaluation["Scores"] for evaluation in evaluations]
        add_reps_column(all_results, rep_names=eval_names, colname=eval_idx_colname)
        all_results = pd.concat(all_results, axis=0, ignore_index=True)
        add_identifier_columns(all_results, identifier_cols_dict)

        out["Scores"] = all_results

        # Extract and save confusion matrices
        if "Confusion Matrix" in evaluations[0]:
            # Initialize collection of confusion matrices
            out["Confusion Matrices"] = ConfusionMatrices(
                classes=nested_getattr(evaluations[0], "Confusion Matrix.classes"),
                class_roles=nested_getattr(
                    evaluations[0], "Confusion Matrix.class_roles"
                ),
                count_names=nested_getattr(
                    evaluations[0], "Confusion Matrix.count_names"
                ),
            )

            # Add confusion matrices to collection
            for eval_idx, evaluation in enumerate(evaluations):
                # Get Split name if available
                # TODO: This is a hack. Perhaps eval should have a name key when meaningful?
                eval_name = eval_idx
                if (
                    "Split" in evaluation["Scores"].columns
                    and evaluation["Scores"].shape[0] == 1
                ):
                    eval_name = (
                        str(evaluation["Scores"]["Split"].tolist()[0])
                        .replace(" ", "_")
                        .replace(".", "_")
                    )

                out["Confusion Matrices"].add(
                    path=f"{eval_idx_colname}.{eval_name}",
                    matrix=nested_getattr(
                        evaluation, "Confusion Matrix.confusion_matrix"
                    ),
                )

        if "One-vs-All" in evaluations[0]:
            all_ova = [evaluation["One-vs-All"] for evaluation in evaluations]
            add_reps_column(all_ova, rep_names=eval_names, colname=eval_idx_colname)
            all_ova = pd.concat(all_ova, axis=0, ignore_index=True)
            add_identifier_columns(all_ova, identifier_cols_dict)

            out["One-vs-All"] = all_ova

        if "ROC" in evaluations[0]:
            out["ROC"] = ROCCurves()
            for eval_idx, evaluation in enumerate(evaluations):
                # Get Split name if available
                # TODO: This is a hack. Perhaps eval should have a name key when meaningful?
                eval_name = eval_idx
                if (
                    "Split" in evaluation["Scores"].columns
                    and evaluation["Scores"].shape[0] == 1
                ):
                    eval_name = (
                        str(evaluation["Scores"]["Split"].tolist()[0])
                        .replace(" ", "_")
                        .replace(".", "_")
                    )

                out["ROC"].add(
                    path=f"{eval_idx_colname}.{eval_name}", roc_curve=evaluation["ROC"]
                )

        return out

    @staticmethod
    def combine_combined_evaluations(
        evaluations: List[dict],
        eval_names: Optional[List[str]] = None,
        eval_idx_colname: str = "Repetition",
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> dict:
        """
        Combine a list of evaluations, e.g. from multiple repetitions.

        Parameters
        ----------
        evaluations : list
            List of evaluation dicts created with one or more calls to
            the `.evaluate()` method.
        eval_names: list of str or None
            Names of evaluations (repetitions).
            When `None`, a simple index (0 -> N-1) is used.
        eval_idx_colname: str
            Name of evaluation index column in the final, concatenated data frame.
        identifier_cols_dict : dict or None
            Dict mapping column name -> string
            to add to the ``Scores`` and ``One-vs-All`` data frames
            *after* repetitions are concatenated.

        Returns
        -------
        dict
            Shared:
                ``Scores`` : `pandas.DataFrame`
                    The concatenated scores data frames with an additional
                    "evaluation index" column (see `eval_idx_colname`).
            Classification:
                ``Confusion Matrices`` : `ConfusionMatrices`
                    Collection of confusion matrices from all evaluations.
                Binary:
                    ``ROC`` : `ROCCurves`
                        A collection of ROC curves from all evaluations.
                Multiclass:
                    ``One-vs-All`` : `pandas.DataFrame`
                        The concatenated one-vs-all result data frames with
                        an additional "evaluation index" column
                        (see `eval_idx_colname`).
        """
        assert isinstance(evaluations, (list, tuple)) and isinstance(
            evaluations[0], dict
        ), (
            "`evaluations` must be a list of dicts, where each dict "
            "is an evaluation made with `evaluate()`."
        )

        out = {"What": "Combined already combined evaluations."}

        # Extract and combine results data frames
        all_results = [evaluation["Scores"] for evaluation in evaluations]
        add_reps_column(all_results, rep_names=eval_names, colname=eval_idx_colname)
        all_results = pd.concat(all_results, axis=0, ignore_index=True)
        add_identifier_columns(all_results, identifier_cols_dict)
        if "Threshold Version" in all_results.columns:
            move_column_inplace(
                all_results, col="Threshold Version", pos=len(all_results.columns) - 1
            )

        out["Scores"] = all_results

        # Extract and save confusion matrices
        if "Confusion Matrices" in evaluations[0]:
            out["Confusion Matrices"] = ConfusionMatrices.merge(
                collections=dict(
                    enumerate(
                        [evaluation["Confusion Matrices"] for evaluation in evaluations]
                    )
                ),
                path_prefix=eval_idx_colname,
            )

        if "One-vs-All" in evaluations[0]:
            all_ova = [evaluation["One-vs-All"] for evaluation in evaluations]
            add_reps_column(all_ova, rep_names=eval_names, colname=eval_idx_colname)
            all_ova = pd.concat(all_ova, axis=0, ignore_index=True)
            add_identifier_columns(all_ova, identifier_cols_dict)

            out["One-vs-All"] = all_ova

        if "ROC" in evaluations[0]:
            out["ROC"] = ROCCurves.merge(
                collections=dict(
                    enumerate([evaluation["ROC"] for evaluation in evaluations])
                ),
                path_prefix=eval_idx_colname,
            )

        return out

    @staticmethod
    def summarize_evaluations(evaluations: List[dict], task: str) -> dict:
        """
        Summarize a list of evaluations, e.g. from multiple repetitions.

        Get the mean, standard deviation, min., and max. for each
        metric in the scores, as well as the counts of `numpy.nan`s
        (e.g. in case of zero-division).

        Parameters
        ----------
        evaluations : list
            List of evaluation dicts created with one or more calls
            to the `Evaluator.evaluate()` method.
        task : str
            The task to evaluate. One of:
                {'binary_classification', 'multiclass_classification', 'regression'}.

        Returns
        -------
        dict
            Shared:
                ``Scores`` : `pandas.DataFrame`
                    A summary of the scores.
            Classification:
                ``Confusion Matrix`` : ``BinaryConfusionMatrix`` or ``MulticlassConfusionMatrix``
                    Sum of the confusion matrices from all evaluations.
        """

        Evaluator._check_task(task)

        # Prepare output dict
        out = {}

        # Add explanation to output
        out["What"] = (
            f"Summary of {len(evaluations)} evaluations from "
            f"{Evaluator.PRETTY_TASK_NAMES[task].lower()}."
        )

        # Summarize the results
        all_results = pd.concat([e["Scores"] for e in evaluations], ignore_index=True)
        use_metrics = Evaluator.SUPPORTED_METRICS[task]
        if task == "binary_classification" and "Threshold" not in use_metrics:
            use_metrics += ["Threshold"]

        out["Scores"] = summarize_data_frame(
            df=all_results,
            drop_cols=list(set(all_results.columns).difference(use_metrics)),
        )

        # Sum the confusion matrices
        conf_matrix, conf_name = Evaluator._sum_confusion_matrices(
            evaluations,
            note=f"Sum of confusion matrices from {len(evaluations)} evaluations.",
        )
        if conf_name is not None:
            out[conf_name] = conf_matrix

        # TODO Summarize One-vs-All by class as well

        return out

    @staticmethod
    def summarize_summaries(summaries: List[dict], task: str) -> dict:
        """
        Summarize a list of summaries, e.g. from multiple repetitions with splits.

        Get the mean, standard deviation, min., and max. for each
        metric in the scores, as well as the total counts of `numpy.nan`s
        from the summaries.

        Parameters
        ----------
        summaries : list
            List of summary dicts created with one or more calls
            to the `Evaluator.summarize_evaluations()` method.
        task : str
            The task that was evaluated. One of:
                {'binary_classification', 'multiclass_classification', 'regression'}.

        Returns
        -------
        dict
            Shared:
                ``Scores`` : `pandas.DataFrame`
                    A summary of the scores.
            Classification:
                ``Confusion Matrices`` : ``ConfusionMatrices``
                    Collection of total confusion matrices.
                    Sums the respective confusion matrices
                    across summaries to a new collection
                    with the same size.
        """

        Evaluator._check_task(task)

        # Prepare output dict
        out = {}

        # Add explanation to output
        out["What"] = (
            f"Summary of {len(summaries)} *summaries* from "
            f"{Evaluator.PRETTY_TASK_NAMES[task].lower()}."
        )

        threshold_versions_enum = None
        if "Threshold Versions" in summaries[0]:
            out["Threshold Versions"] = summaries[0]["Threshold Versions"]

            # Used for ordering data frame
            threshold_versions_enum = {
                val: idx for idx, val in enumerate(out["Threshold Versions"])
            }

        def extract_scores(summary, measure="Average"):
            res = summary["Scores"]
            return res[res["Measure"] == measure]

        #### Summarize summaries ####

        # Summarize the results
        all_average_results = pd.concat(
            [extract_scores(s, measure="Average") for s in summaries], ignore_index=True
        )
        all_nan_counts = pd.concat(
            [extract_scores(s, measure="# NaNs") for s in summaries], ignore_index=True
        )
        use_metrics = Evaluator.SUPPORTED_METRICS[task]
        if task == "binary_classification":
            if "Threshold" not in use_metrics:
                use_metrics += ["Threshold"]
            if (
                "Threshold Version" not in use_metrics
                and "Threshold Version" in all_average_results
            ):
                use_metrics += ["Threshold Version"]

        all_average_results = all_average_results[use_metrics]

        # Temporarily add "Threshold Version" to group by
        if "Threshold Version" not in all_average_results.columns:
            all_average_results["Threshold Version"] = "__tmp__"
        if "Threshold Version" not in all_nan_counts.columns:
            all_nan_counts["Threshold Version"] = "__tmp__"

        average_results_summary = summarize_cols_by(
            df=all_average_results, by=["Threshold Version"], count_nans=False
        )

        # Get total NaN counts
        nan_counts_summary = (
            all_nan_counts.groupby(["Threshold Version"])
            .sum(numeric_only=True)
            .reset_index()
        )
        nan_counts_summary["Measure"] = "# Total NaNs"

        summary_summary = pd.concat([average_results_summary, nan_counts_summary])

        if threshold_versions_enum is not None:
            summary_summary = summary_summary.sort_values(
                by="Threshold Version",
                key=lambda xs: [threshold_versions_enum[x] for x in xs],
                kind="stable",
            ).reset_index(drop=True)
            move_column_inplace(
                summary_summary,
                "Threshold Version",
                pos=len(summary_summary.columns) - 1,
            )
        else:
            del summary_summary["Threshold Version"]

        out["Scores"] = summary_summary

        #### Sum confusion matrices ####
        conf_matrix, conf_name = Evaluator._sum_confusion_matrices(
            summaries,
            note=f"Sum of confusion matrices from {len(summaries)} summaries.",
        )
        if conf_name is not None:
            out[conf_name] = conf_matrix

        return out

    @staticmethod
    def save_evaluations(
        evaluations: Optional[List[dict]] = None,
        combined_evaluations: Optional[dict] = None,
        warnings: Optional[List[List[List]]] = None,
        out_path: Union[pathlib.Path, str] = None,
        eval_names: Optional[List[str]] = None,
        eval_idx_colname: str = "Repetition",
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> None:
        """
        (Combine and) save a list of evaluations (outputs of `Evaluator.evaluate()`).
        Already combined evaluations can be given instead via `combined_evaluations`.

        See `Evaluator.combine_evaluations()` for how the evaluations
        are combined prior to being saved.

        Parameters
        ----------
        evaluations : list
            List of evaluation dicts created with one or more calls
            to the `Evaluator.evaluate()` method.
        combined_evaluations : dict
            Dict with combined evaluations created with
            the `Evaluator.combine_evaluations()` method.
        warnings : list of lists
            A list with sub lists for repetitions with sub lists for folds with warnings.
        out_path : str or `pathlib.Path`
            Path to the directory to save evaluations in.
        eval_names: list of str or None
            Names of evaluations (repetitions).
            When `None`, a simple index (0 -> N-1) is used.
        eval_idx_colname : str
            Name of evaluation index column in the final, concatenated data frame.
            Ignored when `combined_evaluations` is supplied.
        identifier_cols_dict : dict or None
            Dict mapping column name -> string
            to add to the ``Scores`` and ``One-vs-All`` data frames
            *after* repetitions are concatenated.
            Ignored when `combined_evaluations` is supplied.
        """
        if (evaluations is not None and combined_evaluations is not None) or (
            evaluations is None and combined_evaluations is None
        ):
            raise ValueError(
                "Exactly one of `evaluations` and `combined_evaluations` must be specified."
            )
        if evaluations is not None:
            assert isinstance(evaluations, list) and isinstance(evaluations[0], dict), (
                "`evaluations` must be a list of dicts, where each dict "
                "is an evaluation made with `evaluate()`."
            )
        else:
            assert isinstance(
                combined_evaluations, dict
            ), "`combined_evaluations` must be a dict as created with `Evaluator.combine_evaluations()`."
        assert out_path is not None, "`out_path` must be specified."

        # Create out_path directory if necessary
        mk_dir(path=out_path, arg_name="out_path")
        out_path = pathlib.Path(out_path)

        if combined_evaluations is None:
            combined_evaluations = Evaluator.combine_evaluations(
                evaluations=evaluations,
                eval_names=eval_names,
                eval_idx_colname=eval_idx_colname,
                identifier_cols_dict=identifier_cols_dict,
            )

        # Format and save warnings
        if warnings is not None:
            warning_tuples = [
                (
                    eval_names[rep_idx] if eval_names is not None else rep_idx,
                    fold_idx,
                    str(w.category),
                    str(w.message),
                    str(w.filename),
                    str(w.lineno),
                )
                for rep_idx, rep in enumerate(warnings)
                for fold_idx, ws in enumerate(rep)
                for w in ws
            ]
            warnings_df = pd.DataFrame(
                warning_tuples,
                columns=[
                    eval_idx_colname,
                    "Fold",
                    "Category",
                    "Message",
                    "Filename",
                    "LineNo",
                ],
            )

            # Save warnings
            warnings_df.to_csv(out_path / "warnings.csv", index=False)

            # Add warnings count to evaluation scores
            rep_indices, warning_counts = np.unique(
                warnings_df[eval_idx_colname], return_counts=True
            )
            combined_evaluations["Scores"].loc[:, "Num Warnings"] = 0
            for rep_idx, num_warns in zip(rep_indices, warning_counts):
                # NOTE: We may have multiple threshold versions
                # So we need to assign the counts in every row with a
                # given repetition index
                combined_evaluations["Scores"]["Num Warnings"][
                    combined_evaluations["Scores"][eval_idx_colname] == rep_idx
                ] = num_warns

        # Save 'results' data frames
        combined_evaluations["Scores"].to_csv(
            out_path / "evaluation_scores.csv", index=False
        )

        # Save names of threshold versions of the evaluation
        if "Threshold Versions" in combined_evaluations:
            with open(str(out_path / "threshold_versions.txt"), "w") as f:
                f.write(
                    "The evaluations were made with these threshold versions, in this order:\n"
                )
                for thresh_name in combined_evaluations["Threshold Versions"]:
                    f.write(f"{thresh_name}\n")

        # Save confusion matrices
        if "Confusion Matrices" in combined_evaluations:
            confusion_matrices: ConfusionMatrices = combined_evaluations[
                "Confusion Matrices"
            ]
            confusion_matrices.save(file_path=out_path / "confusion_matrices.json")

        if "One-vs-All" in combined_evaluations:
            combined_evaluations["One-vs-All"].to_csv(
                out_path / "one_vs_all.csv", index=False
            )

        if "ROC" in combined_evaluations:
            roc_collection: ROCCurves = combined_evaluations["ROC"]
            roc_collection.save(file_path=out_path / "ROC_curves.json")

    @staticmethod
    def save_evaluation_summary(
        evaluation_summary: dict,
        out_path: Union[pathlib.Path, str],
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> None:
        """
        Save an evaluation summary created with `Evaluator.summarize_evaluations()`.

        # TODO Add summary of warning counts to `Scores` summary.

        Parameters
        ----------
        evaluation_summary : dict
            An evaluation summary created with `Evaluator.summarize_evaluations()`.
        out_path : str or `pathlib.Path`
            Path to the directory to save the summary in.
        identifier_cols_dict : dict or None
            Dict mapping column name -> string
            to add to the ``Scores`` data frame.
        """
        assert isinstance(evaluation_summary, dict)
        assert identifier_cols_dict is None or isinstance(identifier_cols_dict, dict)

        # Create out_path directory if necessary
        mk_dir(path=out_path, arg_name="out_path")
        out_path = pathlib.Path(out_path)

        # Extract and save scores
        scores = evaluation_summary["Scores"]
        add_identifier_columns(scores, identifier_cols_dict)
        scores.to_csv(out_path / "evaluation_summary.csv", index=False)

        # Extract and save total confusion matri(x/ces)
        if "Confusion Matrices" in evaluation_summary:
            confusion_matrices: ConfusionMatrices = evaluation_summary[
                "Confusion Matrices"
            ]
            confusion_matrices.save(
                file_path=out_path / "total_confusion_matrices.json"
            )
        elif "Confusion Matrix" in evaluation_summary:
            confusion_matrices: ConfusionMatrices = evaluation_summary[
                "Confusion Matrix"
            ].to_collection()
            confusion_matrices.save(
                file_path=out_path / "total_confusion_matrices.json"
            )

        if "Splits" in evaluation_summary:
            if "Scores" in evaluation_summary["Splits"]:
                evaluation_summary["Splits"]["Scores"].to_csv(
                    out_path / "splits_summary.csv"
                )

    @staticmethod
    def save_predictions(
        predictions_list: List[Union[np.ndarray, pd.DataFrame]] = None,
        targets: Optional[Union[np.ndarray, list]] = None,
        targets_list: Optional[List[Union[np.ndarray, list]]] = None,
        groups: Optional[Union[np.ndarray, list]] = None,
        groups_list: Optional[List[Union[np.ndarray, list]]] = None,
        sample_ids: Optional[Union[np.ndarray, list]] = None,
        split_indices_list: Optional[List[np.ndarray]] = None,
        target_idx_to_target_label_map: Optional[Dict[int, str]] = None,
        positive_class: Optional[str] = None,
        out_path: Union[pathlib.Path, str] = None,
        idx_names: Optional[List[str]] = None,
        idx_colname: str = "Repetition",
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> None:
        """
        Combine and save a list of predictions.

        Parameters
        ----------
        predictions_list
            List of prediction arrays / data frames.
        targets
            Target values. Expected to have same length
            and order as each prediction set.
        groups
            Sample groups (like subject IDs).
        split_indices_list
            Lists / `numpy.ndarray`s with split indices
            (i.e. fold index per observation).
            The order of both the lists and list elements are assumed
            to match `predictions_list`.
        target_idx_to_target_label_map
            Mapping of target class indices to their names (labels).
            Is used to rename probability columns
            in multiclass classification.
        positive_class
            Name of the predicted class when only one
            probability class is present.
            Optional. Used to rename probability column.
        out_path
            Path to the directory to save the predictions in.
        idx_names
            Names of evaluations (repetitions).
            When `None`, a simple index (0 -> N-1) is used.
        idx_colname
            Name of prediction set index column
            in the final, concatenated data frame.
        identifier_cols_dict
            Dict mapping column name -> string
            to add to the prediction data frame
            *after* repetitions are concatenated.

        """

        all_predictions = Evaluator.combine_predictions(
            predictions_list=predictions_list,
            targets=targets,
            targets_list=targets_list,
            groups=groups,
            groups_list=groups_list,
            sample_ids=sample_ids,
            split_indices_list=split_indices_list,
            target_idx_to_target_label_map=target_idx_to_target_label_map,
            positive_class=positive_class,
            idx_names=idx_names,
            idx_colname=idx_colname,
            identifier_cols_dict=identifier_cols_dict,
        )

        Evaluator.save_combined_predictions(
            combined_predictions=all_predictions,
            out_path=out_path,
        )

    @staticmethod
    def save_combined_predictions(
        combined_predictions: pd.DataFrame,
        out_path: Union[pathlib.Path, str],
    ) -> None:
        """
        Save already combined predictions,
        e.g. as combined with `.combine_predictions()`

        Parameters
        ----------
        combined_predictions
            Data frame with predictions, targets, etc.
            to save to disk.
        out_path
            Path to the directory to save the predictions in.
        """

        # Create out_path directory if necessary
        mk_dir(path=out_path, arg_name="out_path")
        out_path = pathlib.Path(out_path)

        # Save predictions to disk
        combined_predictions.to_csv(out_path / "predictions.csv", index=False)

    @staticmethod
    def combine_predictions(
        predictions_list: List[Union[np.ndarray, pd.DataFrame]] = None,
        targets: Optional[Union[np.ndarray, list]] = None,
        targets_list: Optional[List[Union[np.ndarray, list]]] = None,
        groups: Optional[Union[np.ndarray, list]] = None,
        groups_list: Optional[List[Union[np.ndarray, list]]] = None,
        sample_ids: Optional[Union[np.ndarray, list]] = None,
        split_indices_list: Optional[List[np.ndarray]] = None,
        target_idx_to_target_label_map: Optional[Dict[int, str]] = None,
        positive_class: Optional[str] = None,
        idx_names: Optional[List[str]] = None,
        idx_colname: str = "Repetition",
        identifier_cols_dict: Optional[Dict[str, Union[str, numbers.Number]]] = None,
    ) -> pd.DataFrame:
        """
        Combine a list of predictions.
        Used to prepare the predictions to be saved as well.

        Parameters
        ----------
        predictions_list
            List of prediction arrays / data frames.
        targets
            Target values. Expected to have same length
            and order as each prediction set.
        groups
            Sample groups (like subject IDs).
        split_indices_list
            Lists / `numpy.ndarray`s with split indices
            (i.e. fold index per observation).
            The order of both the lists and list elements are assumed
            to match `predictions_list`.
        target_idx_to_target_label_map
            Mapping of target class indices to their names (labels).
            Is used to rename probability columns
            in multiclass classification.
        positive_class
            Name of the predicted class when only one
            probability class is present.
            Optional. Used to rename probability column.
        idx_names
            Names of evaluations (repetitions).
            When `None`, a simple index (0 -> N-1) is used.
        idx_colname
            Name of prediction set index column
            in the final, concatenated data frame.
        identifier_cols_dict
            Dict mapping column name -> string
            to add to the prediction data frame
            *after* repetitions are concatenated.

        Returns
        -------
        pandas.DataFrame
            The data frame with all the predictions, targets, etc.
        """

        if sum([targets is not None, targets_list is not None]) > 1:
            raise ValueError(
                "Maximally one of {`targets`, `targets_list`} can be specified."
            )
        if sum([groups is not None, groups_list is not None]) > 1:
            raise ValueError(
                "Maximally one of {`groups`, `groups_list`} can be specified."
            )

        # Convert predictions to data frame
        predictions_list = [pd.DataFrame(preds) for preds in predictions_list]

        # Rename prediction column when it's a single column
        if len(predictions_list[0].columns) == 1:
            for preds_df in predictions_list:
                if positive_class is not None:
                    preds_df.columns = [f"P({positive_class})"]
                else:
                    preds_df.columns = ["Prediction"]
        elif target_idx_to_target_label_map is not None:
            prob_columns = [
                "P(" + target_idx_to_target_label_map[idx_key] + ")"
                for idx_key in sorted(target_idx_to_target_label_map.keys())
            ]
            for preds_df in predictions_list:
                preds_df.columns = prob_columns

        # Add targets to each data frame
        if targets is not None:
            for preds_df in predictions_list:
                preds_df["Target"] = targets
                if target_idx_to_target_label_map is not None:
                    preds_df["Target Label"] = [
                        target_idx_to_target_label_map[t] for t in preds_df["Target"]
                    ]
        if targets_list is not None:
            assert len(targets_list) == len(predictions_list)
            for preds_df, tgts in zip(predictions_list, targets_list):
                preds_df["Target"] = tgts

        if groups is not None:
            for preds_df in predictions_list:
                preds_df["Group"] = groups
        if groups_list is not None:
            assert len(groups_list) == len(predictions_list)
            for preds_df, grps in zip(predictions_list, groups_list):
                preds_df["Group"] = grps

        if sample_ids is not None:
            for preds_df in predictions_list:
                preds_df["Sample ID"] = sample_ids

        if split_indices_list is not None:
            assert len(split_indices_list) == len(predictions_list), (
                "When specified, `split_indices_list` must be a list with "
                "the same length as `predictions_list`."
            )
            for preds, splits in zip(predictions_list, split_indices_list):
                preds["Split"] = splits

        # Add the prediction set index column to each data frame
        add_reps_column(predictions_list, rep_names=idx_names, colname=idx_colname)

        # Concatenate to a single data frame
        all_predictions = pd.concat(predictions_list, axis=0, ignore_index=True)

        # Add the identifier columns
        add_identifier_columns(all_predictions, identifier_cols_dict)

        return all_predictions

    @staticmethod
    def get_threshold_at_specificity(
        targets: Union[np.ndarray, list],
        predicted_probabilities: Union[np.ndarray, list],
        above_specificity: float = 0.95,
        positive: int = None,
        task: str = "binary_classification",
    ) -> dict:
        """
        Find first threshold and sensitivity where specificity is `> above_specificity`.

        NOTE: Binary classification only.

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        above_specificity : float
            Specificity above which to find the first threshold from a ROC curve.
        positive : int
            The target value of the positive class. Must be set.

        Returns
        -------
        dict
            Dictionary with the threshold and its specificity and sensitivity.
        """
        if task != "binary_classification":
            raise NotImplementedError(
                f"Only implemented for binary classification but `task` was: {task}"
            )
        return BinaryEvaluator.get_threshold_at_specificity(
            targets=targets,
            predicted_probabilities=predicted_probabilities,
            above_specificity=above_specificity,
            positive=positive,
        )

    @staticmethod
    def get_threshold_at_max_j(
        targets: Union[np.ndarray, list],
        predicted_probabilities: Union[np.ndarray, list],
        positive: int = None,
        task: str = "binary_classification",
    ) -> dict:
        """
        Find first threshold where Youden's J statistic is at its max.

        NOTE: Binary classification only.

        Youden's J statistic is defined as
            `J = sensitivity + Specificity - 1`
        and can thus be written as:
            `J = TPR - FPR`

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        positive : int
            The target value of the positive class. Must be set.

        Returns
        -------
        dict
            Dictionary with the threshold and its specificity and sensitivity.
        """
        if task != "binary_classification":
            raise NotImplementedError(
                f"Only implemented for binary classification but `task` was: {task}"
            )
        return BinaryEvaluator.get_threshold_at_max_j(
            targets=targets,
            predicted_probabilities=predicted_probabilities,
            positive=positive,
        )

    @staticmethod
    def _sum_confusion_matrices(
        evals_or_summaries, note=None
    ) -> Union[BinaryConfusionMatrix, MulticlassConfusionMatrix, ConfusionMatrices]:
        out = None
        conf_name = None
        if "Confusion Matrix" in evals_or_summaries[0]:
            conf_name = "Confusion Matrix"
        elif "Confusion Matrices" in evals_or_summaries[0]:
            conf_name = "Confusion Matrices"
        if conf_name is not None:
            # Combine confusion matrices
            out = sum([e[conf_name] for e in evals_or_summaries])
            if note is not None:
                out.note = note
        return out, conf_name

    @staticmethod
    def _check_task(task):
        if not isinstance(task, str):
            raise ValueError("`task` must be a string.")
        if task not in Evaluator.TASKS:
            raise ValueError(
                f"`task` was not recognized. Allowed tasks: {Evaluator.TASKS}."
            )


def add_reps_column(dfs, rep_names: Optional[List[str]] = None, colname="Repetition"):
    """
    We add to the existing data frames, so it happens "in-place".
    """
    if rep_names is not None and len(rep_names) != len(dfs):
        raise ValueError("`rep_names` must have same length as `dfs`.")
    for rep, df in enumerate(dfs):
        df[colname] = rep_names[rep] if rep_names is not None else rep


def add_identifier_columns(df, identifier_cols_dict):
    if identifier_cols_dict is not None:
        for col in identifier_cols_dict.keys():
            df[col] = identifier_cols_dict[col]
