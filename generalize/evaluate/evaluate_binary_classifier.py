from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd
from utipy import move_column_inplace

from generalize.evaluate.confusion_matrix import BinaryConfusionMatrix
from generalize.evaluate.prepare_inputs import (
    BinaryPreparer,
    aggregate_classification_predictions_by_group,
)
from generalize.evaluate.roc_curves import ROCCurve
from generalize.utils.math import scalar_safe_div


class BinaryEvaluator:
    METRICS = [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "TP",
        "FP",
        "TN",
        "FN",
        "AUC",
    ]

    @staticmethod
    def evaluate(
        targets: Union[list, np.ndarray],
        predicted_probabilities: Union[list, np.ndarray],
        groups: Optional[Union[list, np.ndarray]] = None,
        positive: Optional[Union[int, str]] = None,
        thresh: float = 0.5,
        labels: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, BinaryConfusionMatrix, ROCCurve]:
        """
        Evaluate predicted probabilities against binary targets.

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        groups : list or `numpy.ndarray`
            Group identifiers. Same size as `targets` and `predictions`.
            Probabilities are averaged per group before evaluation.
            Class predictions are selected by majority vote before evaluation.
                In case of tie, the smallest value (or first alphabetically)
                is selected. I.e., `0` if `0` and `1` are tied.
        thresh : float
            Cutoff for determining if a prediction is 0 (<`thresh`) or 1 (>=`thresh`).
        positive : int or str
            The positive label. Must be set.
        thresh : float
            Cutoff for converting predicted probabilities to class predictions.
        labels : dict or None
            Mapping from target value to its pretty label.

        Returns
        -------
        `pandas.DataFrame`
            Evaluation.
        BinaryConfusionMatrix
            The confusion matrix.
        ROCCurve
            The ROC curve.
        """
        bin_evaluator = BinaryEvaluator()
        return bin_evaluator(
            targets=targets,
            predicted_probabilities=predicted_probabilities,
            groups=groups,
            positive=positive,
            thresh=thresh,
            labels=labels,
        )

    def __call__(
        self,
        targets: Union[list, np.ndarray],
        predicted_probabilities: Optional[Union[list, np.ndarray]] = None,
        predictions: Optional[Union[list, np.ndarray]] = None,
        groups: Optional[Union[list, np.ndarray]] = None,
        positive: Optional[Union[int, str]] = None,
        thresh: float = 0.5,
        labels: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, BinaryConfusionMatrix, ROCCurve]:
        """
        Run binary evaluation of a set of targets and predictions.

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        predictions : list or `numpy.ndarray`
            The predicted classes.
            Note: Only one of `predicted_probabilities` and `predictions`
            should be specified.
        groups : list or `numpy.ndarray`
            Group identifiers. Same size as `targets` and `predictions`.
            Probabilities are averaged per group before evaluation.
            Class predictions are selected by majority vote before evaluation.
                In case of tie, the smallest value (or first alphabetically)
                is selected. I.e., `0` if `0` and `1` are tied.
        thresh : float
            Cutoff for determining if a prediction is 0 (<`thresh`) or 1 (>=`thresh`).
        positive : int or str
            The positive label. Must be set.
        thresh : float
            Cutoff for converting predicted probabilities to class predictions.
        labels : dict or None
            Mapping from target value to its pretty label.

        Returns
        -------
        `pandas.DataFrame`
            Evaluation.
        BinaryConfusionMatrix
            The confusion matrix.
        ROCCurve
            The ROC curve.
        """
        if int(predicted_probabilities is None) + int(predictions is None) != 1:
            raise ValueError(
                "Exactly one of {`predicted_probabilities`, `predictions`} should be specified."
            )

        # Reduce to 1D arrays
        targets = BinaryPreparer.prepare_targets(targets=targets)
        predictions = BinaryPreparer.prepare_predictions(predictions=predictions)
        predicted_probabilities = BinaryPreparer.prepare_probabilities(
            probabilities=predicted_probabilities
        )

        # Aggregate by groups (when present)
        (
            targets,
            predicted_probabilities,
            predictions,
        ) = aggregate_classification_predictions_by_group(
            targets=targets,
            probabilities=predicted_probabilities,
            predictions=predictions,
            groups=groups,
        )

        BinaryEvaluator._check_positive(positive)
        assert labels is None or (isinstance(labels, dict) and len(labels) == 2)

        # Convert probabilities to class predictions
        if predicted_probabilities is not None:
            predictions = np.asarray(predicted_probabilities >= thresh, dtype=np.int32)

        # Initialize store for evaluation data frames
        eval_dfs = []

        # ROC curve will only be created when probabilities are provided
        roc = None

        # Create confusion matrix and calculate related metrics
        conf_mat = BinaryConfusionMatrix().fit(
            targets=targets, predictions=predictions, positive=positive, labels=labels
        )
        eval_dfs.append(BinaryEvaluator._evaluate_confusion_matrix(conf_mat=conf_mat))

        # Calculate AUC and create roc curve
        # If we have the probabilities
        if predicted_probabilities is not None:
            roc = ROCCurve.from_data(
                targets=targets,
                predicted_probabilities=predicted_probabilities,
                positive=positive,
            )

            eval_dfs.append(pd.DataFrame({"AUC": [roc.auc]}))

            # Add the threshold used to cut probabilities
            eval_dfs.append(pd.DataFrame({"Threshold": [thresh]}))

        # Combine evaluation data frames
        eval_df = pd.concat(eval_dfs, axis=1)
        move_column_inplace(eval_df, "Positive Class", len(eval_df.columns) - 1)
        move_column_inplace(eval_df, "Num Classes", len(eval_df.columns) - 1)

        return eval_df, conf_mat, roc

    @staticmethod
    def _check_positive(positive: Union[int, str]) -> None:
        """
        Check that the `positive` argument is properly set.

        Parameters
        ----------
        positive : int or str
            The positive label. Must be set.
        """
        assert positive is not None, "The `positive` class must be specified."
        assert isinstance(positive, (int, str))

    @staticmethod
    def _evaluate_confusion_matrix(conf_mat: BinaryConfusionMatrix) -> pd.DataFrame:
        """
        Calculate metrics from a confusion matrix.

        Parameters
        ----------
        conf_mat : BinaryConfusionMatrix
            Confusion matrix to calculate metrics from.

        Returns
        -------
        pd.DataFrame
            Data frame with evaluation metrics.
        """
        tn, fp, fn, tp = conf_mat.confusion_matrix.ravel()
        total = sum([tn, fp, fn, tp])
        recall = sensitivity = scalar_safe_div(tp, tp + fn)
        specificity = scalar_safe_div(tn, tn + fp)
        precision = scalar_safe_div(tp, tp + fp)
        npv = scalar_safe_div(tn, tn + fn)
        f1 = 2 * scalar_safe_div(precision * recall, precision + recall)
        accuracy = scalar_safe_div(tn + tp, total)
        balanced_accuracy = (sensitivity + specificity) / 2
        return pd.DataFrame(
            {
                "Accuracy": [accuracy],
                "Balanced Accuracy": [balanced_accuracy],
                "F1": [f1],
                "Sensitivity": [sensitivity],
                "Specificity": [specificity],
                "PPV": [precision],
                "NPV": [npv],
                "TP": [tp],
                "FP": [fp],
                "TN": [tn],
                "FN": [fn],
                "Positive Class": [conf_mat.positive],
                "Num Classes": 2,
            }
        )

    @staticmethod
    def get_threshold_at_specificity(
        targets, predicted_probabilities, above_specificity=0.95, positive=None
    ):
        """
        Find first threshold and sensitivity where specificity is `> above_specificity`.

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        above_specificity : float
            Specificity above which to find the
            first threshold from a ROC curve.
        positive : int or str
            The positive label. Must be set.

        Returns
        -------
        dict
            Dictionary with threshold, specificity, and sensitivity.
        """
        # Create ROC curve
        roc = ROCCurve.from_data(
            predicted_probabilities=predicted_probabilities,
            targets=targets,
            positive=positive,
        )

        return roc.get_threshold_at_specificity(above_specificity=above_specificity)

    @staticmethod
    def get_threshold_at_max_j(targets, predicted_probabilities, positive=None):
        """
        Find first threshold where Youden's J statistic is at its max.

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
        positive : int or str
            The positive label. Must be set.

        Returns
        -------
        dict
            Dictionary with threshold, specificity, and sensitivity.
        """
        # Create ROC curve
        roc = ROCCurve.from_data(
            predicted_probabilities=predicted_probabilities,
            targets=targets,
            positive=positive,
        )

        return roc.get_threshold_at_max_j()
