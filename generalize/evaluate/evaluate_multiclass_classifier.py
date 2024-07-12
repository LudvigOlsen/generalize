import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from generalize.evaluate.confusion_matrix import MulticlassConfusionMatrix
from generalize.evaluate.evaluate_binary_classifier import BinaryEvaluator
from generalize.evaluate.prepare_inputs import (
    aggregate_classification_predictions_by_group,
)


class MulticlassEvaluator:
    _OVA_METRICS = [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
    ]
    MACRO_METRICS = _OVA_METRICS + ["Macro AUC"]
    METRICS = MACRO_METRICS + ["Overall Accuracy"]

    @staticmethod
    def evaluate(targets, predicted_probabilities, groups=None, labels=None):
        mc_evaluator = MulticlassEvaluator()
        return mc_evaluator(
            targets=targets,
            predicted_probabilities=predicted_probabilities,
            groups=groups,
            labels=labels,
        )

    def __call__(self, targets, predicted_probabilities, groups=None, labels=None):
        predicted_probabilities = np.asarray(predicted_probabilities)
        num_unique_targets = len(np.unique(targets))
        if labels is not None:
            num_classes = len(labels)
            assert (
                num_classes >= num_unique_targets
            ), "`labels` contained fewer elements than the number of unique values in `targets`."
        else:
            num_classes = num_unique_targets

        # If given as class predictions instead of probabilities
        got_class_predictions = (
            len([s for s in predicted_probabilities.shape if s > 1]) == 1
        )

        if got_class_predictions:
            if not issubclass(predicted_probabilities.dtype.type, np.integer):
                raise ValueError(
                    "When multiclass predictions contains discrete class predictions, "
                    f"they must be of type integer. Found {predicted_probabilities.dtype}."
                )
            # Remove potential singleton dimensions
            predictions = predicted_probabilities.squeeze()

        # Aggregate by groups (when present)
        (
            targets,
            predicted_probabilities,
            predictions,
        ) = aggregate_classification_predictions_by_group(
            targets=targets,
            probabilities=(
                predicted_probabilities if not got_class_predictions else None
            ),
            predictions=predictions if got_class_predictions else None,
            groups=groups,
        )

        # Find class prediction
        if not got_class_predictions:
            predictions = np.argmax(predicted_probabilities, axis=-1)

        # Perform the one-vs-all binary evaluations
        one_vs_all_evals = MulticlassEvaluator._evaluate_one_vs_all(
            targets=targets,
            predictions=predictions,
            num_classes=num_classes,
            labels=labels,
        )

        # Macro evaluation (averaging)
        macro_eval = one_vs_all_evals[MulticlassEvaluator._OVA_METRICS].mean(axis=0)
        macro_eval = pd.DataFrame(macro_eval).T

        # Overall evaluation
        overall_accuracy = np.mean(targets == predictions)

        # ROC AUC
        macro_auc = np.nan
        if not got_class_predictions:
            macro_auc = MulticlassEvaluator._evaluate_roc_curve(
                targets=targets, predicted_probabilities=predicted_probabilities
            )

        # Confusion Matrix
        conf_mat = MulticlassConfusionMatrix().fit(targets, predictions)

        # Combine to multiclass evaluation
        mc_eval = macro_eval
        mc_eval["Overall Accuracy"] = overall_accuracy
        mc_eval["Macro AUC"] = macro_auc
        mc_eval["Num Classes"] = num_classes

        return mc_eval, one_vs_all_evals, conf_mat

    @staticmethod
    def _evaluate_one_vs_all(targets, predictions, num_classes, labels=None):
        evaluator = BinaryEvaluator()
        one_vs_all_evals = []
        for cl in range(num_classes):
            ova_targets = [int(t == cl) for t in targets]
            ova_predictions = [int(p == cl) for p in predictions]
            class_eval, _, _ = evaluator(
                targets=ova_targets, predictions=ova_predictions, positive=1
            )
            class_eval["Class"] = cl
            if labels is not None:
                class_eval["Class Label"] = labels[cl]
            one_vs_all_evals.append(class_eval)
        one_vs_all_evals = pd.concat(one_vs_all_evals, ignore_index=True)
        assert all(one_vs_all_evals["Positive Class"] == 1), (
            "Internal error: In One-vs-All, the binomial evaluations "
            "did not use the right `positive` class."
        )
        one_vs_all_evals.drop(columns=["Positive Class", "Num Classes"], inplace=True)
        return one_vs_all_evals

    @staticmethod
    def _evaluate_roc_curve(targets, predicted_probabilities):
        auc = roc_auc_score(
            y_true=targets,
            y_score=predicted_probabilities,
            multi_class="ovr",
            average="macro",
        )
        return auc
