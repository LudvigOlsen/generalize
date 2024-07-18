from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd


class BinaryPreparer:
    """
    Methods for preparing inputs to binary evaluation.
    """

    @staticmethod
    def prepare_targets(
        targets: Optional[Union[list, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        """
        Ensure targets have the right format for binary classification evaluation.

        Parameters
        ----------
        targets : list or `numpy.ndarray` or `None`
            The binary target classes.

        Returns
        -------
        np.ndarray (or None)
            Targets in the right format.
        """
        if targets is not None:
            targets = np.asarray(targets, dtype=np.int32)
            assert targets.ndim <= 2
            if targets.ndim == 2:
                if targets.shape[1] > 1:
                    raise ValueError(
                        (
                            "`targets` must be array with 1 scalar "
                            f"per observation but had shape ({targets.shape})."
                        )
                    )

                # Remove singleton dimension
                targets = targets.squeeze()

        return targets

    @staticmethod
    def prepare_probabilities(
        probabilities: Optional[Union[list, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        """
        Ensure probabilities have the right format for binary classification evaluation.

        Parameters
        ----------
        probabilities : list or `numpy.ndarray` or `None`
            The predicted probabilities.

        Returns
        -------
        np.ndarray (or None)
            Predicted probabilities in the right format.
        """
        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float32)

            assert probabilities.ndim <= 2
            if probabilities.ndim == 2:
                if probabilities.shape[1] not in [1, 2]:
                    raise ValueError(
                        (
                            "Second dimension of `probabilities` must have size "
                            f"(1) or (2), but had size ({probabilities.shape[1]})."
                        )
                    )

                if probabilities.shape[1] == 2:
                    # Get probabilities of second class
                    probabilities = probabilities[:, 1]
                else:
                    # Remove singleton dimensions
                    probabilities = probabilities.squeeze()

        return probabilities

    @staticmethod
    def prepare_predictions(
        predictions: Optional[Union[list, np.ndarray]] = None
    ) -> Optional[np.ndarray]:
        """
        Ensure predictions have the right format for binary classification evaluation.

        Parameters
        ----------
        predictions : list or `numpy.ndarray` or `None`
            The predicted classes.

        Returns
        -------
        np.ndarray (or None)
            Predicted classes in the right format.
        """
        if predictions is not None:
            predictions = np.asarray(predictions, dtype=np.int32)

            assert predictions.ndim <= 2
            if predictions.ndim == 2:
                if predictions.shape[1] > 1:
                    raise ValueError(
                        (
                            "`predictions` must be array with 1 scalar "
                            f"per observation but had shape ({predictions.shape})."
                        )
                    )

                # Remove singleton dimension
                predictions = predictions.squeeze()

        return predictions


def _shared_aggregate_predictions_by_group(
    targets: Optional[Union[list, np.ndarray]],
    numeric: Optional[Union[list, np.ndarray]] = None,
    groups: Optional[Union[list, np.ndarray]] = None,
):
    target_df = (
        pd.DataFrame({"Target": targets, "Group": groups})
        .groupby("Group")
        .Target.first()
        .reset_index()
        .sort_values("Group")
    )

    new_targets = target_df["Target"].to_numpy()

    new_numeric = None
    if numeric is not None:
        if isinstance(numeric, np.ndarray) and numeric.ndim > 1:
            if numeric.ndim > 2:
                raise ValueError(
                    "`probabilities / regression output` must be 1- or 2D when passed as `numpy.ndarray`."
                )
            # Multiclass probabilities
            numerics_df = pd.DataFrame(
                numeric,
                columns=[f"C_{c_idx}" for c_idx in range(numeric.shape[-1])],
            )
            numerics_df["Group"] = groups
            numerics_df = (
                numerics_df.groupby("Group").mean().reset_index().sort_values("Group")
            )
            del numerics_df["Group"]
            new_numeric = numerics_df.to_numpy()

        else:
            numerics_df = (
                pd.DataFrame(
                    {
                        "Numeric": numeric,
                        "Group": groups,
                    }
                )
                .groupby("Group")
                .Numeric.mean()
                .reset_index()
                .sort_values("Group")
            )
            new_numeric = numerics_df["Numeric"].to_numpy()

    return new_targets, new_numeric


def aggregate_classification_predictions_by_group(
    targets: Optional[Union[list, np.ndarray]],
    probabilities: Optional[Union[list, np.ndarray]] = None,
    predictions: Optional[Union[list, np.ndarray]] = None,
    groups: Optional[Union[list, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if groups is None:
        return targets, probabilities, predictions

    new_targets, new_probabilities = _shared_aggregate_predictions_by_group(
        targets=targets, numeric=probabilities, groups=groups
    )

    new_predictions = None
    if predictions is not None:
        predictions_df = (
            pd.DataFrame(
                {
                    "Prediction": predictions,
                    "Group": groups,
                }
            )
            .groupby(["Group", "Prediction"])
            .size()
            .reset_index()
            .sort_values(["Group", 0, "Prediction"], ascending=[True, False, True])
            .groupby("Group")
            .first()
            .reset_index()
            .sort_values("Group")
        )
        new_predictions = predictions_df["Prediction"].to_numpy()

    return new_targets, new_probabilities, new_predictions


def aggregate_regression_predictions_by_group(
    targets: Optional[Union[list, np.ndarray]],
    predictions: Optional[Union[list, np.ndarray]] = None,
    groups: Optional[Union[list, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if groups is None:
        return targets, predictions

    return _shared_aggregate_predictions_by_group(
        targets=targets, numeric=predictions, groups=groups
    )
