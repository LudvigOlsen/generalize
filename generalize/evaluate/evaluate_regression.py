from typing import List, Union, Optional
import numpy as np
import pandas as pd

from generalize.utils.math import scalar_safe_div
from generalize.evaluate.prepare_inputs import aggregate_regression_predictions_by_group


class RegressionEvaluator:
    METRICS = [
        "RMSE",
        "MAE",
        "RMSLE",
        "MALE",
        "RAE",
        "RSE",
        "RRSE",
        "MAPE",
        "NRMSE_RNG",
        "NRMSE_IQR",
        "NRMSE_STD",
        "NRMSE_AVG",
    ]

    @staticmethod
    def evaluate(targets, predictions, groups=None, ignore_missing=False):
        reg_evaluator = RegressionEvaluator()
        return reg_evaluator(
            targets=targets,
            predictions=predictions,
            groups=groups,
            ignore_missing=ignore_missing,
        )

    def __call__(
        self,
        targets: Union[List, np.ndarray],
        predictions: Union[List, np.ndarray],
        groups: Optional[Union[List, np.ndarray]],
        ignore_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate predictions against true values from a regression task.

        See metric formulas at http://ludvigolsen.dk/cvms/metrics/#gaussian-metrics

        Parameters
        ----------
        targets
            The true values as a list or `numpy.ndarray`.
        predictions
            The predicted values as a list or `numpy.ndarray`.
        groups
            Group identifiers (e.g. subject IDs).
            Predictions are averaged per group before evaluation.
        ignore_missing
            Whether to ignore missing values in
            `targets` and `predictions`.

        Returns
        -------
        `pandas.DataFrame`
            Data frame with the following metrics:
                RMSE: Root Mean Square Error
                MAE: Mean Absolute Error
                RMSLE: Root Mean Square Log Error
                MALE: Mean Absolute Log Error
                RAE: Relative Absolute Error
                RSE: Relative Squared Error
                RRSE: Root Relative Squared Error
                MAPE: Mean Absolute Percentage Error
                NRMSE_RNG: Normalized RMSE (by target range)
                NRMSE_IQR: Normalized RMSE (by target IQR)
                NRMSE_STD: Normalized RMSE (by target STD)
                NRMSE_AVG: Normalized RMSE (by target mean)
        """
        # Remove singleton dimensions as they lead to very wrong results
        targets_ndims = targets.ndim
        targets = np.asarray(targets, dtype=np.float32).squeeze()
        predictions = np.asarray(predictions, dtype=np.float32)
        if groups is not None:
            groups = np.asarray(groups)
        assert (
            predictions.ndim == 1
        ), f"Currently only supports 1D prediction arrays but `predictions` had {predictions.ndim} dimensions."
        assert targets.ndim == 1, (
            "Currently only supports 1D target arrays or 2D arrays where the "
            f"last dimension is a singleton dimension (size 1). `targets` had {targets_ndims} dimensions."
        )

        # Average predictions by group when groups are specified
        targets, predictions = aggregate_regression_predictions_by_group(
            targets=targets,
            predictions=predictions,
            groups=groups,
        )

        return RegressionEvaluator._evaluate_regression(
            targets=targets, predictions=predictions, ignore_missing=ignore_missing
        )

    @staticmethod
    def _evaluate_regression(
        targets: Union[List, np.ndarray],
        predictions: Union[List, np.ndarray],
        ignore_missing: bool = False,
    ) -> pd.DataFrame:
        # Aggregation functions with/out ignoring NaNs
        if ignore_missing:
            sum_fn = np.nansum
            mean_fn = np.nanmean
            max_fn = np.nanmax
            min_fn = np.nanmin
            std_fn = np.nanstd

            def iqr_fn(x):
                return np.subtract(*np.nanpercentile(x, [75, 25]))

        else:
            sum_fn = np.sum
            mean_fn = np.mean
            max_fn = np.max
            min_fn = np.min
            std_fn = np.std

            def iqr_fn(x):
                return np.subtract(*np.percentile(x, [75, 25]))

        # Target descriptors
        targets_mean = mean_fn(targets)
        targets_range = max_fn(targets) - min_fn(targets)
        targets_iqr = iqr_fn(targets)
        targets_std = std_fn(targets)

        # Residuals
        residuals = targets - predictions
        squared_residuals = residuals**2
        abs_residuals = np.abs(residuals)

        # Centered targets
        targets_centered = targets - targets_mean
        abs_targets_centered = np.abs(targets_centered)
        square_targets_centered = targets_centered**2

        # Total absolute error
        tae = sum_fn(abs_residuals)
        # Total square error
        tse = sum_fn(squared_residuals)
        # Mean absolute error
        mae = mean_fn(abs_residuals)
        # Mean square error
        mse = mean_fn(squared_residuals)
        # Root mean square error
        rmse = np.sqrt(mse)

        # Normalized RMSE scores https://en.wikipedia.org/wiki/Root-mean-square_deviation
        nrmse_iqr = scalar_safe_div(rmse, targets_iqr)
        nrmse_rng = scalar_safe_div(rmse, targets_range)
        nrmse_std = scalar_safe_div(rmse, targets_std)
        nrmse_avg = scalar_safe_div(rmse, targets_mean)

        # Relative absolute error
        rae = scalar_safe_div(tae, sum_fn(abs_targets_centered))
        # Relative squared error
        rse = scalar_safe_div(tse, sum_fn(square_targets_centered))
        # Root relative squared error
        rrse = np.sqrt(rse)
        # Absolute percentage errors
        # Note: Wiki has percentage error be ((y-p)/y) but we have ((p-y)/y)
        ape = np.abs(residuals / targets)
        # Mean absolute percentage error
        mape = mean_fn(ape)

        # Log error
        if predictions.min() >= 0 or targets.min() >= 0:
            try:
                le = np.log(1 + predictions) - np.log(1 + targets)
            except Exception as e:
                le = np.empty(predictions.shape)
                le[:] = np.nan

            # Mean squared log error
            msle = mean_fn(le**2)
            # Root mean squared log error
            rmsle = np.sqrt(msle)
            # Mean absolute log error
            male = mean_fn(np.abs(le))
        else:
            msle = np.nan
            rmsle = np.nan
            male = np.nan

        return pd.DataFrame(
            {
                "RMSE": [rmse],
                "MAE": [mae],
                "RMSLE": [rmsle],
                "MALE": [male],
                "RAE": [rae],
                "RSE": [rse],
                "RRSE": [rrse],
                "MAPE": [mape],
                "NRMSE_RNG": [nrmse_rng],
                "NRMSE_IQR": [nrmse_iqr],
                "NRMSE_STD": [nrmse_std],
                "NRMSE_AVG": [nrmse_avg],
            }
        )
