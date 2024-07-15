from functools import partial
from typing import Callable, Optional, Tuple, List, Union
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from utipy import Messenger

from generalize.utils.missing_data import remove_missing_data
from generalize.utils.normalization import standardize


def fit_statsmodels_model(
    x: np.ndarray,
    y: np.ndarray,
    task: str = "regression",
    add_intercept: bool = True,
    rm_missing: bool = False,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[List[float], List[float], float, float]:
    """
    Fit model from `statsmodels` and return coefficients, p-values and
    standard error of the first slope.

    Can fit either a `statsmodels.OLS` linear regression model
    or a `statsmodels.Logit` logistic regression model

    Parameters
    ----------
    x: nD `numpy.ndarray`
        Data to fit models to.
        NOTE: Only tested with a single feature.
    y: 1D `numpy.ndarray`
        Target values.
    task
        Task to perform. One of:
            "regression": Fits `statsmodels.OLS` linear regression model.
            "classification": Fits `statsmodels.Logit` logistic regression model.
    add_intercept
        Whether to add an intercept to the linear model.
    rm_missing
        Whether to remove missing data (i.e. `numpy.nan`).

    Returns
    -------
    List of floats
        Coefficients
    List of floats
        P-values
    float
        R2
    float
        Standard error of first slope
    """

    assert len(y) == x.shape[0], "`x` and `y` must have same length."
    assert task in ["classification", "regression"]

    if not isinstance(y, np.ndarray):
        if not isinstance(y, list):
            raise ValueError(
                "`y` should be either a `np.ndarray` or a `list` of numbers."
            )
        y_dtype = np.float32 if task == "regression" else np.int32
        y = np.asarray(y, dtype=y_dtype)

    if rm_missing:
        # Remove values/rows with `numpy.nan` in them
        y, x = remove_missing_data([y, x])

    if len(y) == 0:
        # TODO Handle in using functions?
        msg = " `y` was empty (length 0)."
        messenger(f"fit_statsmodels_model: {msg}")
        warnings.warn(msg)
        return None, None, None, None

    # Add intercept
    if add_intercept:
        x = sm.add_constant(x)

    # Initialize model
    if task == "regression":
        model = sm.OLS(y, x)
    else:
        model = sm.Logit(y, x)

    # Fit model
    try:
        result = model.fit()
    except PerfectSeparationError:
        # Classification with perfectly separated classes
        msg = "Got a `PerfectSeparationError`. Returning NaNs."
        messenger(f"fit_statsmodels_model: {msg}")
        warnings.warn(msg)
        #  [Intercept, Slope] [P(Intercept), P(Slope)]  R2  std_err
        return [np.nan, np.nan], [np.nan, np.nan], np.nan, np.nan

    # Standard error for slope
    # TODO Make it work with n>1 features
    slope_std_err = None
    if x.ndim <= 2:
        slope_std_err = result.t_test([0, 1]).sd[0, 0]

    if hasattr(result, "rsquared"):
        rsquared = result.rsquared
    elif hasattr(result, "prsquared"):
        rsquared = result.prsquared

    return result.params, result.pvalues, rsquared, slope_std_err


def fit_statsmodels_univariate_models(
    x: np.ndarray,
    y: np.ndarray,
    feature_set: Optional[int] = None,
    task: str = "regression",
    standardize_cols: bool = True,
    add_intercept: bool = True,
    df_out: bool = True,
    rm_missing: bool = False,
    seed: int = 1,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], pd.DataFrame]:
    """
    A single-predictor model is fitted to each feature in `data`.

    Parameters
    ----------
    x: nD `numpy.ndarray`
        Data to fit models on each feature of separately.
        Expected shape: (num_samples, feature_sets (optional), features).
    y: 1D `numpy.ndarray`
        Target values.
    feature_set
        Which feature set to use.
        NOTE: Set to `None` when `x` is 2D.
    task
        Task to perform. One of:
            "regression": Fits `statsmodels.OLS` linear regression model.
            "classification": Fits `statsmodels.Logit` logistic regression model.
    standardize_cols
        Whether to standardize the feature before fitting its model.
    add_intercept
        Whether to add an intercept to the model.
    df_out
        Whether to return a `pandas.DataFrame` (default) or separate numpy arrays.
    rm_missing
        Whether to remove missing data (i.e. `numpy.nan`).
    seed:
        Random seed.
        NOTE: Sets `numpy.random.seed()`. It is unclear whether this is enough
        to ensure reproducibility.

    Returns
    -------
    df_out == True:
        pd.DataFrame
            Coefficients, p-values, R2, and the slope standard errors.
    df_out == FALSE:
        numpy.ndarray with floats
            Coefficients
        numpy.ndarray with floats
            P-values
        numpy.ndarray with floats
            R2 values
        numpy.ndarray with floats
            Slope standard errors
    """

    transform_fn = standardize if standardize_cols else lambda x: x

    if x.ndim == 2 and feature_set is not None:
        raise ValueError("When `x` is a 2D array, `feature_set` must be `None`.")
    if x.ndim == 3 and feature_set is None:
        raise ValueError("When `x` is a 3D array, `feature_set` must be specified.")
    assert task in ["classification", "regression"]

    def slicer(x, feature_set, c):
        if feature_set is None:
            return x[:, c]
        return x[:, feature_set, c]

    if seed is not None:
        np.random.seed(seed)

    # Add the static arguments to the model function
    model_fitter = partial(
        fit_statsmodels_model,
        y=y,
        task=task,
        add_intercept=add_intercept,
        rm_missing=rm_missing,
        messenger=messenger,
    )

    # Fit all models and extract their summary information
    coeffs, p_values, r2s, slope_std_errs = zip(
        *[
            model_fitter(x=transform_fn(slicer(x, feature_set, c)))
            for c in range(x.shape[-1])
        ]
    )

    # Return either as numpy arrays or as a data frame

    if not df_out:
        # Return as numpy arrays
        return (
            np.asarray(coeffs),
            np.asarray(p_values),
            np.asarray(r2s),
            np.asarray(slope_std_errs),
        )

    else:
        # Convert to data frame

        # Parameters
        params_df = pd.DataFrame(coeffs)
        params_df.columns = ["Intercept", "Slope"]

        # P-values
        p_values_df = pd.DataFrame(p_values)
        p_values_df.columns = ["P(Intercept)", "P(Slope)"]

        # R squared values
        r2_df = pd.DataFrame(r2s)
        r2_df.columns = ["R2"]

        # R squared values
        se_df = pd.DataFrame(slope_std_errs)
        se_df.columns = ["SE(Slope)"]

        # Combine all data frames
        combined_df = pd.concat([params_df, p_values_df, r2_df, se_df], axis=1)

        # Add model function
        combined_df["Model Function"] = (
            "statsmodels.OLS" if task == "regression" else "statsmodels.Logit"
        )

        # Ensure index is as new!
        combined_df = combined_df.reset_index(drop=True)

        return combined_df
