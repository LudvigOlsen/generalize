"""
This module fits linear/logistic regression models with each feature on their own. 
E.g. y ~ x1, y ~ x2, y ~ x3, ...

"""

from typing import Callable, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import warnings

from utipy import Messenger, check_messenger, move_column_inplace

from generalize.model.univariate.calculate_tree_importance import (
    calculate_tree_feature_importance,
)
from generalize.model.univariate.cross_validate_univariate_models import (
    cross_validate_univariate_models,
)
from generalize.model.univariate.fit_statsmodels_model import (
    fit_statsmodels_univariate_models,
)
from generalize.utils.normalization import standardize_features_3D
from generalize.model.transformers.row_scaler import RowScaler

# TODO: Test in general and multiclass
# TODO: Test: Use groups in cv evaluation and sample weight in models!


def evaluate_univariate_models(
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    names: Optional[List[str]] = None,
    alternative_names: Optional[List[str]] = None,
    feature_sets: Optional[List[int]] = None,
    groups: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    k: int = 10,
    split: Optional[Union[List[Union[int, str]], np.ndarray]] = None,
    eval_by_split: bool = False,
    bonferroni_correct: bool = True,
    standardize_cols: bool = True,
    standardize_rows: bool = False,
    standardize_rows_feature_groups: Optional[List[List[int]]] = None,
    weight_loss_by_groups: bool = False,
    weight_loss_by_class: bool = False,
    weight_per_split: bool = False,
    aggregate_by_groups: bool = False,
    positive_label: Optional[Union[int, str]] = None,
    y_labels: Optional[dict] = None,
    name_cols: List[str] = ["name", "alt_name"],
    feature_set_prefix: str = "Feature Set",
    num_jobs: int = 1,
    seed: int = 1,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Tuple[pd.DataFrame, str]:
    """
    Measures how much potential each feature has on its own in a regression/classification task.

    1) Fits linear/logistic regression model with each feature separately, using all samples in `x`.
       In output: Intercept, slope, slope's standard error (std_err),
                  p value (p_value), negative log-p-value (neg_log_p_value),
                  significant/insignificant, significance threshold, and
                  group (0: insignificant, 1: signicant and negative slope coefficient,
                         2: significant and positive slope coefficient).

    2) Cross-validates linear/logistic regression model with each feature separately,
       using 10/10 nested cross-validation with randomly split folds (stratified in classification).
       In output: Mean Absolute Error (MAE), Root Mean Square Error (RMSE).

    3) Fits a random forest regressor/classifier on all features at once and extracts its feature importance weights.
       In output: Feature importance (tree_importance).

    Parameters
    ----------
    x: 2/3D `numpy.ndarray`
        Data to evaluate each feature of.
        Expected shape: (num_samples, (optional: feature_sets), features).
    y: 1D `numpy.ndarray`
        Target values.
    groups: 1D `numpy.ndarray` or `None` (for cross-valididation only)
        An array of group IDs (one for each of the data points).
        (Almost always) ensures that all data points with the same
        ID are put in the same fold during **cross-validation**.
        WARNING: When `split` is specified, it has precedence. IDs that
        are present in multiple splits are NOT respected as a single
        entity.
        When `aggregate_by_groups` is enabled:
            During evaluation of cross-validation predictions, the predictions
            are first aggregated per group by either averaging (regression or
            classification probabilities) or majority vote (class predictions).
        **Train Only**: Wrap a group identifier in the `"train_only(ID)"` string
        (where `ID` is the group identifier) for samples that should only be used
        for training. When `split` is specified, indices that are
        specified as "train only" in either are used only during training!
        NOTE: Beware of introducing train/test leakage.
    task: str
        Task to evaluate features on.
        One of {"classification", "regression"}.
    names
        List of names/IDs for each of the features.
        The order must match the feature indices in `x`.
    alternative_names
        List of alternative names for each of the features.
        The order must match the feature indices in `x`. (Optional)
    feature_sets
        Which feature sets to use.
    alpha
        The significance threshold for the model coefficients.
        May be bonferroni corrected (see `bonferroni_correct`).
    bonferroni_correct
        Whether to apply bonferroni correction to the p-values.
        I.e. `p = p * num_features`
    standardize_rows
        Whether to standardize the features row-wise first.
        This is done prior to the potential column standardization.
    standardize_cols
        Whether to standardize the feature columns first.
    aggregate_by_groups : bool
        Whether to aggregate predictions per group, prior to evaluation **in cross-validation**.
        For regression predictions and predicted probabilities,
        the values are averaged per group.
        For class predictions, we use majority vote. In ties, the
        lowest class index is selected.
    weight_loss_by_groups : bool
        Whether to weight samples by the group size in training loss **during cross-validation**.
        Each sample in a group gets the weight `1 / group_size`.
        Passed to model's `.fit(sample_weight=)` method.
    weight_loss_by_class : bool
        Whether to weight the loss by the inverse class frequencies **during cross-validation**.
        The weights are calculated per training set.
    weight_per_split : bool
        Whether to perform the loss weighting separately per split **during cross-validation**.
        Affects both class- and group-based weighting.
        E.g. when each fold is a dataset with some set of biases
        that shouldn't be ascribed to the majority class. Instead of weighting based
        on the overall class imbalance, we fix the imbalance within each dataset.
        NOTE: May not be meaningful when `split` is not specified.
    k
        Number of folds in the cross-validation.
        Ignored when `split` is specified.
    split
        Split indices for the cross-validation.
    eval_by_split : bool
        Whether to evaluate by splits instead of with all predictions at once.
        When paired with `split`, the output will also have metrics
        for each split. E.g. when each part is a dataset and we wish to
        have scores for each separately.
    positive_label
        Label (likely class index) for the positive class,
        in cross-validation of binary classification.
        **Required** for *binary classification*.
    y_labels
        Dict mapping unique values in `y` to their class names.
        Only used during classification (see `task`).
        When `None`, the names in `y` are used.
    name_cols
        List of names for the name columns. First element is for the normal names
        and the second is for the alternative names.
    seed
        Random seed.
    messenger: `utipy.Messenger` or None
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g. `print`)
        and potential indentation.

    Returns
    -------
    `pandas.DataFrame`
        Data frame with results. Has columns:
            name_col[0]: Name/ID of the feature. The name of this column is provided through `name_cols`.
            name_col[1]: Alternative name of the feature. (Only when `alternative_names` is not `None`).
                The name of this column is provided through `name_cols`.
            <feature set>: The feature set used to create the activation scores.
            intercept: Intercept of the full-dataset, single-predictor linear regression model.
            slope: Slope coefficient of the full-dataset, single-predictor linear regression model.
            std_err: Standard error of the slope.
            p_value: P-value for the slope coefficient (slope).
            neg_log_p_value: -log10(p_value).
            significant: Whether the p_value is <= alpha (after potential bonferroni correction).
            R2: R squared of the single-predictor linear/logistic regression model.
            if task is "regression":
                MAE: Mean absolute error from 10-fold cross-validated single-predictor linear regression model.
                RMSE: Root Mean Square Error from 10-fold cross-validated single-predictor linear regression model.
            if task is "classification":
                TODO Add metric cols here
            tree_importance: Feature importance from random forest fitted to all features, using the full dataset.
            group:
                0 = insignificant
                1 = signicant and negative slope coefficient
                2 = significant and positive slope coefficient
                Used in volcano plots.
            threshold: The (potentially corrected) `alpha` used when calculating significance.
            abs_slope: Absolute value of slope.
            array_index: The position of the feature in the dataset array (third dimension).

    str
        A string that explains the analysis for writing to a README file.
    """

    assert task in ["regression", "classification"]
    assert isinstance(aggregate_by_groups, bool)
    assert isinstance(weight_loss_by_groups, bool)
    assert feature_sets is None or (
        isinstance(feature_sets, list) and len(feature_sets) >= 1
    )
    if x.ndim == 2 and feature_sets is not None:
        raise ValueError("When `x` is a 2D array, `feature_sets` should be `None`.")

    feature_sets_was_none = feature_sets is None
    if feature_sets is None:
        # Code currently expects this
        # TODO: Update code base to allow feature_sets = None
        if x.ndim == 3:
            warnings.warn(
                "`feature_sets` is `None` is not implemented "
                "yet - uses first feature set `[0]`."
            )
        feature_sets = [0]

    if x.ndim == 2:
        x = np.expand_dims(x, axis=1)
    elif x.ndim != 3:
        raise ValueError(f"`x` must have either 2 or 3 dimensions, but had {x.ndim}.")

    if names is None:
        names = [str(i) for i in range(x.shape[-1])]
    assert len(names) == x.shape[2]
    assert isinstance(name_cols, list)
    if alternative_names is not None:
        assert len(names) == len(alternative_names)
        if len(name_cols) != 2:
            raise ValueError(
                "When `alternative_names` are supplied, `name_cols` must have length 2."
            )
    elif len(name_cols) < 1 or len(name_cols) > 2:
        raise ValueError(
            "`name_cols` must be a list with 1 or 2 strings (2 are required when `alternative_names` are supplied)."
        )

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    messenger(f"Will evaluate each feature on {task}")

    # Standardize each sample row-wise (in feature groups)
    if standardize_rows:
        messenger("Standardizing data row-wise")
        assert x.ndim == 3
        if standardize_rows_feature_groups is None or not len(
            standardize_rows_feature_groups
        ):
            standardize_rows_feature_groups = [range(x.shape[-1])]
        for fs in range(x.shape[1]):
            for sr_feature_group in standardize_rows_feature_groups:
                row_scaler = RowScaler(center="mean", scale="std", copy=False).fit(
                    X=x[:, fs, sr_feature_group], y=y
                )
                x[:, fs, sr_feature_group] = row_scaler.transform(
                    X=x[:, fs, sr_feature_group]
                )

    # Standardize each feature for each feature set
    if standardize_cols:
        messenger("Standardizing features column-wise")
        x = standardize_features_3D(x)

    # Initialize list of data frames
    feature_set_evaluations = []

    # Prepare bonferroni correction
    num_tests = x.shape[-1] * len(feature_sets)

    # Evaluate features for each of the feature sets
    messenger("Evaluating each feature set")
    for feature_set in feature_sets:
        # Extract current feature set
        current_data = x[:, feature_set, :]

        # Fit models to each feature separately
        # using the full dataset to get coefficients and p-values
        # TODO use num_jobs and parallelize
        coeffs, p_values, r2s, slope_std_errs = fit_statsmodels_univariate_models(
            x=current_data,
            y=y,
            task=task,
            standardize_cols=False,
            df_out=False,
            seed=seed,
            messenger=messenger,
        )

        # Separate intercepts and coefficients
        intercepts, slopes = zip(*coeffs)

        # Separate p-values for intercepts (not used) and coefficients
        _, p_values = zip(*p_values)
        p_values = np.asarray(list(p_values))

        # Apply bonferroni correction to p-values
        # I.e. multiple p-values with the number of tests
        if bonferroni_correct:
            p_values *= num_tests
            # Truncate to 0-1 range
            p_values[p_values > 1] = 1.0

        # Cross-validate each feature separately
        # to get RMSE and MAE scores
        cv_metrics = cross_validate_univariate_models(
            x=current_data,
            y=y,
            groups=groups,
            task=task,
            standardize_cols=True,
            weight_loss_by_groups=weight_loss_by_groups,
            weight_loss_by_class=weight_loss_by_class,
            weight_per_split=weight_per_split,
            aggregate_by_groups=aggregate_by_groups,
            k=k,
            split=split,
            eval_by_split=eval_by_split,
            positive_label=positive_label,
            y_labels=y_labels,
            num_jobs=num_jobs,
            seed=seed,
            add_info_cols=False,
        )
        # TODO: Should probably have metrics per split when splits are pre-defined (e.g. cross-datasets)

        # Calculate random forest feature importances
        tree_importances = calculate_tree_feature_importance(
            x=current_data, y=y, standardize_cols=False, seed=seed
        )

        # Whether the p-value is below significance level
        is_significant = p_values <= alpha

        def get_group_id(slope, is_signif):
            """
            Place data point in group:
                0: non-significant
                1: coeff < 0
                2: coeff >= 0
            """
            if not is_signif:
                return 0
            return 1 if slope < 0 else 2

        # Put into groups of significance + pos/neg coefficient
        group_factor = [
            get_group_id(slope, is_signif)
            for slope, is_signif in zip(slopes, is_significant)
        ]

        # Calculate negative log10 p-values
        neg_log_p_values = -np.log10(p_values)

        # Build data frame
        df_pre_cv = pd.DataFrame(
            {
                feature_set_prefix: (
                    feature_set if not feature_sets_was_none else np.nan
                ),
                "intercept": list(intercepts),
                "slope": list(slopes),
                "std_err": list(slope_std_errs),
                "p_value": list(p_values),
                "neg_log10_p_value": list(neg_log_p_values),
                "significant": list(is_significant),
                "R2": list(r2s),
            }
        )
        df_post_cv = pd.DataFrame(
            {
                "tree_importance": tree_importances,
                "group": group_factor,
                "significance_threshold": alpha,
                "num_tests": num_tests,
                "abs_slope": list(np.abs(slopes)),
                "array_index": range(len(names)),
            }
        )

        # Combine data frame columns
        df = pd.concat([df_pre_cv, cv_metrics, df_post_cv], axis=1)

        # Add name column at first index
        df[name_cols[0]] = names
        move_column_inplace(df=df, col=name_cols[0], pos=0)

        # Add alternative names when available
        if alternative_names is not None:
            df[name_cols[1]] = alternative_names
            move_column_inplace(df=df, col=name_cols[1], pos=1)

        # Add evaluation df to list of dfs
        feature_set_evaluations.append(df)

    messenger("Concatenating results for all features")
    all_feature_sets_df = pd.concat(feature_set_evaluations, ignore_index=True)

    # Sort by most interesting features across feature_sets
    messenger("Sorting evaluations by how interesting they are")
    all_feature_sets_df.sort_values(
        ["p_value", "abs_slope", "R2", feature_set_prefix],
        ascending=[True, False, False, True],
        inplace=True,
        ignore_index=True,
    )

    messenger("Generating README text")
    readme_text = explain_feature_evaluation_df(
        task=task,
        k=k,
        split=split,
        name_col=name_cols[0],
        alt_name_col=name_cols[1],
        feature_set_prefix=feature_set_prefix,
    )

    return all_feature_sets_df, readme_text


def explain_feature_evaluation_df(
    task: str,
    k: int,
    split: Optional[Union[List[Union[int, str]], np.ndarray]],
    name_col: str,
    alt_name_col: str,
    feature_set_prefix: str,
) -> str:
    """
    Creates a written explanation of the output of `evaluate_univariate_models()`.
    """
    model_name = "linear" if task == "regression" else "logistic"
    metrics = {
        "regression": "Mean Absolute Error (MAE), Root Mean Square Error (RMSE)",
        "classification": "Accuracy, Area Under The ROC Curve (AUC",
    }
    rsquared = "McFadden's Pseudo-" if task == "classification" else ""
    split_text = (
        f"{k} {'stratified' if task== 'classification' else 'randomly split'} folds"
        if split is None
        else f"{len(np.unique(split))} pre-specified folds"
    )
    explanation = f"""
The feature evaluation estimates the {task} task relevance separately for each feature.

The creation has three main steps that are performed for each 
of the specified features separately.

1) We fit a single-predictor {model_name} regression model for each feature separately, 
   using {'all' if task == 'regression' else 'the relevant'} samples in the data. 
   In output: Intercept, slope, slope's standard error (std_err),
       p-value, negative log-p-value (neg_log_p_value), {rsquared}R-squared (R2),
       significant/insignificant, significance threshold, and
       group (0: insignificant, 1: signicant and negative slope coefficient,
              2: significant and positive slope coefficient).

2) We cross-validate a single-predictor {model_name} regression model for each feature
   separately, using {split_text}.
   In output: {metrics[task]}.

3) We fit a random forest regressor on all features at once and extract its 
   feature importance weights.
   In output: Feature importance (tree_importance).

The columns in the csv file are thus:
    {name_col} Name/ID of the feature.
    {alt_name_col}: Alternative name of the feature. (May not be present)
    {feature_set_prefix}: The feature set used.
    intercept: Intercept of the full-dataset, single-predictor {model_name} regression 
      model.
    slope: Slope coefficient of the full-dataset, single-predictor {model_name} 
      regression model.
    std_err: Standard error of the slope.
    p_value: P-value for the slope coefficient (slope).
    neg_log10_p_value: -log10(p_value).
    significant: Whether the p_value is <= alpha (after potential bonferroni
      correction).
    R2: R squared of the single-predictor {model_name} regression model.
    <Cross-validation Metrics>: Multiple columns with metrics from
      cross-validation of the single-predictor {model_name}
      regression model.
    tree_importance: Feature importance from random forest fitted to all features,
      using the full dataset.
    group: 
        0 = insignificant
        1 = signicant and negative slope coefficient
        2 = significant and positive slope coefficient
        Used in volcano plots.
    threshold: The alpha used when calculating significance.
    abs_slope: Absolute value of the slope. Used when sorting the rows.
    array_index: The position of the feature in the dataset array (third dimension).

The rows are sorted by the lowest p_value, highest abs_slope, highest R2,
and lowest feature_set index.
        """

    return explanation
