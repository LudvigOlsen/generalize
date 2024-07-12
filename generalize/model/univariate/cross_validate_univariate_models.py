from typing import Optional, Union, List
from functools import partial
import warnings
import numpy as np
import pandas as pd
from utipy import move_column_inplace
from sklearn.linear_model import LinearRegression, LogisticRegression

from generalize.model.cross_validate.cross_validate import cross_validate
from generalize.model.pipeline.pipeline_designer import PipelineDesigner


def cross_validate_univariate_models(
    x: np.ndarray,
    y: np.ndarray,
    task: str,
    k: int = 10,
    groups: Optional[np.ndarray] = None,
    split: Optional[Union[List[Union[int, str]], np.ndarray]] = None,
    eval_by_split: bool = False,
    positive_label: Optional[Union[str, int]] = None,
    y_labels: Optional[dict] = None,
    standardize_cols: bool = True,
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_loss_by_class: bool = False,
    weight_per_split: bool = False,
    rm_missing: bool = False,
    reps: int = 1,
    num_jobs: int = 1,
    seed: int = 1,
    add_info_cols: bool = True,
) -> pd.DataFrame:
    """
    For each feature in `x`, a single-predictor model is cross-validated to evaluate
    predictive potential.

    Parameters
    ----------
    x: 2D `numpy.ndarray` with data to cross-validate each feature of.
        Expected shape: (num_samples, features).
    y: 1D `numpy.ndarray` with target values.
    task: The task to perform. Either "classification" or "regression".
    k: Number of folds in the cross-validation.
    groups: 1D `numpy.ndarray` or `None`
        An array of group IDs (one for each of the data points).
        (Almost always) ensures that all data points with the same
        ID are put in the same fold during cross-validation.
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
    eval_by_split : bool
        Whether to evaluate by splits instead of with all predictions at once.
        When paired with `split`, the output will also have metrics
        for each split. E.g. when each part is a dataset and we wish to
        have scores for each separately.
    positive_label: Label (likely class index) for the positive class,
        in binary classification.
    y_labels: Dict mapping unique values in `y` to their class names.
        Only used during classification (see `task`).
        When `None`, the names in `y` are used.
    standardize_cols: Whether to standardize the feature before fitting its model
        (within the cross-validation).
    aggregate_by_groups : bool
        Whether to aggregate predictions per group, prior to evaluation.
        For regression predictions and predicted probabilities,
        the values are averaged per group.
        For class predictions, we use majority vote. In ties, the
        lowest class index is selected.
    weight_loss_by_groups : bool
        Whether to weight samples by the group size in training loss.
        Each sample in a group gets the weight `1 / group_size`.
        Passed to model's `.fit(sample_weight=)` method.
    weight_loss_by_class : bool
        Whether to weight the loss by the inverse class frequencies.
        The weights are calculated per training set.
    weight_per_split : bool
        Whether to perform the loss weighting separately per split.
        Affects both class- and group-based weighting.
        E.g. when each fold is a dataset with some set of biases
        that shouldn't be ascribed to the majority class. Instead of weighting based
        on the overall class imbalance, we fix the imbalance within each dataset.
        NOTE: May not be meaningful when `split` is not specified.
    rm_missing: Whether to remove missing data (i.e. `numpy.nan`).
    reps: Number of repetitions.
    seed: Random seed.
    add_info_cols: Whether to add information columns to output.
        Currently `Num Repetitions` and `Feature Index`.

    Returns
    -------
    `pandas.DataFrame`
        Concatenated outputs of `cross_validate()` for each feature.
    """
    assert x.ndim == 2
    assert reps >= 1
    if reps > 1 and split is not None:
        warnings.warn(
            "When `split` is specified, `reps > 1` leads to redundant computation."
        )
    assert y_labels is None or isinstance(y_labels, dict)
    assert positive_label is None or isinstance(positive_label, (str, int))
    assert task in ["classification", "regression"]

    # Get model object
    if task == "classification":
        unique_labels = np.unique(y)
        is_binary = not len(unique_labels) > 2
        task = "binary_classification" if is_binary else "multiclass_classification"
        assert is_binary, "Multiclass classification has not been implemented yet."
        model = LogisticRegression(random_state=seed, tol=0.0001, max_iter=5000)
    elif task == "regression":
        # TODO: Should be just pure linear regression - only one feature at a time!
        model = LinearRegression()

    transformers = None
    if standardize_cols:
        transformers = (
            PipelineDesigner()
            .add_step(
                "standardize", "scale_features", add_dim_transformer_wrapper=False
            )
            .build()
        )

    cross_validator = partial(
        cross_validate,
        y=y,
        model=model,
        groups=groups,
        positive=positive_label,
        y_labels=y_labels,
        k=k,
        split=split,
        eval_by_split=eval_by_split,
        task=task,
        rm_missing=rm_missing,
        reps=reps,
        transformers=transformers,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        aggregate_by_groups=aggregate_by_groups,
        num_jobs=num_jobs,
        seed=seed,
        messenger=None,  # sets verbose = False
    )

    def _call_cross_validator(d, feature_idx):
        """
        Wrapper for calling cross-validator on `x`.
        """

        cv_out = cross_validator(d)

        if reps > 1:
            evaluation_summary = cv_out["Evaluation"]["Summary"]["Scores"].copy()
            evaluation_summary = evaluation_summary.loc[
                evaluation_summary["Measure"] == "Average"
            ]
            evaluation_summary.drop(columns=["Measure"], inplace=True)
            out = evaluation_summary

            # Add info columns
            out["Num Classes"], out["Positive Class"] = cv_out["Evaluation"][
                "Evaluations"
            ]["Scores"].loc[0, ["Num Classes", "Positive Class"]]

            if eval_by_split and split is not None:
                out = _add_splits_metrics(cv_out, out)
        else:
            rep_evaluation = cv_out["Evaluation"]["Evaluations"]["Scores"]
            rep_evaluation = rep_evaluation.drop(
                columns=["Repetition"], errors="ignore"
            )
            out = rep_evaluation

            if eval_by_split and split is not None:
                out = _add_splits_metrics(cv_out, out)

        for col in ["Num Classes", "Positive Class"]:
            if col in out.columns:
                move_column_inplace(out, col, len(out.columns) - 1)

        if add_info_cols:
            out.loc[:, "Num Repetitions"] = reps
            out.loc[:, "Feature Index"] = feature_idx

        # Get Max. Youden's J threshold version
        out = _filter_threshold_versions(out, rm_cols=True)

        if out.shape[0] > 1:
            raise ValueError(
                "Output had more than 1 row. Something has not been implemented right. Please report."
            )
        return out

    # Run cross-validation on each feature separately
    # And concatenate the output data frames
    return pd.concat(
        [
            _call_cross_validator(d=x[:, feature_idx], feature_idx=feature_idx)
            for feature_idx in range(x.shape[-1])
        ],
        ignore_index=True,
        axis=0,
    )


def _add_splits_metrics(cv_out, out):
    split_evals = cv_out["Evaluation"]["Summary"]["Splits"]["Scores"].copy()
    split_evals = split_evals.loc[split_evals["Measure"] == "Average"]
    split_evals.drop(columns=["Measure"], inplace=True)

    out["Fold"] = "Average"
    out = pd.concat([out, split_evals])
    out = _filter_threshold_versions(out, rm_cols=False)
    out = out.pivot_table(columns=["Fold"], index=["Threshold Version"])
    out = out.sort_index(axis=1, level=1)
    out = out.sort_index(axis=1, level=1, key=lambda ks: [len(ki) for ki in ks])
    out.columns = [f"{x}_{y}" for x, y in out.columns]
    out = out.reset_index()
    out.columns = [s.replace("_Average", "") for s in out.columns]

    return out


def _filter_threshold_versions(df: pd.DataFrame, rm_cols: bool = True) -> pd.DataFrame:
    # Get Max. Youden's J threshold version
    if "Threshold Version" in df.columns:
        df = df.loc[df["Threshold Version"] == "Max. J Threshold"]
        if rm_cols:
            df = df.drop(
                columns=["Threshold Version", "Threshold"],
                errors="ignore",
            )
    return df
