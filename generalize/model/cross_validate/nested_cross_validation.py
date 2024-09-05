from functools import partial
import pathlib
import string
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from utipy import mk_dir, rm_dir, letter_strings, Messenger, check_messenger

from generalize.evaluate.prepare_inputs import BinaryPreparer
from generalize.model.cross_validate.sklearn_cross_validate import (
    cross_validate_with_predictions,
)
from generalize.model.cross_validate.grid_search import NestableGridSearchCV
from generalize.model.cross_validate.kfolder import (
    KFolder,
    GroupSpecifiedFolder,
    specified_folds_iterator,
)
from generalize.evaluate.evaluate_repetitions import evaluate_repetitions
from generalize.utils.missing_data import remove_missing_data
from generalize.model.pipeline.pipelines import (
    AttributeToDataFrameExtractor,
    create_pipeline,
)
from generalize.model.utils import (
    detect_train_only,
    add_split_to_groups,
    detect_no_evaluation,
)

# TODO Consider order of the arguments
# TODO Should outer_split be allowed to have a split per repetition?
# TODO Add option to retrieve an attribute from the pipeline. E.g. "nmf__H" for NMF components.
#      Should probably be a list of attribute names to extract from "best_estimator_" objects


def nested_cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    confounders: Optional[np.ndarray] = None,
    grid: Optional[Dict[str, List[any]]] = None,
    groups: Optional[np.ndarray] = None,
    positive: Optional[Union[str, int]] = None,
    y_labels: Optional[dict] = None,
    k_outer: Optional[int] = 10,
    k_inner: Optional[int] = 10,
    outer_split: Union[
        Optional[Union[List[Union[int, str]], np.ndarray]],
        Dict[str, Union[List[Union[int, str]], np.ndarray]],
    ] = None,
    split_weights: Optional[Dict[str, float]] = None,
    eval_by_split: bool = False,
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_loss_by_class: bool = False,
    weight_per_split: bool = False,
    tmp_path=None,
    process_predictions_fn: Callable = None,
    inner_metric: Union[str, List[str]] = "balanced_accuracy",
    refit: Union[bool, str, Callable] = True,
    task: str = "binary_classification",
    transformers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    train_test_transformers: List[str] = [],
    add_channel_dim: bool = False,
    add_y_singleton_dim: bool = False,
    rm_missing: bool = False,
    reps: int = 1,
    num_jobs: int = 1,
    seed: int = 1,
    grid_error_score: Union[str, int, float] = np.nan,
    cv_error_score: Union[str, int, float] = np.nan,
    store_attributes: List[AttributeToDataFrameExtractor] = None,
    identifier_cols_dict: dict = None,
    eval_idx_colname: str = "Repetition",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Perform nested cross-validation in order to evaluate the potential of a model architecture.

    Specify data points to only use as training data in both the inner and outer cross-validation
    via `outer_split` and/or `groups`.

    Parameters
    ----------
    x : 1/2/3D `numpy.ndarray`
        The data. Shape: (# samples, # feature sets (optional), # features).
        When given a 1D array, it is expanded to two dims with `np.expand_dims(x, 1)`.
    y : 1D `numpy.ndarray`
        Targets (one per sample).
    model : callable
        Model instance with a scikit-learn compatible interface.
        We actively support scikit-learn and skorch models.
        Added to a Pipeline (see `sklearn.pipeline.Pipeline`),
        why it must have `fit()`, `predict()` and `transform()` methods.
    grid : Dict[str, List[any]]
        Mapping of argument names to lists of values to try in grid search (i.e. hyperparameters).
        Cannot be empty (in that case, use regular cross-validation).
        Argument names must be prefixed with the name of the pipeline element using it.
        That is, `"model__"` (two underscores) for model parameters or the
        name of a specified transformer (see `transformers`) followed by two underscores
        (e.g. `"my_transformer__xxx"` if `transformers` had a `"my_transformer"` transformer
        accepting the `xxx` keyword during initialization).
    groups : 1D `numpy.ndarray` or `None`
        An array of group IDs (one for each of the data points).
        (Almost always) ensures that all data points with the same
        ID are put in the same fold during cross-validation (both inner and outer).
        WARNING: When `outer_split` is specified, it has precedence. IDs that
        are present in multiple outer splits are NOT respected as a single
        entity.
        See `aggregate_by_groups` for prediction aggregation during evaluation.
        **Train Only**: Wrap a group identifier in the `"train_only(ID)"` string
        (where `ID` is the group identifier) for samples that should only be used
        for training. When `outer_split` is specified, indices that are
        specified as "train only" in either are used only during training
        for both the inner and outer loop! NOTE: Beware of introducing train/test leakage.
        **No Evaluation**:  Wrap a group identifier in the `"no_eval(ID)"` string
        (where `ID` is the group identifier) for samples that should not be included
        in the outer loop evaluation. These samples are removed in-between prediction
        and evaluation. E.g. used for samples that were used for calculating batch
        correction during prediction and should thus not be evaluated on.
        NOTE: Max. one of the "train_only()" and "no_eval()" wrappers can
        be used for the same sample (train-only already leads to no evaluation).
    positive : str, int or `None`
        Value that the positive class has in `y`.
        Required when `task` is 'binary_classification' and ignored otherwise.
    y_labels : Dict[int, str]
        Dict mapping unique values in `y` to their class names.
        When `None`, the names in `y` are used.
        Only used for classification tasks.
    k_outer : int or None (when `outer_split` is specified)
        Number of folds in the outer cross-validation.
        Ignored when `outer_split` is specified.
    k_inner : int or None (when `outer_split` is specified)
        Number of folds in the inner cross-validation.
        When `None`, the present splits from `outer_split` is used.
    outer_split : list or `numpy.ndarray` or dict with list/array per repetition
        Pre-specified fold identifiers. Can be either integers or strings.
        Useful in cross-dataset-validation (aka. leave-one-dataset-out) where each fold
        is a separate dataset.
        When a dict, keys are "names" of repetitions and values are the lists/arrays
        of splits to use (one per repetitions).
        **Train Only**: Wrap a split identifier in the `"train_only(ID)"` string
        (where `ID` is the split identifier) for samples that should always be in
        the training set (i.e. never be in the validation fold).
        This "train only" setting is used in both the outer and inner cross-validation.
        When `groups` is specified, indices that are specified as "train only" in
        either are used only during training.
    split_weights
        Dictionary with loss weights per outer split ID (i.e. `split id -> weight`).
        Ignored when `outer_split` is not specified.
        NOTE: The split IDs must match those in `outer_split`, including
        `train_only()` wrapping.
    eval_by_split : bool
        Whether to evaluate by splits instead of with all predictions at once.
        When paired with `outer_split`, the output will also have summaries
        for each split. E.g. when each part is a dataset and we wish to
        have scores for each separately.
    aggregate_by_groups : bool
        Whether to aggregate predictions per group, prior to evaluation.
        For regression predictions and predicted probabilities,
        the values are averaged per group.
        For class predictions, we use majority vote. In ties, the
        lowest class index is selected. TODO: Perhaps this should be settable?
    weight_loss_by_groups : bool
        Whether to weight samples by the group size in training loss.
        Each sample in a group gets the weight `1 / group_size`.
        Passed to model's `.fit(sample_weight=)` method.
    weight_loss_by_class : bool
        Whether to weight the loss by the inverse class frequencies.
        The weights are calculated per outer training set and used in both the outer
        and inner cross-validation. Since it is not calculated per inner
        training set, the weighting may not be completely balanced during inner
        cross-validation.
        NOTE: When not enabling `weight_per_split` as well,
        it may be preferable to instead use the balancing
        integrated directly in the scikit-learn models.
        NOTE: Take care not to enable class balancing here AND in the model
        (scikit-learn models has this option)!
    weight_per_split : bool
        Whether to perform the loss weighting separately per outer split.
        Affects both class- and group-based weighting.
        E.g. when each outer split is a dataset with some set of biases
        that shouldn't be ascribed to the majority class. Instead of weighting based
        on the overall class imbalance, we fix the imbalance within each dataset.
        NOTE: May not be meaningful when `outer_split` is not specified.
    tmp_path : str, `pathlib.Path` or `None`
        Path to store grid search results from inner loops temporarily.
        NOTE The results from the inner loops will only be returned when this is specified.
        NOTE Unless interrupted, the (uniquely named) subfolders created in this folder are deleted again.
    process_predictions_fn : callable
        A function for processing the predictions from `model` prior to evaluation.
        E.g. shape flattening, applying softmax or truncation.
        The function should take a single argument (the predictions)
        and return a single output (the processed predictions).
        Skorch models use their `.predict_nonlinearity()` method for this by default.
    inner_metric : str or List[str]
        Metric(s) to select inner models from.
        Passed to `sklearn.model_selection.GridSearchCV(scoring=)`.
        Can be a single string or a list of strings. When multiple strings
        are given, the *first* is used to select the best model from the inner loop.
        See metric names at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        Tip: Choose a metric that handles class imbalances decently.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression'}.
    transformers: list of tuples with (<name>, <transformer>)
        Transformation steps to add to start of the pipeline.
        Should handle 2- and/or 3-dimensional arrays. See `DimTransformerWrapper` for this.
        Can be built with `PipelineDesigner`.
    train_test_transformers: str
        Names of transformers with the `training_mode: bool` argument
        to disable during testing.
    add_channel_dim : bool
        Whether to add a singleton dimension to `x` at the second index
        for models that require a channel dimension (such as convolutional neural networks).
    add_y_singleton_dim : bool
        Whether to add a singleton dimension to `y` at the second index
        for models that require a 2D targets array.
        Only applied when `y` is a 1D array.
    rm_missing : bool
        Whether to remove missing data (i.e. `numpy.nan`).
    reps : int
        How many repetitions of the outer cross-validation to perform.
        When `outer_split` is specified, the same outer folds are used in all repetitions.
    num_jobs : int
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors.
    seed : int
        Random state used for splitting data into folds.
        Each repetition will use `seed`+repetition as seed.
    grid_error_score : int, float or str
        Error score passed to `GridSearch`. Use `"raise"` to get errors
        from inner-loop convergence warnings.
        See `GridSearch` from scikit-learn for more details.
    cv_error_score : int, float or str
        Error score passed to `cross_validate`. Use `"raise"` to get errors.
        See `cross_validate` from scikit-learn for more details.
    store_attributes : List[AttributeToDataFrameExtractor]
        List of `AttributeToDataFrameExtractor` instances
        for extracting attributes from the `.best_estimator_`
        and converting them to `pandas.DataFrame`s to store.
    identifier_cols_dict : Dict[str, str]
        Dict mapping colname -> string to add to the results
        and one-vs-all data frames *after* repetitions are concatenated.
        TODO Explain this better!
    eval_idx_colname : str
        Name of evaluation index column in the final results data frame.
    messenger : `utipy.Messenger` or `None`
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g. `print`)
        and potential indentation.

    Returns
    -------
    dict
        ``Evaluation`` : dict
            ``Summary`` : dict or `None`
                Summarized evaluation scores and potential confusion matrices.
                NOTE: `None` when reps == 1.
                When `outer_split` is specified and `eval_by_split` is enabled,
                it includes summaries per split group.
            ``Evaluations`` : dict
                Evaluation scores for each repetition along with potential
                confusion matrices, ROC curves and One-vs-All evaluations.
        ``Outer Predictions`` : List of `numpy.ndarray`s
            Predictions from the outer cross-validation.
        ``Outer Indices`` : List of `numpy.ndarray`s
            Indices in ``x`` that the ``Outer Predictions`` refer to.
            Useful when not all data in ``x`` was used in testing.
        ``Outer Splits`` : List of integers
            Indices that determined the splits during the outer cross-validation.
            Can be used to identify the folds for each of the predictions.
        ``Outer Train Scores`` : List of dicts with `numpy.ndarray`s TODO check this is true
            The score arrays for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
        ``Inner Results`` : List of lists with dicts
            GridSearch results from inner cross-validation.
            The outer list has the repetitions, the inner list has results from
            each inner cross-validation (one per outer fold).
            NOTE: `None` when `tmp_path` is not specified.
        ``Best Coefficients`` : List of data frames with
            the best coefficients found by GridSearchCV in the inner cross-validation.
            NOTE: `None` when `tmp_path` is not specified.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Start by copying the input data
    # to avoid altering them in parent scope
    x = x.copy()
    y = y.copy()
    if confounders is not None:
        confounders = confounders.copy()

    if grid is None:
        grid = {}

    # Convert specified outer folds to a numpy array
    outer_split_names = None
    if outer_split is not None:
        if isinstance(outer_split, dict):
            outer_split_dict = {
                key: np.asarray([str(s) for s in split_], dtype=object)
                for key, split_ in outer_split.items()
            }
            assert (
                len(outer_split_dict) == reps
            ), "When `outer_split` is a dict, it must contain one entry per `reps`."
            unique_split_names = [set(split_) for split_ in outer_split_dict.values()]
            assert _all_sets_equal(unique_split_names), (
                "When `outer_split` is a dict, all split arrays "
                "(dict values) must have the same split names."
            )
            outer_split = np.vstack(list(outer_split_dict.values())).transpose()
            outer_split_names = [str(s) for s in outer_split_dict.keys()]
        else:
            outer_split = np.asarray([str(s) for s in outer_split], dtype=object)

        if split_weights is not None:
            # Ensure keys are strings
            # TODO: Should this be settable per repetition when `outer_split` is a dict?
            split_weights = {
                str(split_id): weight for split_id, weight in split_weights.items()
            }

    # Convert groups to strings to make it easier
    # to check against (avoids bugs)
    if groups is not None:
        groups = np.array([str(g) for g in groups], dtype=object)

    # Check some of the arguments
    _check_args_for_nested_cv(
        x=x,
        y=y,
        confounders=confounders,
        groups=groups,
        grid=grid,
        y_labels=y_labels,
        k_outer=k_outer,
        k_inner=k_inner,
        outer_split=outer_split,
        split_weights=split_weights,
        inner_metric=inner_metric,
        eval_by_split=eval_by_split,
        aggregate_by_groups=aggregate_by_groups,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        task=task,
        transformers=transformers,
        train_test_transformers=train_test_transformers,
        add_channel_dim=add_channel_dim,
        add_y_singleton_dim=add_y_singleton_dim,
        rm_missing=rm_missing,
        reps=reps,
        num_jobs=num_jobs,
        seed=seed,
        messenger=messenger,
    )

    # Remove values/rows with `numpy.nan` in them
    # For string arrays, we check for the string 'nan'
    if rm_missing:
        messenger("Removing values/rows with `numpy.nan` in them")
        (y, x, confounders, outer_split, groups), removed_indices = remove_missing_data(
            [y, x, confounders, outer_split, groups]
        )
        messenger(f"Removed ({len(removed_indices)}) values/rows", indent=2)

    # Ensure `x` is 2/3D
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    assert x.ndim in [2, 3]

    if confounders is not None and confounders.ndim == 1:
        confounders = np.expand_dims(confounders, 1)
        assert confounders.ndim in [2, 3]

    # Add singleton dimension to targets (`y`)
    # Needed for neural nets but doesn't always work with sklearn models
    if add_y_singleton_dim and y.ndim == 1:
        y = np.expand_dims(y, 1)
    assert y.ndim in [1, 2]

    # Extract train-only group IDs
    # NOTE: `groups` may become 2D from this (when split-weighting)
    groups, outer_split = detect_train_only(
        y=y,
        groups=groups,
        split=outer_split,
        weight_per_split=weight_per_split,
        messenger=messenger,
    )

    groups, no_eval_indices = detect_no_evaluation(
        groups=groups,
        messenger=messenger,
    )

    # Add splits to groups when necessary
    groups, outer_split = add_split_to_groups(
        groups=groups,
        split=outer_split,
        weight_per_split=weight_per_split,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        k_inner=k_inner,
    )

    # Check the model can be used with the pipeline
    required_methods = ["fit", "predict"]
    for method in required_methods:
        if not hasattr(model, method):
            raise ValueError(f"`model` does not have the required `{method}` method.")

    # Create pipeline
    pipe = create_pipeline(
        model=model,
        # TODO We could have models preferring non-flattened input in the future
        flatten_feature_sets=False,  # TODO How should this be handled?
        add_channel_dim=add_channel_dim,
        transformers=transformers,
        train_test_transformers=train_test_transformers,
        # num_confounders=confounders.shape[-1] if confounders is not None else 0,
        weight_loss_by_groups=False,  # Taken care of in GridSearch
        weight_loss_by_class=False,  # Taken care of in GridSearch
        weight_per_split=False,  # Taken care of in GridSearch
        split_weights=None,  # Taken are of in GridSearch
    )

    messenger(pipe)

    # Create a partial grid where only the cv and seed args need to be set

    messenger("Preparing inner cross-validation grid search")

    # Ensure grid names point to their pipeline step
    pipeline_keys = list(pipe.named_steps.keys())
    for key in grid.keys():
        if "__" not in key or key.split("__")[0] not in pipeline_keys:
            messenger(f"Pipeline keys for debugging:\n{pipeline_keys}")
            raise ValueError(
                f"Grid keys must be prefixed by either 'model__' or "
                f"'<transformer name>__' but got {key}."
            )

    # Create function that returns a NestableGridSearchCV object
    partial_grid = partial(
        NestableGridSearchCV,
        estimator=pipe,
        param_grid=grid,
        scoring=inner_metric,
        n_jobs=num_jobs,
        refit=refit,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        split_weights=split_weights,
        messenger=messenger,
        store_attributes=store_attributes,
        error_score=grid_error_score,
    )

    # Prepare list of paths to save inner results in temporarily
    inner_results_paths, tmp_subdir_path = _prepare_tmp_folders(
        tmp_path=tmp_path, reps=reps, messenger=messenger
    )

    # Add the confounders to the data
    # NOTE: You can use the `MeanDuringTest` transformer to ensure that
    # they get the average column value during testing
    # to remove their information from the test predictions
    if confounders is not None:
        x = np.concatenate([confounders, x], axis=-1)
        messenger(
            f"Concatenated confounders and features. New shape: {x.shape}. "
            f"(0:{confounders.shape[-1]} = confounders; {confounders.shape[-1]}: {x.shape[-1]} = features)."
        )

    # We need to delete the temporary directories even
    # when the code fails or is interrupted
    try:
        # Cross-validate with n repetitions
        messenger("Running cross-validations")
        cv_outputs_list = [
            _single_nested_cross_validate(
                x=x,
                y=y,
                groups=(
                    groups[:, rep]
                    if groups is not None and groups.ndim == 2
                    else groups
                ),
                partial_grid=partial_grid,
                k_outer=k_outer,
                k_inner=k_inner,
                outer_split=(
                    outer_split[:, rep]
                    if outer_split is not None and outer_split.ndim == 2
                    else outer_split
                ),
                task=task,
                num_jobs=num_jobs,
                seed=seed + rep if seed is not None else None,
                tmp_path=tmp_path_for_rep,
                verbose=messenger.verbose,
                error_score=cv_error_score,
            )
            for rep, tmp_path_for_rep in zip(range(reps), inner_results_paths)
        ]

        # Load and format inner results
        inner_results, best_coeffs, custom_attributes = _get_inner_results(
            inner_results_paths=inner_results_paths,
            tmp_path=tmp_path,
            store_attributes=store_attributes,
            messenger=messenger,
        )

        # Delete created temporary folders
        _del_tmp_subdir(
            tmp_path=tmp_path, tmp_subdir_path=tmp_subdir_path, messenger=messenger
        )

    except Exception as e:
        # Delete created temporary folders
        _del_tmp_subdir(
            tmp_path=tmp_path, tmp_subdir_path=tmp_subdir_path, messenger=messenger
        )
        messenger(f"Cross-validation failed: {e}")
        raise

    predictions_list = [res["predictions"] for res in cv_outputs_list]
    split_indices_list = [res["split_indices"] for res in cv_outputs_list]
    train_scores_list = [res["train_score"] for res in cv_outputs_list]
    test_indices_list = [res["test_indices"] for res in cv_outputs_list]
    warnings_list = [res.get("warnings", []) for res in cv_outputs_list]

    if no_eval_indices:
        # Find out which elements to remove before evaluation
        no_eval_indices_to_remove = [
            np.argwhere(np.isin(test_indices, no_eval_indices)).flatten()
            for test_indices in test_indices_list
        ]
        # Remove test indices for no-evaluation samples
        test_indices_list = [
            np.delete(test_indices, rm_indices, axis=0)
            for test_indices, rm_indices in zip(
                test_indices_list, no_eval_indices_to_remove
            )
        ]
        # Remove predictions for no-evaluation samples
        predictions_list = [
            np.delete(predictions, rm_indices, axis=0)
            for predictions, rm_indices in zip(
                predictions_list, no_eval_indices_to_remove
            )
        ]
        if split_indices_list[0] is not None:
            # Remove predictions for no-evaluation samples
            split_indices_list = [
                np.delete(split_indices, rm_indices, axis=0)
                for split_indices, rm_indices in zip(
                    split_indices_list, no_eval_indices_to_remove
                )
            ]

    # Process the predictions
    # NOTE: Seems it cannot be added to the pipeline
    # as the model must be the last step in it
    # Skorch models can use `predict_nonlinearity` method instead
    if process_predictions_fn is not None:
        predictions_list = [process_predictions_fn(preds) for preds in predictions_list]

    # Remove potential singleton dimension from binary predictions
    if task == "binary_classification":
        predictions_list = [
            BinaryPreparer.prepare_probabilities(probabilities=probs)
            for probs in predictions_list
        ]

    # Get fold names instead of the index
    # when `outer_split` was specified
    if outer_split is not None:
        # Get names for each fold
        split_index_names = dict(
            enumerate(specified_folds_iterator(folds=outer_split, yield_names=True))
        )

        # Get max name length for specifying array size
        max_name_length = max([len(str(name)) for name in split_index_names.values()])

        # Convert fold indices to fold names
        new_splits_list = []
        for splits in split_indices_list:
            new_splits = splits.astype(f"U{max_name_length}")
            for idx, name in split_index_names.items():
                new_splits[np.array(splits) == np.array(idx)] = str(name)
            new_splits_list.append(new_splits)
        split_indices_list = new_splits_list

    test_targets_list = [y[test_indices] for test_indices in test_indices_list]
    test_groups_list = None
    if groups is not None:
        if groups.ndim == 1:
            test_groups_list = [
                groups[test_indices] for test_indices in test_indices_list
            ]
        else:
            test_groups_list = [
                groups[test_indices, group_col]
                for group_col, test_indices in enumerate(test_indices_list)
            ]

    # Evaluate the repetitions and combine evaluations (when possible)
    evaluations = evaluate_repetitions(
        predictions_list=predictions_list,
        targets_list=test_targets_list,
        task=task,
        groups_list=test_groups_list if aggregate_by_groups else None,
        splits_list=split_indices_list if eval_by_split else None,
        summarize_splits=eval_by_split and outer_split is not None,
        summarize_splits_allow_differing_members=(
            outer_split.ndim == 2 if outer_split is not None else False
        ),
        positive=positive,
        target_labels=y_labels,
        identifier_cols_dict=identifier_cols_dict,
        eval_names=outer_split_names,
        eval_idx_colname=eval_idx_colname,
        messenger=messenger,
    )

    return {
        "Evaluation": evaluations,
        "Outer Predictions": predictions_list,
        "Outer Targets": test_targets_list,
        "Outer Groups": test_groups_list,
        "Outer Indices": test_indices_list,
        "Outer Splits": split_indices_list,
        "Outer Train Scores": train_scores_list,
        "Inner Results": inner_results,
        "Best Coefficients": best_coeffs,
        "Custom Attributes": custom_attributes,
        "Warnings": warnings_list,
    }


def _single_nested_cross_validate(
    x,
    y,
    groups,
    partial_grid,
    k_outer,
    k_inner,
    outer_split,
    task,
    num_jobs,
    seed,
    tmp_path,
    verbose,
    error_score,
):
    """
    Run a single nested cross-validation.

    Parameters
    ----------
    x : 2/3D `numpy.ndarray`
        The data. Shape: (# samples, # feature sets (optional), # features).
    y : 1D `numpy.ndarray`
        Targets (one per sample).
    groups : 1D `numpy.ndarray`
        Group IDs to ensure each group is always in the same fold.
        Note: `outer_split` takes precedence over this! Overlapping
        IDs across outer splits will not be respected.
    partial_grid : callable
        Wrapped `GridSearchCV` that requires the `cv`, `seed` and `save_cv_results_path` arguments.
    k_outer : int or None (when `outer_split` is specified)
        Number of folds in the outer cross-validation.
        Ignored when `outer_split` is specified.
    k_inner : int or None (when `outer_split` is specified)
        Number of folds in the inner cross-validation.
        When `None`, the present splits from `outer_split` is used.
    outer_split : `numpy.ndarray`
        Pre-specified fold identifiers.
        Can be either integers or strings.
        **Train Only**: Wrap a split identifier in the `"train_only(ID)"` string
        (where `ID` is the split identifier) for samples that should always be in
        the training set (i.e. never be in the validation fold).
        When `groups` is specified, indices that are specified as "train only" in
        either are used only during training.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression'}.
    num_jobs : int
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors.
    seed : int
        Random state used for splitting data into folds.

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
        A dict of arrays containing the score/time arrays for each scorer is
        returned. The possible keys for this ``dict`` are:
            ``test_score``
                The score array for test scores on each cv split.
                Suffix ``_score`` in ``test_score`` changes to a specific
                metric like ``test_r2`` or ``test_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``train_score``
                The score array for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``)
            ``predictions``
                Outer cross-validation predictions for the dataset.
                This is available only if ``return_predictions`` parameter
                is set to ``predict`` or ``predict_proba``.
    """
    # Declare the inner and outer cross-validation

    if outer_split is None:
        outer_folder = KFolder(
            n_splits=k_outer,
            stratify="classification" in task,
            by_group=groups is not None,
            shuffle=True,
            random_state=seed,
        )
    else:
        outer_folder = specified_folds_iterator(folds=outer_split)

    if k_inner is None:
        assert outer_split is not None
        inner_folder = GroupSpecifiedFolder(
            n_splits=sum([int("train_only" not in s) for s in np.unique(outer_split)])
            - 1
        )

    else:
        inner_folder = KFolder(
            n_splits=k_inner,
            stratify="classification" in task,
            by_group=groups is not None,
            shuffle=True,
            random_state=seed,
        )

    save_cv_results_path = None
    if tmp_path is not None:
        save_cv_results_path = tmp_path / "inner_cv_results.csv"
    estimator = partial_grid(
        cv=inner_folder, seed=seed, save_cv_results_path=save_cv_results_path
    )

    # Cross-validate
    # Returns dict with scores and predictions
    predict_method_name = (
        "predict_proba"
        if "classification" in task and hasattr(estimator, "predict_proba")
        else "predict"
    )

    # Convert group indicators to strings
    # Should create a string array instead of object array?
    if groups is not None:
        groups = np.array([str(g) for g in groups])

    return cross_validate_with_predictions(
        estimator=estimator,
        X=x,
        y=y,
        cv=outer_folder,
        n_jobs=num_jobs,
        groups=groups,
        scoring=None,
        allow_partial_test=outer_split is not None
        or (groups is not None and any(["train_only" in str(g) for g in groups])),
        pass_groups_to_estimator=True,
        return_train_score=True,
        return_estimator=False,
        return_predictions=predict_method_name,
        verbose=int(verbose),
        error_score=error_score,
    )


def _prepare_tmp_folders(tmp_path, reps, messenger):
    """
    Create a uniquely named directory in `tmp_path` with
    one subdirectory per repetition (reps).
    """
    # Prepare list of paths to save inner results in temporarily
    inner_results_paths = [None] * reps
    tmp_subdir_path = None
    if tmp_path is not None:
        messenger(
            "Preparing paths for temporary directories for "
            "saving inner cross-validation results"
        )
        assert isinstance(tmp_path, (str, pathlib.Path))
        tmp_path = pathlib.Path(tmp_path)
        unique_dir_name = _create_unique_name_within_folder(
            path=tmp_path, prefix="nested_cv"
        )
        # This directory (`tmp_subdir_path`) will contain all the repetition directories,
        # making it quick to remove all the temporary files and avoid
        # mingling with other stuff in the `tmp_path` folder
        tmp_subdir_path = tmp_path / unique_dir_name
        inner_results_paths = [tmp_subdir_path / f"rep_{rep}" for rep in range(reps)]
        for _path in inner_results_paths:
            mk_dir(
                path=_path,
                arg_name="`tmp_path` subdirectory",
                raise_on_exists=True,  # Should not exist!
                messenger=messenger,
            )
    return inner_results_paths, tmp_subdir_path


def _create_unique_name_within_folder(
    path, prefix="tmp_", num_chars=16, max_attempts=20
):
    """
    Create a unique name suitable for directories and files,
    made up of `num_chars` alphanumeric characters.

    Uniqueness: There are no files or directories in `path` starting with the name.

    The random generation is *not* affected by an external random state (seed).
    """
    assert isinstance(path, (str, pathlib.Path))
    assert isinstance(prefix, str)

    path = pathlib.Path(path)

    # Create a unique random generator
    # which is not affected by seeds elsewhere
    randomgen = random.Random()

    def _is_unique(name):
        """
        Check if a file/directory with this name exists
        (fully or as prefix) in the path.
        """
        return not ((path / new_name).exists() or list(path.glob(f"{name}*")))

    # Create name (e.g. of directory)
    # Which is not available in the path
    num_attempts = 0
    while num_attempts < max_attempts:
        random_alphanumeric = "".join(
            randomgen.choices(string.ascii_letters + string.digits, k=num_chars)
        )
        new_name = prefix + f"_{random_alphanumeric}"
        num_attempts += 1
        if _is_unique(new_name):
            break
        # Avoid returning non-unique name
        # in case we reach max attempts
        new_name = None

    if new_name is None:
        warnings.warn(f"Could not create a unique name in {max_attempts} attempts.")

    return new_name


def _check_args_for_nested_cv(
    x: np.ndarray,
    y: np.ndarray,
    confounders: Optional[np.ndarray],
    groups: Optional[np.ndarray],
    grid: Dict[str, List[any]],
    y_labels: Optional[dict],
    k_outer: Optional[int],
    k_inner: Optional[int],
    outer_split: Optional[Union[List[Union[int, str]], np.ndarray]],
    split_weights: Optional[Dict[str, float]],
    inner_metric: Union[str, List[str]],
    eval_by_split: bool,
    aggregate_by_groups: bool,
    weight_loss_by_groups: bool,
    weight_loss_by_class: bool,
    weight_per_split: bool,
    task: str,
    transformers: Optional[List[Tuple[str, BaseEstimator]]],
    train_test_transformers,
    add_channel_dim: bool,
    add_y_singleton_dim: bool,
    rm_missing: bool,
    reps: int,
    num_jobs: int,
    seed: int,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Check some of the arguments for `nested_cross_validate()`.
    """

    assert task in ["binary_classification", "multiclass_classification", "regression"]

    if not (
        isinstance(inner_metric, str)
        or (
            isinstance(inner_metric, list)
            and inner_metric  # Not empty
            and isinstance(inner_metric[0], str)
        )
    ):
        raise ValueError("`inner_metric` must be either a string or a list of strings.")

    if not grid:
        raise ValueError(
            "Grid was empty. Does not support models with self-contained inner cross-validation."
        )

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)
    if confounders is not None:
        assert isinstance(confounders, np.ndarray)
        assert len(confounders) == len(x)
    if groups is not None:
        assert isinstance(groups, np.ndarray)
        assert len(x) == len(groups)

    assert y_labels is None or isinstance(y_labels, dict)
    assert (k_outer is None and outer_split is not None) or isinstance(k_outer, int)
    assert (k_inner is None and outer_split is not None) or isinstance(k_inner, int)
    assert isinstance(reps, int) and reps > 0
    assert isinstance(num_jobs, int)
    assert isinstance(seed, int)
    assert isinstance(add_channel_dim, bool)
    assert isinstance(add_y_singleton_dim, bool)
    assert isinstance(rm_missing, bool)
    assert isinstance(eval_by_split, bool)
    assert isinstance(aggregate_by_groups, bool)
    assert isinstance(weight_loss_by_groups, bool)
    assert isinstance(weight_loss_by_class, bool)
    assert isinstance(weight_per_split, bool)

    if weight_per_split and outer_split is None:
        # If we have train-only data in single-CV it could be useful
        raise warnings.warn(
            "Enabling `weight_per_split` may not be meaningful when `outer_split` is `None`."
        )

    if outer_split is not None:
        assert isinstance(outer_split, (list, np.ndarray))
        if isinstance(outer_split, np.ndarray):
            # In case a string array is
            # instead an array of objects
            # We can recreate the array for testing purposes
            outer_split = outer_split.tolist()
        outer_split = np.asarray(outer_split)

        assert outer_split.ndim in [
            1,
            2,
        ], (  # will be 2d when dict
            f"`outer_split` must be 1-dimensional but had shape: {outer_split.shape}"
        )
        if not (
            np.issubdtype(outer_split.dtype, np.number)
            or np.issubdtype(outer_split.dtype, np.str_)
        ):
            raise TypeError(
                f"`outer_split` should contain either numbers or strings. "
                f"Got dtype {outer_split.dtype}."
            )
        # if reps > 1:
        #     msg = (
        #         "`outer_split` was specified and `reps` was > 1. "
        #         "This might only be meaningful for non-deterministic model types. "
        #         "Inner folds are still stochastic."
        #     )
        #     messenger(msg)
        #     warnings.warn(msg)

        if split_weights is not None:
            assert isinstance(split_weights, dict)
            assert len(set(split_weights.keys()).intersection(set(outer_split))) == len(
                split_weights
            ), "When `split_weights` is defined, all split IDs must have a weight."
            assert not len(
                set(split_weights.keys()).difference(set(outer_split))
            ), "When `split_weights` is defined, it can only contain split IDs also present in `outer_split`."

    if transformers is not None:
        assert isinstance(transformers, list)
        for i, transformer in enumerate(transformers):
            assert isinstance(
                transformer, tuple
            ), "`transformers` must be a list of tuples with (name, transformer)."
            assert isinstance(transformer[0], str), (
                f"The first element in a transformer tuple must be a string. "
                f"Tuple {i}'s first element had type {type(transformer[0])}."
            )
            assert isinstance(transformer[1], BaseEstimator), (
                f"The second element in a transformer tuple must inherit from `BaseEstimator`. "
                f"Tuple {i}'s second element had type {type(transformer[1])}."
            )

    if train_test_transformers:
        assert isinstance(train_test_transformers[0], str), (
            "`train_test_transformers` must contain the names (str) of transformers. "
            f"Got type: {type(train_test_transformers[0])}"
        )


def _get_inner_results(
    inner_results_paths, tmp_path, store_attributes, messenger
) -> Tuple[List[pd.DataFrame], List[Optional[pd.DataFrame]]]:
    # Handle grid search cv results dicts
    inner_results = None
    if tmp_path is not None:
        # Load and combine gridsearch cv results dicts
        messenger("Loading inner-cv results from temporary directories")

        # Get headers first
        # Each header is saved in its own file
        def get_header(path):
            with open(str(path)) as f:
                return f.readline().split(",")

        # Read headers
        # A header is a list of strings (column names)
        inner_results_headers = [
            get_header(path / "inner_cv_results.header.csv")
            for path in inner_results_paths
        ]

        # Read inner results
        # and set header (`names`)
        inner_results = [
            pd.read_csv(path / "inner_cv_results.csv", header=None, names=col_names)
            for path, col_names in zip(inner_results_paths, inner_results_headers)
        ]

        # Read best coefficients
        try:
            best_coefficients = [
                pd.read_csv(
                    path / "inner_cv_results.best_coefficients.csv",
                    header=None,
                )
                for path in inner_results_paths
            ]
            for bcoeffs in best_coefficients:
                bcoeffs.columns = [str(i) for i in range(bcoeffs.shape[-1] - 1)] + [
                    "random_id"
                ]
        except BaseException as e:
            messenger(
                f"Failed to read best coefficients: {str(e)}", add_msg_fn=warnings.warn
            )
            available_files = [str(p) for p in inner_results_paths[0].glob("*.csv")]
            messenger(
                f"{len(available_files)} available .csv files in tmp folder: "
                f"{', '.join(available_files)}",
                add_msg_fn=warnings.warn,
            )
            # If there's no coefficients to be read, we create
            # a list of None's to loop over with zip
            best_coefficients = [None for _ in inner_results]

        custom_attributes = {}
        if store_attributes is not None:
            for extractor in store_attributes:
                try:
                    extracted_attrs = [
                        pd.read_csv(
                            path / f"inner_cv_results.{extractor.name}.csv",
                            header=None,
                        )
                        for path in inner_results_paths
                    ]
                    for extracted_attr in extracted_attrs:
                        extracted_attr.columns = [
                            str(i) for i in range(extracted_attr.shape[-1] - 1)
                        ] + ["random_id"]
                except BaseException as e:
                    messenger(
                        f"Failed to read {extractor.name}: {str(e)}",
                        add_msg_fn=warnings.warn,
                    )
                    available_files = [
                        str(p) for p in inner_results_paths[0].glob("*.csv")
                    ]
                    messenger(
                        f"{len(available_files)} available .csv files in tmp folder: "
                        f"{', '.join(available_files)}",
                        add_msg_fn=warnings.warn,
                    )
                    # If the attribute is not found, we create
                    # a list of None's to loop over with zip
                    extracted_attrs = [None for _ in inner_results]

                custom_attributes[extractor.name] = extracted_attrs

        # Change random IDs to letter IDs (AA, AB, AC, ...)
        random_ids_store = {}
        for idx, (in_res, in_coefs) in enumerate(zip(inner_results, best_coefficients)):
            if "random_id" not in in_res:
                messenger("Bad inner results data frame: ", in_res)
                with messenger.indentation(add_indent=2):
                    messenger("with column names: ", in_res.columns)
                raise RuntimeError("'random_id' was not in saved results.")
            unique_random_ids = sorted(in_res["random_id"].unique())
            letter_ids = letter_strings(n=len(unique_random_ids), upper=True)
            random_ids_store[idx] = (unique_random_ids, letter_ids)

            in_res.replace(
                {"random_id": {n: l for n, l in zip(unique_random_ids, letter_ids)}},
                inplace=True,
            )
            in_res.rename(
                columns={"random_id": "outer_split (unordered)"}, inplace=True
            )

            if in_coefs is not None:
                in_coefs.replace(
                    {
                        "random_id": {
                            n: l for n, l in zip(unique_random_ids, letter_ids)
                        }
                    },
                    inplace=True,
                )
                in_coefs.rename(
                    columns={"random_id": "outer_split (unordered)"}, inplace=True
                )
        for _, in_attr_dfs in custom_attributes.items():
            for idx, in_attr_df in enumerate(in_attr_dfs):
                unique_random_ids, letter_ids = random_ids_store[idx]

                in_attr_df.replace(
                    {
                        "random_id": {
                            n: l for n, l in zip(unique_random_ids, letter_ids)
                        }
                    },
                    inplace=True,
                )
                in_attr_df.rename(
                    columns={"random_id": "outer_split (unordered)"}, inplace=True
                )

    return inner_results, best_coefficients, custom_attributes


def _del_tmp_subdir(tmp_path, tmp_subdir_path, messenger):
    if tmp_path is not None:
        # Delete created temporary folders
        messenger("Deleting temporary directories")
        rm_dir(
            path=tmp_subdir_path,
            arg_name="`tmp_path` subdirectory",
            raise_missing=True,  # Should exist, since we created it!
            raise_not_dir=True,  # Should be a directory...
            messenger=messenger,
        )


def _all_sets_equal(sets: List[set]) -> bool:
    if not sets:
        return True
    reference_set = sets[0]
    return all(s == reference_set for s in sets)
