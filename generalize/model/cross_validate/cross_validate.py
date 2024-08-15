from typing import Callable, List, Optional, Tuple, Union, Dict
import warnings
import numpy as np
from sklearn.base import BaseEstimator
from utipy import Messenger, check_messenger

from generalize.evaluate.prepare_inputs import BinaryPreparer
from generalize.model.cross_validate.sklearn_cross_validate import (
    cross_validate_with_predictions,
)
from generalize.model.cross_validate.kfolder import KFolder, specified_folds_iterator
from generalize.evaluate.evaluate_repetitions import evaluate_repetitions
from generalize.utils.missing_data import remove_missing_data
from generalize.model.pipeline.pipelines import create_pipeline
from generalize.model.utils import detect_train_only

# TODO Consider order of the arguments
# TODO Should split be allowed to have a split per repetition?


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    groups: Optional[np.ndarray] = None,
    positive: Optional[Union[str, int]] = None,
    y_labels: Optional[dict] = None,
    k: int = 10,
    split: Optional[Union[List[Union[int, str]], np.ndarray]] = None,
    split_weights: Optional[Dict[str, float]] = None,
    eval_by_split: bool = False,
    aggregate_by_groups: bool = False,
    weight_loss_by_groups: bool = False,
    weight_loss_by_class: bool = False,
    weight_per_split: bool = False,
    process_predictions_fn: Callable = None,
    task: str = "binary_classification",
    transformers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    train_test_transformers: List[str] = [],
    add_channel_dim: bool = False,
    add_y_singleton_dim: bool = False,
    rm_missing: bool = False,
    reps: int = 1,
    num_jobs: int = 1,
    seed: int = 1,
    cv_error_score: Union[str, int, float] = np.nan,
    identifier_cols_dict: dict = None,
    eval_idx_colname: str = "Repetition",
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Perform cross-validation (without hyperparameter optimization, see `nested_cross_validate()` for that).

    Specify data points to only use as training data in both the inner and outer cross-validation
    via `split` and/or `groups`.

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
    groups : 1D `numpy.ndarray` or `None`
        An array of group IDs (one for each of the data points).
        (Almost always) ensures that all data points with the same
        ID are put in the same fold during cross-validation.
        WARNING: When `split` is specified, it has precedence. IDs that
        are present in multiple splits are NOT respected as a single
        entity.
        See `aggregate_by_groups` for prediction aggregation during evaluation.
        **Train Only**: Wrap a group identifier in the `"train_only(ID)"` string
        (where `ID` is the group identifier) for samples that should only be used
        for training. When `split` is specified, indices that are
        specified as "train only" in either are used only during training!
        NOTE: Beware of introducing train/test leakage.
    positive : str, int or `None`
        Value that the positive class has in `y`.
        Required when `task` is 'binary_classification' and ignored otherwise.
    y_labels : Dict[int, str]
        Dict mapping unique values in `y` to their class names.
        When `None`, the names in `y` are used.
        Only used for classification tasks.
    k : int
        Number of folds in the cross-validation.
        Ignored when `split` is specified.
    split : list or `numpy.ndarray`
        Pre-specified fold identifiers. Can be either integers or strings.
        Useful in cross-dataset-validation (aka. leave-one-dataset-out) where each fold
        is a separate dataset.
        **Train Only**: Wrap a split identifier in the `"train_only(ID)"` string
        (where `ID` is the split identifier) for samples that should always be in
        the training set (i.e. never be in the validation fold).
        When `groups` is specified, indices that are specified as "train only" in
        either are used only during training.
    split_weights
        Dictionary with loss weights per split ID (i.e. `split id -> weight`).
        Ignored when `split` is not specified.
        NOTE: The split IDs must match those in `outer_split`, including
        `train_only()` wrapping.
    eval_by_split : bool
        Whether to evaluate by splits instead of with all predictions at once.
        When paired with `split`, the output will also have summaries
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
        The weights are calculated per training set.
        NOTE: Take care not to enable class balancing here AND in the model
        (scikit-learn models has this option)!
    weight_per_split : bool
        Whether to perform the loss weighting separately per split.
        Affects both class- and group-based weighting.
        E.g. when each fold is a dataset with some set of biases
        that shouldn't be ascribed to the majority class. Instead of weighting based
        on the overall class imbalance, we fix the imbalance within each dataset.
        NOTE: May not be meaningful when `split` is not specified.
    process_predictions_fn : callable
        A function for processing the predictions from `model` prior to evaluation.
        E.g. shape flattening, applying softmax or truncation.
        The function should take a single argument (the predictions)
        and return a single output (the processed predictions).
        Skorch models use their `.predict_nonlinearity()` method for this by default.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression'}.
    transformers: list of tuples with (<name>, <transformer>)
        Transformation steps to add to the first part of the pipeline.
        Should handle 2- and/or 3-dimensional arrays. See `DimTransformerWrapper` for this.
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
        How many repetitions of the cross-validation to perform.
        When `split` is specified, the same folds are used in all repetitions.
    num_jobs : int
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors.
    seed : int
        Random state used for splitting data into folds.
        Each repetition will use `seed`+repetition as seed.
    cv_error_score : int, float or str
        Error score passed to `cross_validate`. Use `"raise"` to get errors.
        See `cross_validate` from scikit-learn for more details.
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
                When `split` is specified and `eval_by_split` is enabled,
                it includes summaries per split group.
            ``Evaluations`` : dict
                Evaluation scores for each repetition along with potential
                confusion matrices, ROC curves and One-vs-All evaluations.
        ``Predictions`` : List of `numpy.ndarray`s
            Predictions from the cross-validation.
        ``Indices`` : List of `numpy.ndarray`s
            Indices in ``x`` that the ``Predictions`` refer to.
            Useful when not all data in ``x`` was used in testing.
        ``Splits`` : List of integers
            Indices that determined the splits during the cross-validation.
            Can be used to identify the folds for each of the predictions.
        ``Train Scores`` : List of dicts with `numpy.ndarray`s TODO check this is true
            The score arrays for train scores on each cv split.
                Suffix ``_score`` in ``train_score`` changes to a specific
                metric like ``train_r2`` or ``train_auc`` if there are
                multiple scoring metrics in the scoring parameter.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Start by copying the input data
    # to avoid altering them in parent scope
    x = x.copy()
    y = y.copy()

    # Convert specified folds to a numpy array
    if split is not None:
        split = np.asarray([str(s) for s in split], dtype=object)

        if split_weights is not None:
            # Ensure keys are strings
            split_weights = {
                str(split_id): weight for split_id, weight in split_weights.items()
            }

    # Convert groups to strings to make it easier
    # to check against (avoids bugs)
    if groups is not None:
        groups = np.array([str(g) for g in groups], dtype=object)

    # Check some of the arguments
    _check_args_for_cv(
        x=x,
        y=y,
        groups=groups,
        y_labels=y_labels,
        k=k,
        split=split,
        split_weights=split_weights,
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
        (y, x, split, groups), removed_indices = remove_missing_data(
            [y, x, split, groups]
        )
        messenger(f"Removed ({len(removed_indices)}) values/rows", indent=2)

    # Ensure `x` is 2/3D
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    assert x.ndim in [2, 3]

    # Add singleton dimension to targets (`y`)
    # Needed for neural nets but doesn't always work with sklearn models
    if add_y_singleton_dim and y.ndim == 1:
        y = np.expand_dims(y, 1)
    assert y.ndim in [1, 2]

    # Extract train-only group IDs
    groups, split = detect_train_only(
        y=y,
        groups=groups,
        split=split,
        weight_per_split=weight_per_split,
        messenger=messenger,
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
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        split_weights=split_weights,
    )

    messenger(pipe)

    # We need to delete the temporary directories even
    # when the code fails or is interrupted
    try:
        # Cross-validate with n repetitions
        messenger("Running cross-validations")
        cv_outputs_list = [
            _single_cross_validate(
                x=x,
                y=y,
                estimator=pipe,
                groups=groups,
                k=k,
                split=split,
                task=task,
                num_jobs=num_jobs,
                seed=seed + rep if seed is not None else None,
                verbose=messenger.verbose,
                error_score=cv_error_score,
            )
            for rep in range(reps)
        ]
    except Exception as e:
        messenger(f"Cross-validation failed: {e}")
        raise

    predictions_list = [res["predictions"] for res in cv_outputs_list]
    split_indices_list = [res["split_indices"] for res in cv_outputs_list]
    train_scores_list = [res["train_score"] for res in cv_outputs_list]
    test_indices_list = [res["test_indices"] for res in cv_outputs_list]
    warnings_list = [res.get("warnings", []) for res in cv_outputs_list]

    # Ensure that the data indices are the same for all repetitions
    # As that really should be the case
    # Note: This is only relevant when `split` is specified
    # as it may specify not to use all samples in testing
    if split is not None:
        for i, test_indices in enumerate(test_indices_list):
            if i == 0:
                continue
            if not (test_indices == test_indices_list[0]).all():
                raise ValueError(
                    "All `test_indices` arrays in `test_indices_list` "
                    "must be the same across repetitions. "
                    "Please open a GitHub issue, as this error should not occur."
                )
    test_indices = test_indices_list[0]

    # Process the predictions
    # NOTE: Cannot be added to the pipeline
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
    # when `split` was specified
    if split is not None:
        # Get names for each fold
        split_index_names = dict(
            enumerate(specified_folds_iterator(folds=split, yield_names=True))
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

    test_targets = y[test_indices]
    test_groups = None
    if groups is not None:
        test_groups = groups[test_indices]

    # Evaluate the repetitions and combine evaluations (when possible)
    evaluations = evaluate_repetitions(
        predictions_list=predictions_list,
        targets=test_targets,
        task=task,
        groups=test_groups if aggregate_by_groups else None,
        splits_list=split_indices_list if eval_by_split else None,
        summarize_splits=eval_by_split and split is not None,
        positive=positive,
        target_labels=y_labels,
        identifier_cols_dict=identifier_cols_dict,
        eval_idx_colname=eval_idx_colname,
        messenger=messenger,
    )

    return {
        "Evaluation": evaluations,
        "Predictions": predictions_list,
        "Targets": test_targets,
        "Groups": test_groups,
        "Indices": test_indices_list,
        "Splits": split_indices_list,
        "Train Scores": train_scores_list,
        "Warnings": warnings_list,
    }


def _single_cross_validate(
    x,
    y,
    estimator,
    groups,
    k,
    split,
    task,
    num_jobs,
    seed,
    verbose,
    error_score,
    seed_callback_name: str = "fix_random_seed",
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
        Note: `split` takes precedence over this! Overlapping
        IDs across splits will not be respected.
    estimator : callable
        Estimator / pipeline to run. When pipeline, final step must be a model named "model".
    k : int
        Number of folds in the cross-validation.
        Ignored when `split` is specified.
    split : `numpy.ndarray`
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
    seed_callback_name : str
        Name of callback to set seed in.
        For skorch models only.

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
                Cross-validation predictions for the dataset.
                This is available only if ``return_predictions`` parameter
                is set to ``predict`` or ``predict_proba``.
    """

    if split is None:
        folder = KFolder(
            n_splits=k,
            stratify="classification" in task,
            by_group=groups is not None,
            shuffle=True,
            random_state=seed,
        )
    else:
        folder = specified_folds_iterator(folds=split)

    if (
        hasattr(estimator, "is_seedable")
        and estimator.is_seedable
        and seed is not None
        and has_callback(estimator.named_steps["model"], seed_callback_name)
    ):
        estimator.named_steps["model"].set_params(
            **{f"callbacks__{seed_callback_name}__seed": seed}
        )

    # Cross-validate
    # Returns dict with scores and predictions
    predict_method_name = (
        "predict_proba"
        if "classification" in task and hasattr(estimator, "predict_proba")
        else "predict"
    )

    # Convert group IDs to strings
    if groups is not None:
        groups = np.array([str(g) for g in groups])

    return cross_validate_with_predictions(
        estimator=estimator,
        X=x,
        y=y,
        cv=folder,
        n_jobs=num_jobs,
        groups=groups,
        scoring=None,
        allow_partial_test=split is not None
        or (groups is not None and any(["train_only" in str(g) for g in groups])),
        pass_groups_to_estimator=True,
        return_train_score=True,
        return_estimator=False,
        return_predictions=predict_method_name,
        verbose=int(verbose),
        error_score=error_score,
    )


def _check_args_for_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    y_labels: Optional[dict],
    k: int,
    split: Optional[Union[List[Union[int, str]], np.ndarray]],
    split_weights: Optional[Dict[str, float]],
    eval_by_split: bool,
    aggregate_by_groups: bool,
    weight_loss_by_groups: bool,
    weight_loss_by_class: bool,
    weight_per_split: bool,
    task: str,
    transformers: Optional[List[Tuple[str, BaseEstimator]]],
    train_test_transformers: List[str],
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

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)
    if groups is not None:
        assert isinstance(groups, np.ndarray)
        assert len(x) == len(groups)

    assert y_labels is None or isinstance(y_labels, dict)
    assert isinstance(k, int)
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

    if weight_per_split and split is None:
        raise ValueError(
            "Enabling `weight_per_split` is not meaningful when `split` is `None`."
        )

    if split is not None:
        assert isinstance(split, (list, np.ndarray))
        if isinstance(split, np.ndarray):
            # In case a string array is
            # instead an array of objects
            # We can recreate the array for testing purposes
            split = split.tolist()
        split = np.asarray(split)
        assert (
            split.ndim == 1
        ), f"`split` must be 1-dimensional but had shape: {split.shape}"
        if not (
            np.issubdtype(split.dtype, np.number) or np.issubdtype(split.dtype, np.str_)
        ):
            raise TypeError(
                f"`split` should contain either numbers or strings. "
                f"Got dtype {split.dtype}."
            )
        if reps > 1:
            msg = (
                "`split` was specified and `reps` was > 1. "
                "This might only be meaningful for non-deterministic model types."
            )
            messenger(msg)
            warnings.warn(msg)

        if split_weights is not None:
            assert isinstance(split_weights, dict)
            assert len(set(split_weights.keys()).intersection(set(split))) == len(
                split_weights
            ), "When `split_weights` is defined, all split IDs must have a weight."
            assert not len(
                set(split_weights.keys()).difference(set(split))
            ), "When `split_weights` is defined, it can only contain split IDs also present in `split`."

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


def has_callback(model, cb_name):
    return hasattr(model, "callbacks") and cb_name in [cb[0] for cb in model.callbacks]
