from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator
from utipy import Messenger, check_messenger

from generalize.evaluate.prepare_inputs import BinaryPreparer
from generalize.model.cross_validate.kfolder import specified_folds_iterator, KFolder
from generalize.model.cross_validate.grid_search import NestableGridSearchCV
from generalize.evaluate.evaluate_repetitions import evaluate_repetitions
from generalize.utils.missing_data import remove_missing_data
from generalize.model.pipeline.pipelines import create_pipeline
from generalize.model.utils import detect_train_only, add_split_to_groups
from generalize.evaluate.roc_curves import ROCCurves, ROCCurve

# TODO Consider order of the arguments
# TODO Should split be allowed to have a split per repetition?


def train_full_model(
    x: np.ndarray,
    y: np.ndarray,
    model: Callable,
    grid: Optional[Dict[str, List[any]]] = None,
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
    split_id_colname: str = "Split",
    process_predictions_fn: Callable = None,
    metric: Union[str, List[str]] = "balanced_accuracy",
    task: str = "binary_classification",
    refit_fn: Optional[Callable] = None,
    transformers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    train_test_transformers: List[str] = [],
    add_channel_dim: bool = False,
    add_y_singleton_dim: bool = False,
    rm_missing: bool = False,
    num_jobs: int = 1,
    seed: int = 1,
    grid_error_score: Union[str, int, float] = np.nan,
    # cv_error_score: Union[str, int, float] = np.nan,
    identifier_cols_dict: dict = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
):
    """
    Train a model on all the given data and evaluate the performance on the training set.
    The evaluation is purely to see that the model is capable of predicting its
    training data to a meaningful degree.

    Specify data points to only use as training data during hyperparameter tuning cross-validation
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
        We actively support scikit-learn models.
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
        ID are put in the fold during cross-validation.
        WARNING: When `split` is specified, it has precedence. IDs that
        are present in multiple splits are NOT respected as a single
        entity.
        See `aggregate_by_groups` for prediction aggregation during evaluation.
        **Train Only**: Wrap a group identifier in the `"train_only(ID)"` string
        (where `ID` is the group identifier) for samples that should only be used
        for training.  When `split` is specified, indices that are
        specified as "train only" in either are used only during training
        for both the cross-validation and full model!
        NOTE: Beware of introducing train/test leakage.
    positive : str, int or `None`
        Value that the positive class has in `y`.
        Required when `task` is 'binary_classification' and ignored otherwise.
    y_labels : Dict[int, str]
        Dict mapping unique values in `y` to their class names.
        When `None`, the names in `y` are used.
        Only used for classification tasks.
    k : int
        Number of folds in the hyperparameter tuning cross-validation.
        Ignored when `split` is specified.
    split : list or `numpy.ndarray` or `None`
        Pre-specified fold identifiers for hyperparameter tuning.
        Can be either integers or strings.
        E.g. Useful when passing multiple datasets and hyperparameters should be chosen
        by best generalizability across the datasets.
        **Train Only**: Wrap a split identifier in the `"train_only(ID)"` string
        (where `ID` is the split identifier) for samples that should always be in
        the training set (i.e. never be in the validation fold).
        This "train only" setting is used in both the hyperparameter tuning cross-validation
        and the full model training.
        When `groups` is specified, indices that are specified as "train only" in
        either are used only during training.
    split_weights
        Dictionary with loss weights per split ID (i.e. `split id -> weight`).
        Ignored when `split` is not specified.
        NOTE: The split IDs must match those in `outer_split`, including
        `train_only()` wrapping.
    eval_by_split : bool
        Whether to evaluate by splits instead of all predictions at once.
        When enabled, `split` must be specified.
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
        The weights are calculated per on the entire training set and used in both
        model fitting and grid search. Since it is not calculated per grid search
        cross-validaton fold, the weighting may not be completely balanced during
        grid search.
        NOTE: When not enabling `weight_per_split` as well,
        it may be preferable to instead use the balancing
        integrated directly in the scikit-learn models.
        NOTE: Take care not to enable class balancing here AND in the model
        (scikit-learn models has this option)!
    weight_per_split : bool
        Whether to perform the loss weighting separately per split.
        Affects both class- and group-based weighting.
        E.g. when each split is a dataset with some set of biases
        that shouldn't be ascribed to the majority class. Instead of weighting based
        on the overall class imbalance, we fix the imbalance within each dataset.
        NOTE: May not be meaningful when `split` is not specified.
    process_predictions_fn : callable
        A function for processing the predictions from `model` prior to evaluation.
        E.g. shape flattening, applying softmax or truncation.
        The function should take a single argument (the predictions)
        and return a single output (the processed predictions).
        Skorch models use their `.predict_nonlinearity()` method for this by default.
    metric : str or List[str]
        Metric(s) to select hyperparameters from.
        Passed to `sklearn.model_selection.GridSearchCV(scoring=)`.
        Can be a single string or a list of strings. When multiple strings
        are given, the *first* is used to select the best set of hyperparameters.
        See metric names at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        Tip: Choose a metric that handles class imbalances decently.
    task : str
        Which task to cross-validate. One of:
            {'binary_classification', 'multiclass_classification', 'regression'}.
    refit_fn
        An optional function for finding the best hyperparameter
        combination from `cv_results_` in grid search.
    transformers: list of tuples with (<name>, <transformer>)
        Transformation steps to add to first part of the pipeline.
        Should handle 2- and/or 3-dimensional arrays. See `DimTransformerWrapper` for this.
        Can be built with `PipelineDesigner`.
    add_channel_dim : bool
        Whether to add a singleton dimension to `x` at the second index
        for models that require a channel dimension (such as convolutional neural networks).
    add_y_singleton_dim : bool
        Whether to add a singleton dimension to `y` at the second index
        for models that require a 2D targets array.
        Only applied when `y` is a 1D array.
    rm_missing : bool
        Whether to remove missing data (i.e. `numpy.nan`).
    num_jobs : int
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``-1`` means using all processors.
    seed : int
        Random state used for splitting data into folds.
        Each repetition will use `seed`+repetition as seed.
    grid_error_score : int, float or str
        Error score passed to `GridSearch`. Use `"raise"` to get errors.
        See `GridSearch` from scikit-learn for more details.
    cv_error_score : int, float or str TODO for this fn?
        Error score passed to `cross_validate`. Use `"raise"` to get errors.
        See `cross_validate` from scikit-learn for more details.
    identifier_cols_dict : Dict[str, str]
        Dict mapping colname -> string to add to the results
        and one-vs-all data frames *after* repetitions are concatenated.
        TODO Explain this better!
    messenger : `utipy.Messenger` or `None`
        A `utipy.Messenger` instance used to print/log/... information.
        When `None`, no printing/logging is performed.
        The messenger determines the messaging function (e.g. `print`)
        and potential indentation.

    Returns
    -------
    dict
        ``Estimator`` : Fitted `model`
        ``Evaluation`` : dict TODO!!!!
            ``Summary`` : dict or `None`
                Summarized evaluation scores and potential confusion matrices.
                NOTE: `None` when reps == 1.
                When `split` is specified and `eval_by_split` is enabled,
                it includes summaries per split group.
            ``Evaluations`` : dict
                Evaluation scores for each repetition along with potential
                confusion matrices, ROC curves and One-vs-All evaluations.
        ``Predictions`` : `Pandas.DataFrame`
            Predictions from applying the model on the **training** set.
        ``Targets`` : `numpy.ndarray`
            The targets for the tested-on training data. Matches
            order of the `Predictions`.
        ``Indices`` : Indices of the training data for the predictions.
        ``Split`` : The splits of the training data ``Predictions``.
            These differ from the input `split` by not including the
            "train-only" training samples.
        ``Warnings`` : Warnings found when fitting ``Estimator`` on all the data.
    """

    # Check messenger (always returns Messenger instance)
    messenger = check_messenger(messenger)

    # Start by copying the input data
    # to avoid altering them in parent scope
    x = x.copy()
    y = y.copy()

    if grid is None:
        grid = {}

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
    _check_args(
        x=x,
        y=y,
        grid=grid,
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
        metric=metric,
        task=task,
        transformers=transformers,
        train_test_transformers=train_test_transformers,
        add_channel_dim=add_channel_dim,
        add_y_singleton_dim=add_y_singleton_dim,
        rm_missing=rm_missing,
        num_jobs=num_jobs,
        seed=seed,
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

    # Add splits to groups when necessary
    groups, split = add_split_to_groups(
        groups=groups,
        split=split,
        weight_per_split=weight_per_split,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        k_inner=k,  #
    )

    # Check the model can be used with the pipeline
    required_methods = ["fit", "predict"]
    for meth in required_methods:
        if not hasattr(model, meth):
            raise ValueError(f"`model` does not have the required `{meth}` method.")

    # Create pipeline
    pipe = create_pipeline(
        model=model,
        # TODO We could have models preferring non-flattened input in the future
        flatten_feature_sets=False,  # TODO How should this be handled?
        add_channel_dim=add_channel_dim,
        transformers=transformers,
        train_test_transformers=train_test_transformers,
        weight_loss_by_groups=False,  # Taken care of in GridSearch
        weight_loss_by_class=False,  # Taken care of in GridSearch
        weight_per_split=False,  # Taken care of in GridSearch
        split_weights=None,  # Taken care of in GridSearch
    )

    messenger("Pipeline:\n", pipe)

    messenger("Preparing grid search")

    # Ensure grid names point to their pipeline step
    pipeline_keys = list(pipe.named_steps.keys())
    for key in grid.keys():
        if "__" not in key or key.split("__")[0] not in pipeline_keys:
            raise ValueError(
                f"Grid keys must be prefixed by either 'model__' or "
                f"'<transformer name>__' but got {key}."
            )

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

    refit = True
    if refit_fn is not None:
        refit = refit_fn
    elif isinstance(metric, list):
        refit = metric[0]

    # Create function that returns a NestableGridSearchCV object
    estimator = NestableGridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring=metric,
        n_jobs=num_jobs,
        refit=refit,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        split_weights=split_weights,
        messenger=messenger,
        error_score=grid_error_score,
        cv=folder,
        seed=seed,
    )

    # Perform grid search
    messenger("Running grid search")
    estimator.fit(X=x, y=y, groups=groups)

    # Get best model (refitted on all the data)
    best_estimator = estimator.best_estimator_

    # Predict the *TRAINING* set
    # But only on the non-"train-only" data points
    x_test = x
    y_test = y
    test_groups = groups
    split_test = split
    test_indices = np.arange(len(x))
    if groups is not None and any(["train_only" in str(g) for g in groups]):
        test_indices = test_indices[["train_only" not in str(g) for g in groups]]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
        test_groups = groups[test_indices]
        if split is not None:
            split_test = split[test_indices]

    predictions = best_estimator.predict_proba(X=x_test)

    # Process the predictions
    # NOTE: It cannot be added to the pipeline
    # as the model must be the last step in it
    # Skorch models can use `predict_nonlinearity` method instead
    if process_predictions_fn is not None:
        predictions = process_predictions_fn(predictions)

    # Remove potential singleton dimension from binary predictions
    if task == "binary_classification":
        predictions = BinaryPreparer.prepare_probabilities(probabilities=predictions)

    # Get fold names instead of the index
    # when `split` was specified
    named_split = None
    if split_test is not None:
        # Get names for each fold
        split_index_names = dict(
            enumerate(specified_folds_iterator(folds=split_test, yield_names=True))
        )

        # Get max name length for specifying array size
        max_name_length = max([len(str(name)) for name in split_index_names.values()])

        # Convert fold indices to fold names
        named_split = split_test.astype(f"U{max_name_length}")
        for idx, name in split_index_names.items():
            named_split[np.array(split_test) == np.array(idx)] = str(name)

    # Evaluate the predictions
    evaluation = evaluate_repetitions(
        predictions_list=[predictions],
        targets=y_test,
        task=task,
        positive=positive,
        groups=test_groups if aggregate_by_groups else None,
        splits_list=[split_test] if split_test is not None else None,
        thresholds=None,
        target_labels=y_labels,
        identifier_cols_dict=identifier_cols_dict,
        summarize_splits=eval_by_split,
        split_id_colname=split_id_colname,
        messenger=messenger,
    )

    # Move the single evaluation out in outer dict
    del evaluation["Summary"]
    evaluation.update(evaluation.pop("Evaluations"))
    evaluation["What"] = (
        f"Evaluation of full model ({task.replace('_',' ')}) on *training* data."
    )

    # Calculate average and overall ROC curves for extracting thresholds during inference
    if "ROC" in evaluation:
        # Average ROC curve
        roc_curves: ROCCurves = evaluation["ROC"]
        average_roc_curve = roc_curves.get_average_roc_curve(paths=roc_curves.paths)
        evaluation["ROC"].add(path="Average", roc_curve=average_roc_curve)

        # Overall ROC curve
        evaluation["ROC"].add(
            path="Overall",
            roc_curve=ROCCurve.from_data(
                targets=y_test,
                predicted_probabilities=predictions,
                positive=1,
            ),
        )

    return {
        "Estimator": best_estimator,
        "CV Results": estimator.cv_results_,
        "Evaluation": evaluation,
        "Predictions": predictions,
        "Targets": y_test,
        "Groups": test_groups,
        "Indices": test_indices,
        "Split": named_split,
        "Warnings": estimator.warnings,
    }


def _check_args(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    grid: Dict[str, List[any]],
    y_labels: Optional[dict],
    k: int,
    split: Optional[Union[List[Union[int, str]], np.ndarray]],
    split_weights: Optional[Dict[str, float]],
    eval_by_split: bool,
    aggregate_by_groups: bool,
    weight_loss_by_groups: bool,
    weight_loss_by_class: bool,
    weight_per_split: bool,
    metric: Union[str, List[str]],
    task: str,
    transformers: Optional[List[Tuple[str, BaseEstimator]]],
    train_test_transformers: List[str],
    add_channel_dim: bool,
    add_y_singleton_dim: bool,
    rm_missing: bool,
    num_jobs: int,
    seed: int,
):
    """
    Check some of the arguments for `train_full_model()`.
    """

    assert task in ["binary_classification", "multiclass_classification", "regression"]

    if not (
        isinstance(metric, str)
        or (
            isinstance(metric, list)
            and metric  # Not empty
            and isinstance(metric[0], str)
        )
    ):
        raise ValueError("`metric` must be either a string or a list of strings.")

    if not grid:
        raise NotImplementedError("Grid was empty. This must be supported! Complain!")

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == len(y)
    assert y_labels is None or isinstance(y_labels, dict)
    if groups is not None:
        assert isinstance(groups, np.ndarray)
        assert len(x) == len(groups)

    assert isinstance(k, int)
    assert isinstance(num_jobs, int)
    assert isinstance(seed, int)
    assert isinstance(add_channel_dim, bool)
    assert isinstance(add_y_singleton_dim, bool)
    assert isinstance(rm_missing, bool)
    assert isinstance(aggregate_by_groups, bool)
    assert isinstance(weight_loss_by_groups, bool)
    assert isinstance(weight_loss_by_class, bool)
    assert isinstance(weight_per_split, bool)

    if weight_per_split and split is None:
        raise ValueError(
            "Enabling `weight_per_split` is not meaningful when `split` is `None`."
        )

    if eval_by_split and split is None:
        raise ValueError("When `eval_by_split` is enabled, `split` must be specified.")
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
