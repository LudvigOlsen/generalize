"""
A version of sklearn's `cross_validate()` that allows returning 
predictions as well as the scores.

TODO: Update to work with never scikit-learn and scipy versions.
Currently works with v1.0.2 and scipy 1.7.3

Some of the changes were taken from:
https://github.com/scikit-learn/scikit-learn/pull/18390

"""

import numpy as np

import numbers
import time
from traceback import format_exc

import numpy as np
from joblib import Parallel, logger

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _score, _warn_about_fit_failures, _insert_error_scores, _aggregate_score_dicts, _normalize_score_results, _enforce_prediction_order, _check_is_permutation

# TODO If the changes in https://github.com/scikit-learn/scikit-learn/pull/18390
# (plus returning split indices)
# are merged this function can be removed
# TODO Must be held up-to-date with scikit-learn, as these or the imported functions may change


def cross_validate_with_predictions(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    allow_partial_test=False,
    pass_groups_to_estimator=False,
    return_train_score=False,
    return_estimator=False,
    return_predictions=False,
    error_score=np.nan,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.
    Read more in the :ref:`User Guide <multimetric_cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).
    scoring : str, callable, list, tuple, or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.
        If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
        See :ref:`multimetric_grid_search` for an example.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`.Fold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : int, default=0
        The verbosity level.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    allow_partial_test : bool, default=False
        Whether to allow the testing data not to contain all
        the training samples. Can be used when some training
        data should not be tested on.
    pass_groups_to_estimator : bool, default=False
        Whether to pass the groups of the training data to 
        ``estimator.fit(..., groups=groups_train)``.
        Useful when the estimator (e.g. ``GridSearchCV``) implements 
        cross-validation.
        Ignored when ``groups`` is None.
    return_train_score : bool, default=False
        Whether to include train scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.
        .. versionadded:: 0.19
        .. versionchanged:: 0.21
            Default value was changed from ``True`` to ``False``
    return_estimator : bool, default=False
        Whether to return the estimators fitted on each split.
        .. versionadded:: 0.20
    return_predictions : {'predict_proba', 'predict'}, default=None
        What predictions to return. If 'predict', return output of
        :term:`predict` and if 'predict_proba', return output of
        :term:`predict_proba`.
        If None, do not return predictions.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
        .. versionadded:: 0.20

    Returns
    -------
    scores : dict of float arrays of shape (n_splits,) # TODO: This is no longer just "scores" and probably the shape in this line is wrong
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
                This is available only if ``return_train_score`` parameter
                is ``True``.
            ``fit_time``
                The time for fitting the estimator on the train
                set for each cv split.
            ``score_time``
                The time for scoring the estimator on the test set for each
                cv split. (Note time for scoring on the train set is not
                included even if ``return_train_score`` is set to ``True``
            ``estimator``
                The estimator objects for each cv split.
                This is available only if ``return_estimator`` parameter
                is set to ``True``.
            ``predictions``
                Cross-validation predictions for the dataset.
                This is available only if ``return_predictions`` parameter
                is set to ``predict`` or ``predict_proba``.
            ``split_indices``
                Cross-validation split indices (fold identifiers).
                Can be used calculate scores per fold.
                This is available only if ``return_predictions`` parameter
                is set to ``predict`` or ``predict_proba``.
            ``test_indices``
                Indices in ``X`` that the predictions refer to.
                Useful when not all elements in ``X`` are in
                the test folds. See ``allow_partial_test``.
                This is available only if ``return_predictions`` parameter
                is set to ``predict`` or ``predict_proba``.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_validate
    >>> from sklearn.metrics import make_scorer
    >>> from sklearn.metrics import confusion_matrix
    >>> from sklearn.svm import LinearSVC
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    Single metric evaluation using ``cross_validate``
    >>> cv_results = cross_validate(lasso, X, y, cv=3)
    >>> sorted(cv_results.keys())
    ['fit_time', 'score_time', 'test_score']
    >>> cv_results['test_score']
    array([0.33150734, 0.08022311, 0.03531764])
    Multiple metric evaluation using ``cross_validate``
    (please refer the ``scoring`` parameter doc for more information)
    >>> scores = cross_validate(lasso, X, y, cv=3,
    ...                         scoring=('r2', 'neg_mean_squared_error'),
    ...                         return_train_score=True)
    >>> print(scores['test_neg_mean_squared_error'])
    [-3635.5... -3573.3... -6114.7...]
    >>> print(scores['train_r2'])
    [0.28010158 0.39088426 0.22784852]
    See Also
    ---------
    cross_val_score : Run cross-validation for single metric evaluation.
    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.
    """
    if (return_predictions is not None) and \
            (not isinstance(return_predictions, str)
             or return_predictions not in ["predict", "predict_proba"]):
        raise ValueError("return_predictions must either be 'predict' "
                         "or 'predict_proba'")

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])

    # TODO: Explain this part in a comment
    if allow_partial_test:
        if len(np.unique(test_indices)) != len(test_indices):
            raise ValueError(
                "cross_validate_with_predictions only works for partitions"
            )
    elif not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError(
            "cross_validate_with_predictions only works for partitions"
        )

    # Inverse indices to sort predictions by
    # same as `test_indices.argsort()` but faster (no sorting)
    try:
        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))
    except IndexError:
        if not allow_partial_test:
            # The above should not fail if test data is
            # expected to be complete
            raise

        # When not all indices in `X` are also in `test_indices`
        # we use sorting (slower) instead
        inv_test_indices = test_indices.argsort()

    if callable(scoring):
        scorers = scoring
    elif scoring is None or isinstance(scoring, str):
        scorers = check_scoring(estimator, scoring)
    else:
        scorers = _check_multimetric_scoring(estimator, scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        pre_dispatch=pre_dispatch
    )

    results = parallel(
        delayed(_fit_predict_score)(
            estimator=clone(estimator),
            X=X,
            y=y,
            groups=groups,
            scorer=scorers,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=fit_params,
            pass_groups_to_estimator=pass_groups_to_estimator,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            return_predictions=return_predictions,
            error_score=error_score,
            split_progress=(split_idx, cv.get_n_splits())
        )
        for split_idx, (train, test) in enumerate(splits)
    )

    _warn_about_fit_failures(results, error_score)

    # For callable scoring, the return type is only known after calling. If the
    # return type is a dictionary, the error scores can now be inserted with
    # the correct key.
    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]
    if "warnings" in results:
        ret["warnings"] = results["warnings"]

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    if return_predictions:
        if return_predictions == "predict":
            predictions = np.hstack(results["predictions"])
        else:
            predictions = np.vstack(results["predictions"])
        split_indices = np.hstack(results["split_indices"])
        ret['predictions'] = predictions[inv_test_indices]
        ret['split_indices'] = split_indices[inv_test_indices]
        ret['test_indices'] = test_indices[inv_test_indices]

    return ret


def _fit_predict_score(
    estimator,
    X,
    y,
    groups,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    pass_groups_to_estimator=False,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    return_predictions=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like of shape (n_samples,) or None
        The group IDs to pass to the estimator in case it
        has cross-validation built-in (e.g. nested cross-validation).
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    pass_groups_to_estimator : bool, default=False
        Whether to pass the groups of the training data to 
        ``estimator.fit(..., groups=groups_train)``.
        Useful when the estimator (e.g. ``GridSearchCV``) implements 
        cross-validation.
        Ignored when ``groups`` is None.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    return_predictions : {'predict_proba', 'predict'}, default=None
        What predictions to return. If 'predict', return output of
        :term:`predict` and if 'predict_proba', return output of
        :term:`predict_proba`. The test set indices are also returned under a
        separate dict key.
        If None, do not return predictions or test data indices.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
        test_indices : array-like of shape (n_test_samples,)
            Indices of test samples.
        predictions : float
            Predicted class or predicted class probability of
            test samples.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    # TODO Not in cross_val_predict - should it be?
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    if groups is not None and pass_groups_to_estimator:
        groups_train = _safe_indexing(groups, train)
        fit_params["groups"] = groups_train

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train,
                                  y_train, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    if return_predictions:
        result["split_indices"] = [split_progress[0]] * _num_samples(X_test)
        result["predictions"] = _predict(
            estimator=estimator,
            X_test=X_test,
            y=y,
            method=return_predictions
        )
    if hasattr(estimator, "warnings"):
        result["warnings"] = estimator.warnings
    return result


def _predict(estimator, X_test, y, method):
    func = getattr(estimator, method)
    predictions = func(X_test)

    encode = (
        method in ["decision_function", "predict_proba", "predict_log_proba"]
        and y is not None
    )

    if encode:
        if isinstance(predictions, list):
            predictions = [
                _enforce_prediction_order(
                    estimator.classes_[i_label],
                    predictions[i_label],
                    n_classes=len(set(y[:, i_label])),
                    method=method,
                )
                for i_label in range(len(predictions))
            ]
        else:
            # A 2D y array should be a binary label indicator matrix
            n_classes = len(set(y)) if y.ndim == 1 else y.shape[1]
            predictions = _enforce_prediction_order(
                estimator.classes_, predictions, n_classes, method
            )
    return predictions
