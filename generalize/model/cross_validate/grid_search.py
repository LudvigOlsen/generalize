from numbers import Number
import random
import pathlib
from typing import Callable, List, Optional, Union, Dict, Tuple
import warnings
import copy
import numpy as np
import pandas as pd
from traceback import format_exc
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from utipy import Messenger, check_messenger
from copy import deepcopy

from generalize.model.utils.weighting import calculate_sample_weight
from generalize.model.pipeline.pipelines import AttributeToDataFrameExtractor


class NestableGridSearchCV(GridSearchCV):
    def __init__(
        self,
        estimator,
        param_grid: Union[dict, List[dict]],
        seed: Optional[int],
        scoring: Optional[Union[str, Callable, list, tuple, dict]] = None,
        n_jobs: Optional[int] = None,
        refit: Union[bool, str, Callable] = True,
        cv=None,
        weight_loss_by_groups: bool = False,
        weight_loss_by_class: bool = False,
        weight_per_split: bool = False,
        weight_splits_equally: bool = False,
        split_weights: Optional[Dict[str, float]] = None,
        verbose: int = 0,
        pre_dispatch: Union[str, int] = "2*n_jobs",
        error_score: Union[Number, str] = np.nan,
        return_train_score: bool = False,
        seed_callback_name: str = "fix_random_seed",
        store_attributes: List[AttributeToDataFrameExtractor] = None,
        save_cv_results_path: Optional[Union[str, pathlib.Path]] = None,
        messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
    ) -> None:
        """
        GridSearch class used in nested cross-validation with options to:
            - Save .cv_results_ to disk.
            - Calculate `sample_weight`
            - Set the seed in a `skorch` callback

        NOTE: `estimator` is expected to be a `Pipeline` where
        the final model is called `model`.

        weight_loss_by_groups : bool
            Whether to weight samples by the group size in training loss.
            Each sample in a group gets the weight `1 / group_size`.
            Passed to model's `.fit(sample_weight=)` method.
            When existing sample weights are passed via
            `NestableGridSearchCV.fit(fit_params={"sample_weight":})`
            the two sets of sample weights are multiplied
            and normalized to sum-to-length.
            Note: Requires `estimator` to be a `Pipeline` where
                  the final model is called `model`, as the sample
                  weights are passed as `model__sample_weights`.
        split_weights
            A dictionary mapping `split ID -> weight`.
        seed_callback_name : str
            Name of callback to set seed in.
            For skorch models only.
        store_attributes
            List of `AttributeToDataFrameExtractor` instances
            for extracting attributes from the `.best_estimator_`
            and converting them to `pandas.DataFrame`s to store.
        save_cv_results_path : str or pathlib.Path
            Path to csv file to append `.cv_results_` dict to.
            Is appended with headers, allowing for multiple threads
            to append without locks.
        messenger : `utipy.Messenger` or None
            A `utipy.Messenger` instance used to print/log/... information.
            When `None`, no printing/logging is performed.
            The messenger determines the messaging function (e.g. `print`)
            and potential indentation.
            Currently only used when fitting fails.
        """
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self._check_save_path(save_cv_results_path)

        self.weight_loss_by_groups = weight_loss_by_groups
        self.weight_loss_by_class = weight_loss_by_class
        self.weight_per_split = weight_per_split
        self.weight_splits_equally = weight_splits_equally
        self.split_weights = split_weights
        self.seed = seed
        self.seed_callback_name = seed_callback_name
        self.store_attributes = store_attributes
        self.save_cv_results_path = save_cv_results_path

        # Check messenger (always returns Messenger instance)
        self.messenger = check_messenger(messenger)

    def _check_save_path(self, path):
        # Check cv results path
        if path is not None:
            if (
                not isinstance(path, (str, pathlib.Path))
                or not str(path)  # empty
                or str(path)[-4:] != ".csv"
            ):
                raise ValueError(
                    "When specified, `save_cv_results_path` must be a path ending in '.json'."
                )
            # Parent folder must exist
            if not path.resolve().parent.exists():
                raise RuntimeError(
                    "`save_cv_results_path´s parent folder did not exist: "
                    f"{path.resolve().parent}"
                )

    def fit(self, X, y=None, groups=None, **fit_params):
        # We cannot reinitialize sklearn models
        # but skorch models is seeded via a callback
        # so we change the seed for each search (fold)
        if (
            hasattr(self.estimator, "is_seedable")
            and self.estimator.is_seedable
            and self.seed is not None
            and has_callback(
                self.estimator.named_steps["model"], self.seed_callback_name
            )
        ):
            self.estimator.named_steps["model"].set_params(
                **{f"callbacks__{self.seed_callback_name}__seed": self.seed}
            )

        # Calculate sample weights
        # We keep the group IDs are are (sometimes includes splits)
        # as some folders need them
        sample_weights, _ = calculate_sample_weight(
            y=y,
            groups=groups,
            weight_loss_by_groups=self.weight_loss_by_groups,
            weight_loss_by_class=self.weight_loss_by_class,
            weight_per_split=self.weight_per_split,
            weight_splits_equally=self.weight_splits_equally,
            split_weights=self.split_weights,
        )

        # Either multiply with passed sample weights
        # Or use group-based sample weights
        if "model__sample_weight" in fit_params:
            fit_params["model__sample_weight"] = (
                np.asarray(fit_params["model__sample_weight"]) * sample_weights
            )

            # Normalize to sum-to-length
            fit_params["model__sample_weight"] = (
                fit_params["model__sample_weight"]
                / fit_params["model__sample_weight"].sum()
            ) * len(fit_params["model__sample_weight"])
        else:
            fit_params["model__sample_weight"] = sample_weights

        fit_problems = []
        if self.error_score == "raise":
            # We may have many convergence warnings, so we catch them
            # and display _one_ afterwards
            with warnings.catch_warnings(record=True) as w:  # Restores filters on exit
                saved_filters = copy.deepcopy(warnings.filters)
                warnings.simplefilter("error")  # 'error'
                warnings.filterwarnings("ignore", category=FutureWarning)

                try:
                    super().fit(X=X, y=y, groups=groups, **fit_params)
                except ConvergenceWarning as e:
                    # Restore the previously saved warning filters
                    # so we don't fail on convergence errors
                    warnings.filters = saved_filters
                    fit_problems.append(
                        (
                            e,
                            "Convergence warning during NestableGridSearchCV.fit():\n  "
                            + str(e)
                            + "\nRunning fit() again.",
                        )
                    )
                    super().fit(X=X, y=y, groups=groups, **fit_params)
                except FutureWarning as e:
                    # Restore the previously saved warning filters
                    # so we don't fail on future warnings
                    warnings.filters = saved_filters
                    fit_problems.append(
                        (
                            e,
                            "Future warning during NestableGridSearchCV.fit():\n  "
                            + str(e)
                            + "\nRunning fit() again.",
                        )
                    )
                    super().fit(X=X, y=y, groups=groups, **fit_params)
                except RuntimeError:
                    raise
                except BaseException as e:
                    fit_problems.append(
                        (
                            e,
                            "Error in NestableGridSearchCV.fit():\n  "
                            + str(e)
                            + "\n"
                            + format_exc(),
                        )
                    )
                    warnings.warn(e)

            # Show warnings / errors (outside catcher)
            if w:
                self.messenger(
                    f"\nGot {len(w)} warnings/errors ({len(set(w))} unique).",
                    add_msg_fn=warnings.warn,
                )
            for problem in fit_problems:
                self.messenger(
                    problem[1],
                    add_msg_fn=warnings.warn,
                )

                # The estimator has not been fitted
                # So we can't continue
                if "Error" in problem:
                    raise problem[0]
        else:
            # We may have many `FitFailedWarning`s, so we catch them
            # and display them afterwards
            with warnings.catch_warnings(record=True) as w:  # Restores filters on exit
                warnings.filterwarnings("ignore", category=FutureWarning)
                super().fit(X=X, y=y, groups=groups, **fit_params)

                # Show warnings / errors (outside catcher)
                warning_count_string = f"Got {len(w)} warnings ({len(set(w))} unique)."

                warning_messages = []
                if len(w) > 0:
                    # Collect warning messages
                    warning_messages = [str(warn.message) for warn in w]

            # Message warnings
            # NOTE: Cannot do this within the warning catcher as
            # it seems to become a recursive catch/warn/catch/warn/...
            # and blows up the memory
            if warning_messages:
                self.messenger(
                    warning_count_string,
                    add_msg_fn=warnings.warn,
                )
                for msg in warning_messages:
                    self.messenger(
                        msg,
                        add_msg_fn=warnings.warn,
                    )

        # TODO Perhaps handle if there's no best_estimator_?
        # Surely it will just fail downstream?
        if not hasattr(self, "best_estimator_"):
            self.messenger(
                "`NestableGridSearchCV` has no `best_estimator_` "
                "after `GridSearchCV.fit()`:",
                add_msg_fn=warnings.warn,
            )
        elif hasattr(self, "best_estimator_") and hasattr(
            self.best_estimator_, "warnings"
        ):
            self.warnings = self.best_estimator_.warnings

        # Save cv results to file as data frame,
        # so we have access to it
        if self.save_cv_results_path is not None:
            try:
                # Convert results to data frame
                cv_results_df = pd.DataFrame(self.cv_results_)

                try:
                    # Add flag for selected model
                    selection_idx = self.refit(self.cv_results_.copy())
                    cv_results_df["selected_parameters"] = False
                    cv_results_df.loc[selection_idx, "selected_parameters"] = True
                except:  # noqa: E722
                    # When refit is not a function, etc.
                    # we don't add the selected model parameters
                    pass

                # Add random identifier to separate appended data frames
                randomgen = random.Random()
                random_id = randomgen.randint(0, 100000)
                cv_results_df["random_id"] = random_id

                # Save header to disk
                save_cv_results_header_path = pathlib.Path(
                    self.save_cv_results_path
                ).with_suffix(".header.csv")
                if not save_cv_results_header_path.exists():
                    with open(str(save_cv_results_header_path), "w") as f:
                        header_string = ",".join(list(cv_results_df.columns))
                        f.write(header_string)

                # Save data frame to disk
                cv_results_df.to_csv(
                    str(self.save_cv_results_path), mode="a", index=False, header=False
                )

                # Save coefficients from best estimator
                if hasattr(self.best_estimator_.named_steps["model"], "coef_"):
                    save_coeffs_path = pathlib.Path(
                        self.save_cv_results_path
                    ).with_suffix(".best_coefficients.csv")

                    best_coeffs = pd.DataFrame(
                        self.best_estimator_.named_steps["model"].coef_
                    )

                    try:
                        # Pad to a larger number to ensure same number of coeffs per model later
                        best_coeffs = pad_best_coeffs(
                            new_data=best_coeffs, total_columns=2000
                        )
                    except:  # noqa: E722
                        pass

                    best_coeffs["random_id"] = random_id

                    # Save data frame to disk
                    best_coeffs.to_csv(
                        str(save_coeffs_path),
                        mode="a",
                        index=False,
                        header=False,
                    )

                if self.store_attributes:
                    pipe_copy = deepcopy(self.best_estimator_)
                    for extractor in self.store_attributes:
                        save_attr_path = pathlib.Path(
                            self.save_cv_results_path
                        ).with_suffix(f".{extractor.name}.csv")
                        attribute_df = extractor(pipe_copy)
                        attribute_df["random_id"] = random_id
                        # Save data frame to disk
                        attribute_df.to_csv(
                            str(save_attr_path),
                            mode="a",
                            index=False,
                            header=False,
                        )

            except BaseException as e:
                self.messenger(
                    "Error when saving in NestableGridSearchCV.fit():",
                    add_msg_fn=warnings.warn,
                )
                self.messenger(e, add_msg_fn=warnings.warn, add_indent=2)
                self.messenger(format_exc(), add_msg_fn=warnings.warn, add_indent=4)

        return self


def pad_best_coeffs(new_data: pd.DataFrame, total_columns: int = 2000):
    # Get the current maximum column number in the new data
    max_col_num = max(map(int, new_data.columns))

    # Determine the starting column number for padding
    start_col_num = max_col_num + 1

    # Pad the DataFrame to ensure it has total_columns columns
    if new_data.shape[1] < total_columns:
        # Add NaN columns to the new_data DataFrame
        padding_cols = total_columns - new_data.shape[1]
        new_data = new_data.reindex(
            columns=[
                *new_data.columns,
                *range(start_col_num, start_col_num + padding_cols),
            ],
            fill_value=np.nan,
        )
    elif new_data.shape[1] > total_columns:
        raise ValueError(f"New data has more than {total_columns} columns")

    return new_data


def has_callback(model, cb_name):
    return hasattr(model, "callbacks") and cb_name in [cb[0] for cb in model.callbacks]


def make_lowest_c_refit_strategy(
    c_name: str, score_name: str, score_direction: str = "auto", verbose: bool = False
):
    """
    Create function for the `refit` argument that finds
    the lowest C value where the average score is within the
    standard deviation of the best scoring C value.
    """

    if score_direction == "auto":
        score_direction = "maximize" if get_scorer(score_name)._sign > 0 else "minimize"
    assert score_direction in ["maximize", "minimize"]

    def lowest_c_refit_strategy(cv_results):
        """Define the strategy to select the best estimator.

        Gets the lowest C value where the average score is within the
        standard deviation of the best scoring C value.

        Parameters
        ----------
        cv_results : dict of numpy (masked) ndarrays or `pandas.DataFrame` based on that dict
            CV results as returned by the `GridSearchCV`.

        Returns
        -------
        best_index : int
            The index of the best estimator as it appears in `cv_results`.
        """
        cv_results = cv_results.copy()

        if isinstance(cv_results, dict):
            cv_results = pd.DataFrame(cv_results)
        else:
            assert isinstance(cv_results, pd.DataFrame)

        used_score_name = score_name
        if f"mean_test_{score_name}" not in cv_results:
            used_score_name = "score"

        if score_direction == "maximize":
            best_score_index = cv_results[f"mean_test_{used_score_name}"].idxmax()
        else:
            best_score_index = cv_results[f"mean_test_{used_score_name}"].idxmin()
        best_score = cv_results.loc[best_score_index, f"mean_test_{used_score_name}"]
        best_score_std = cv_results.loc[best_score_index, f"std_test_{used_score_name}"]

        if score_direction == "maximize":
            score_threshold = best_score - best_score_std
            # Filter out all results below the threshold
            made_threshold_cv_results = cv_results[
                cv_results[f"mean_test_{used_score_name}"] >= score_threshold
            ].copy()

        else:
            score_threshold = best_score + best_score_std
            # Filter out all results above the threshold
            made_threshold_cv_results = cv_results[
                cv_results[f"mean_test_{used_score_name}"] <= score_threshold
            ].copy()

        # Make C a column
        made_threshold_cv_results.loc[:, c_name] = made_threshold_cv_results[
            "params"
        ].apply(lambda d: d[c_name])

        lowest_c_index = made_threshold_cv_results[c_name].idxmin()

        if verbose:
            print("lowest_c_refit_strategy: ")
            print(f"  best_score: {best_score}")
            print(f"  score std: {best_score_std}")
            print(f"  score threshold: {score_threshold}")
            print(
                f"  C: {made_threshold_cv_results[c_name].min()} | "
                f"score: {made_threshold_cv_results.loc[lowest_c_index, f'mean_test_{used_score_name}']}"
            )

        return lowest_c_index

    return lowest_c_refit_strategy


def make_simplest_model_refit_strategy(
    main_var: Tuple[str, str],
    score_name: str,
    other_vars: Optional[List[Tuple[str, str]]] = None,
    score_direction: str = "auto",
    messenger: Optional[Callable] = Messenger(verbose=False, indent=0, msg_fn=print),
):
    """
    Create function for the `refit` argument that finds the hyperparameter
    combination where the average score is within the
    standard deviation of the best scoring combination and
    the lowest or highest value of a main hyperparameter value (e.g. lowest C in LASSO)
    and ensures the values of other hyperparameters are equal to or either higher or lower
    than in the best combination (e.g. lower or same number of PCA components).

    Parameters
    ----------
    main_var : (name, direction)
        The name of the hyperparameter in `cv_results_` and whether
        a lower or higher value would create the simpler model.
        The direction should be one of {'minimize', 'maximize'}.
        If a variable is nested inside a dict, the name can
        specify this with `'[transformer__dict]__variable'`. In
        which case we will first get the 'transformer__dict' column
        from cv_results_ and then the 'variable' element
        from that dict.
    other_vars : List of (name, direction) tuples or None
        List of other hyperparameters which should have a value
        either equal to their value in the best combination
        or a higher / lower value.
        The direction should be one of {'minimize', 'maximize'}.
        If a variable is nested inside a dict, the name can
        specify this with `'[transformer__dict]__variable'`. In
        which case we will first get the 'transformer__dict' column
        from cv_results_ and then the 'variable' element
        from that dict.

    Returns
    -------
    callable
        The refit strategy function.

    """

    if score_direction == "auto":
        score_direction = "maximize" if get_scorer(score_name)._sign > 0 else "minimize"
    assert score_direction in ["maximize", "minimize"]

    def get_direction_fn(direction: bool, or_eq: bool = True):
        direction = direction.lower()
        assert direction in ["maximize", "minimize"]
        return {
            ("maximize", True): lambda x, y: x >= y,
            ("maximize", False): lambda x, y: x > y,
            ("minimize", True): lambda x, y: x <= y,
            ("minimize", False): lambda x, y: x < y,
        }[(direction, or_eq)]

    def simplest_model_refit_strategy(cv_results):
        """Define the strategy to select the best estimator.

        Gets the lowest/highest `main_var` value where the average score is within the
        standard deviation of the best scoring `main_var` value and equal to or higher/lower
        in the `other_vars`.

        Parameters
        ----------
        cv_results : dict of numpy (masked) ndarrays or `pandas.DataFrame` based on that dict
            CV results as returned by the `GridSearchCV`.

        Returns
        -------
        selected_index : int
            The index of the selected estimator as it appears in `cv_results`.
        """
        # Ensure we don't alter the outer cv_results
        cv_results = cv_results.copy()

        if isinstance(cv_results, dict):
            cv_results = pd.DataFrame(cv_results)
        else:
            assert isinstance(cv_results, pd.DataFrame)

        # Create index column
        cv_results["original_index"] = range(len(cv_results))

        # Create columns for the hyperparameters
        # TODO: What happens when the value is a dict?
        assert (
            isinstance(main_var, Tuple)
            and isinstance(main_var[0], str)
            and isinstance(main_var[1], str)
        ), "`main_var` must be a tuple of two strings `(name, direction)`."
        all_hyperparameter_names = [main_var[0]]
        if other_vars is not None:
            all_hyperparameter_names += [var_nm for (var_nm, _) in other_vars]

        for var_nm in all_hyperparameter_names:
            if "[" in var_nm:
                # Column is nested in a dict
                # E.g. '{xx__yy}__zz' means '{xx__yy}' is the column with the dict
                # and 'zz' is the key within the dict
                if var_nm[0] != "[":
                    raise ValueError(
                        "Var name: When specifying a column->dict relationship "
                        "with '[xx]', the first character must be `[`. "
                        f"Got: `{var_nm}`."
                    )
                var_split = var_nm[1:].split("]")
                if len(var_split) != 2:
                    raise ValueError(
                        "Var name: When specifying a column->dict relationship with '[xx]', "
                        "there must be exactly one `]` character and it must not be "
                        "located in the end."
                    )
                var_col_nm, var_key_nm = var_split

                cv_results.loc[:, var_nm] = cv_results["params"].apply(
                    lambda d: d[var_col_nm][var_key_nm.lstrip("_")]
                )

            else:
                cv_results.loc[:, var_nm] = cv_results["params"].apply(
                    lambda d: d[var_nm]
                )

        used_score_name = score_name
        if f"mean_test_{score_name}" not in cv_results:
            used_score_name = "score"

        if score_direction == "maximize":
            best_score_index = cv_results[f"mean_test_{used_score_name}"].idxmax()
        else:
            best_score_index = cv_results[f"mean_test_{used_score_name}"].idxmin()
        best_score = cv_results.loc[best_score_index, f"mean_test_{used_score_name}"]
        best_score_std = cv_results.loc[best_score_index, f"std_test_{used_score_name}"]

        best_score_hparams = cv_results.loc[best_score_index, all_hyperparameter_names]

        score_threshold = (
            best_score - best_score_std
            if score_direction == "maximize"
            else best_score + best_score_std
        )
        made_threshold_cv_results = cv_results.loc[
            get_direction_fn(score_direction)(
                cv_results[f"mean_test_{used_score_name}"],
                score_threshold,
            )
        ].copy()

        # Only keep solutions where all specified variables
        # are equal to or higher/lower (specified per var)
        # than the best solution
        all_vars = [main_var]
        if other_vars is not None:
            all_vars += other_vars

        for var_nm, var_direction in reversed(all_vars):
            made_threshold_cv_results = made_threshold_cv_results.loc[
                get_direction_fn(var_direction)(
                    made_threshold_cv_results[var_nm],
                    best_score_hparams[var_nm],
                )
            ].sort_values(
                [var_nm],
                ascending=var_direction == "minimize",
                kind="stable",  # NOTE: Required for iterative sorting!
            )

        selected_index = made_threshold_cv_results.reset_index(drop=True).loc[
            0, "original_index"
        ]

        if messenger.verbose:
            messenger("simplest_model_refit_strategy: ")
            messenger(f"best_score: {best_score}", add_indent=2)
            messenger(f"score std: {best_score_std}", add_indent=2)
            messenger(f"score threshold: {score_threshold}", add_indent=2)

            messenger("Best model", add_indent=2)
            messenger(
                f"Parameters: {cv_results.loc[best_score_index, all_hyperparameter_names].to_dict()}",
                add_indent=4,
            )
            messenger(
                f"score: {cv_results.loc[best_score_index, f'mean_test_{used_score_name}']}",
                add_indent=4,
            )

            messenger("Selected model")
            messenger(
                f"Parameters: {cv_results.loc[selected_index, all_hyperparameter_names].to_dict()}",
                add_indent=4,
            )
            messenger(
                f"score: {cv_results.loc[selected_index, f'mean_test_{used_score_name}']}",
                add_indent=4,
            )

        return int(selected_index)

    return simplest_model_refit_strategy
