from typing import Optional, Union, List, Set, Any
import numpy as np
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


class RowScaler(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        center: Optional[Union[str, List[str]]] = "mean",
        scale: Optional[Union[str, List[str]]] = "std",
        feature_groups: Optional[List[List[int]]] = None,
        center_by_features: Optional[Union[int, List[int]]] = None,
        rm_center_by_features: bool = False,
        copy: bool = True,
    ):
        """
        Scales and/or centers rows with a selection of metrics per transformation.

        Parameters
        ----------
        center
            What metric to center by. One of {"mean", "median"}.
            When `feature_groups` are specified, can be a list
            with one metric per feature group.
        scale
            What metric to scale by. One of {"std", "iqr", "mad", "mean", "median"}.
            When `feature_groups` are specified, can be a list
            with one metric per feature group.
        feature_groups
            Optional list of lists with indices of feature groups to transform together.
            When specified, indices that are not in a sub list are ignored.
        center_by_features
            Optional index / list of indices of feature(s) to use for centering
            instead of the row mean/median.
            When `feature_groups` is specified, a list of one centering feature
            per group. Specify `-1` to use the row mean for the specific feature group.
        rm_center_by_features
            Whether to remove `center_by_features` from `X` after the centering.
        copy
            Whether to copy the input data on `.transform()`
            prior to the transformation.
        """
        self.center = center
        self.scale = scale
        self.feature_groups = feature_groups
        self.center_by_features = center_by_features
        self.rm_center_by_features = rm_center_by_features
        self.copy = copy

        if self.feature_groups is not None:
            all_indices = [idx for indices in self.feature_groups for idx in indices]
            unique_indices = set(all_indices)
            if len(all_indices) != len(unique_indices):
                raise ValueError(
                    "One or more duplicate indices were found in `feature_groups`."
                )
            if any([i < 0 for i in unique_indices]):
                raise ValueError(
                    "One or more indices in `feature_groups` was negative."
                )

        if self.center_by_features is not None:
            if not isinstance(self.center_by_features, (list, int)):
                raise TypeError(
                    "`center_by_features` must be either an integer or "
                    "a list of integers (when `feature_groups` are specified)."
                )
            if self.feature_groups is None:
                if isinstance(self.center_by_features, list):
                    raise TypeError(
                        "When `feature_groups` is not specified, `center_by_features` "
                        "should be a single integer (not list)."
                    )
            else:
                if not isinstance(self.center_by_features, list):
                    raise TypeError(
                        "When `feature_groups` is specified, "
                        "`center_by_features` must be a list (or `None`)."
                    )
                if len(self.center_by_features) != len(self.feature_groups):
                    raise ValueError(
                        "When both `feature_groups` and `center_by_features` are specified, "
                        "they must have the same number of elements."
                    )
                if unique_indices.intersection(self.center_by_features):
                    raise ValueError(
                        "Indices from `center_by_features` was part of one or more `feature_groups`."
                    )
        elif self.rm_center_by_features:
            raise ValueError(
                "`rm_center_by_features` was enabled but `center_by_features` was `None`. "
                "That is not meaningful."
            )

        self._check_param_arg(
            arg=center, valid_options={"mean", "median"}, arg_name="center"
        )
        self._check_param_arg(
            arg=scale,
            valid_options={"std", "iqr", "mad", "mean", "median"},
            arg_name="scale",
        )

    def _check_param_arg(
        self, arg: Any, valid_options: Set[str], arg_name: str
    ) -> None:
        if arg is not None:
            if not isinstance(arg, (str, list)):
                raise TypeError(
                    f"When specified, `{arg_name}` must be either a string or a list of strings."
                )

            if isinstance(arg, list):
                if self.feature_groups is None:
                    raise TypeError(
                        f"`{arg_name}` was a list but `feature_groups` was `None`. "
                        "For non-grouped transformations, provide a string instead."
                    )
                if len(self.feature_groups) != len(arg):
                    raise ValueError(
                        f"When `{arg_name}` is a list it must have the same length as the number of feature groups."
                    )
                unique_passed = set(arg)
            else:
                unique_passed = {arg}

            if any([not isinstance(p, str) for p in unique_passed]):
                raise TypeError(
                    f"When specified, `{arg_name}` must be either a string or a list of strings."
                )

            non_allowed = set(unique_passed).difference(valid_options)
            if non_allowed:
                raise ValueError(
                    f"The following options were invalid for {arg_name}: {non_allowed}. Valid options are {set(valid_options)}"
                )

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Required but does nothing. Row scaling is independent per sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features).
            Checked for validity, used to extract number of features and then ignored.
        y : None
            Checked for validity and then ignored.

        Returns
        -------
        self : object
            Scaler.
        """
        # Reset internal state before fitting
        X, y = check_X_y(X, y, copy=False)
        self.num_features_ = X.shape[-1]
        if self.feature_groups is not None:
            max_feature_group_idx = max(
                [idx for indices in self.feature_groups for idx in indices]
            )
            if max_feature_group_idx >= self.num_features_:
                raise ValueError(
                    "An index in `feature_groups` was higher than the number of features in `X`."
                )
        return self

    def transform(self, X: np.ndarray, copy: Optional[bool] = None):
        """Perform row-wise scaling and/or centering with the chosen metrics.

        First calculates the parameters, then applies the transformations.

        Scaling is performed *after* centering.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        check_is_fitted(self)
        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, dtype=np.float64)
        if not X.shape[-1] == self.num_features_:
            raise ValueError(
                "`X` had a different number of features in "
                f"`transform` ({X.shape[-1]}) than `fit` ({self.num_features_})."
            )

        # Calculate parameters
        if self.feature_groups is None:
            feature_indices = np.arange(X.shape[-1])
            if self.center_by_features is not None and self.center_by_features != -1:
                feature_indices = [
                    idx for idx in feature_indices if idx != self.center_by_features
                ]
            X[:, feature_indices] = self._transform_group(
                X_subset=X[:, feature_indices],
                scale=self.scale,
                center=self.center,
                center_feature=(
                    X[:, self.center_by_features]
                    if self.center_by_features is not None
                    and self.center_by_features != -1  # -1 means use row `center`
                    else None
                ),
            )
        else:
            for group_idx, group_indices in enumerate(self.feature_groups):
                X[:, group_indices] = self._transform_group(
                    X_subset=X[:, group_indices],
                    scale=(
                        self.scale
                        if isinstance(self.scale, str) or self.scale is None
                        else self.scale[group_idx]
                    ),
                    center=(
                        self.center
                        if isinstance(self.center, str) or self.center is None
                        else self.center[group_idx]
                    ),
                    center_feature=(
                        X[:, self.center_by_features[group_idx]]
                        if self.center_by_features is not None
                        and self.center_by_features[group_idx]
                        != -1  # -1 means use row `center`
                        else None
                    ),
                )

        if self.rm_center_by_features:
            X = np.delete(X, self.center_by_features, axis=-1)

        return X

    @staticmethod
    def _transform_group(
        X_subset: np.ndarray,
        scale: Optional[str],
        center: Optional[str],
        center_feature: Optional[np.ndarray],
    ) -> np.ndarray:
        if scale is not None:
            if scale == "std":
                scaling_factors = np.std(X_subset, axis=-1)
            elif scale == "iqr":
                q75, q25 = np.percentile(X_subset, [75, 25], axis=-1)
                scaling_factors = q75 - q25
            elif scale == "mad":
                scaling_factors = np.mean(
                    np.abs(X_subset - np.mean(X_subset, axis=-1)), axis=-1
                )
            elif scale == "mean":
                scaling_factors = np.mean(X_subset, axis=-1)
            elif scale == "median":
                scaling_factors = np.median(X_subset, axis=-1)

        if center_feature is not None:
            centers = center_feature
        elif center is not None:
            if center == "mean":
                centers = np.mean(X_subset, axis=-1)
            elif center == "median":
                centers = np.median(X_subset, axis=-1)

        # Apply transformations

        if center is not None:
            X_subset -= np.expand_dims(centers, axis=-1)

        if scale is not None:
            X_subset /= np.expand_dims(scaling_factors, axis=-1)

        return X_subset

    def _more_tags(self):
        return {"allow_nan": False}
