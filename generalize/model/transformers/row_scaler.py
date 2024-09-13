from typing import Optional, Union, List, Set, Any
import warnings
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
        center_by_features: Optional[List[int]] = None,
        scale_by_features: Optional[List[int]] = None,
        rm_center_by_features: bool = False,
        rm_scale_by_features: bool = False,
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
            To center by the actual value in a feature, simply
            supply only that feature (or a feature per feature group)
            via `center_by_features` as the mean of a single value
            is just that value.
        scale
            What metric to scale by. One of {"std", "iqr", "mad", "mean", "median", "sum"}.
            When `feature_groups` are specified, can be a list
            with one metric per feature group.
        feature_groups
            Optional list of lists with indices of feature groups to transform together.
            When specified, indices that are not in a sub list are ignored.
        center_by_features
            Optional subset of feature indices to use for centering.
            The `center` value is calculated from these features.
            When `feature_groups` are specified, the overlapping `center_by_features`
            are used within each feature group. Feature groups with no overlapping
            `center_by_features` use all the features in the group.
            NOTE: When a single column (or single column per feature group),
            it is not centered itself, as it would simply become 0 everywhere.
        scale_by_features
            Optional subset of feature indices to use for scaling.
            The `scale` value is calculated from these features.
            When `feature_groups` are specified, the overlapping `scale_by_features`
            are used within each feature group. Feature groups with no overlapping
            `scale_by_features` use all the features in the group.
            NOTE: When a single column (or single column per feature group),
            it is not scaled itself, as it would simply become 1 everywhere.
        rm_center_by_features
            Whether to remove `center_by_features` from `X` after the centering.
        rm_scale_by_features
            Whether to remove `scale_by_features` from `X` after the centering.
        copy
            Whether to copy the input data on `.transform()`
            prior to the transformation.
        """
        self.center = center
        self.scale = scale
        self.feature_groups = feature_groups
        self.center_by_features = center_by_features
        self.scale_by_features = scale_by_features
        self.rm_center_by_features = rm_center_by_features
        self.rm_scale_by_features = rm_scale_by_features
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
            if not isinstance(self.center_by_features, list):
                raise TypeError(
                    "When specified, `center_by_features` must be a list of integers."
                )
        elif self.rm_center_by_features:
            raise ValueError(
                "`rm_center_by_features` was enabled but `center_by_features` was `None`. "
            )
        if self.scale_by_features is not None:
            if not isinstance(self.scale_by_features, list):
                raise TypeError(
                    "When specified, `scale_by_features` must be a list of integers."
                )
        elif self.rm_scale_by_features:
            raise ValueError(
                "`rm_scale_by_features` was enabled but `scale_by_features` was `None`. "
            )

        self._check_param_arg(
            arg=center,
            valid_options={"mean", "median"},
            arg_name="center",
        )
        self._check_param_arg(
            arg=scale,
            valid_options={"std", "iqr", "mad", "mean", "median", "sum"},
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
        """Required for checks but does nothing. Row scaling is independent per sample.

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

        # Check indices are within range
        for nm, check_indices in [
            ("feature_groups", self.feature_groups),
            ("center_by_features", self.center_by_features),
            ("scale_by_features", self.scale_by_features),
        ]:
            if check_indices is not None:
                all_indices = np.asarray(check_indices).flatten()
                max_idx = all_indices.max()
                min_idx = all_indices.min()
                if max_idx >= self.num_features_:
                    raise ValueError(
                        f"An index in `{nm}` was higher than the number of features in `X`: {max_idx}"
                    )
                if min_idx < 0:
                    raise ValueError(f"An index in `{nm}` was negative: {min_idx}")

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
            X = self._transform_group(
                X_subset=X,
                scale=self.scale,
                center=self.center,
                centering_features=(
                    X[:, self.center_by_features]
                    if self.center_by_features is not None
                    else None
                ),
                scaling_features=(
                    X[:, self.scale_by_features]
                    if self.scale_by_features is not None
                    else None
                ),
            )
        else:
            for group_idx, group_indices in enumerate(self.feature_groups):
                # Get overlapping scaling and centering features
                group_center_by_features = RowScaler._get_group_calc_by_features(
                    group_indices=group_indices,
                    calc_by_features=self.center_by_features,
                )
                group_scale_by_features = RowScaler._get_group_calc_by_features(
                    group_indices=group_indices, calc_by_features=self.scale_by_features
                )

                # When centering/scaling by a single feature
                # it does not make sense center/scale that feature as well
                # since it would just become 0 or 1 all over.
                # So we remove it from the affected features
                # But for >1 features, we also center/scale them!

                x_indices = group_indices
                to_exclude = []
                if (
                    group_center_by_features is not None
                    and len(group_center_by_features) == 1
                ):
                    to_exclude.append(group_center_by_features[0])
                if (
                    group_scale_by_features is not None
                    and len(group_scale_by_features) == 1
                ):
                    to_exclude.append(group_scale_by_features[0])
                if to_exclude:
                    x_indices = np.setdiff1d(
                        x_indices, np.asarray(to_exclude).flatten()
                    )

                # Apply transformations
                X[:, x_indices] = self._transform_group(
                    X_subset=X[:, x_indices],
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
                    centering_features=(
                        X[:, group_center_by_features]
                        if group_center_by_features is not None
                        else None
                    ),
                    scaling_features=(
                        X[:, group_scale_by_features]
                        if group_scale_by_features is not None
                        else None
                    ),
                )

        # Remove the centering and or scaling features
        if self.rm_center_by_features and self.rm_scale_by_features:
            X = np.delete(
                X, np.union1d(self.center_by_features, self.scale_by_features), axis=-1
            )
        elif self.rm_center_by_features:
            X = np.delete(X, self.center_by_features, axis=-1)
        elif self.rm_scale_by_features:
            X = np.delete(X, self.scale_by_features, axis=-1)

        return X

    @staticmethod
    def _get_group_calc_by_features(group_indices, calc_by_features):
        group_calc_by_features = None
        if calc_by_features is not None:
            group_calc_by_features = np.intersect1d(calc_by_features, group_indices)
            if len(group_calc_by_features) == 0:
                group_calc_by_features = None
        return group_calc_by_features

    @staticmethod
    def _transform_group(
        X_subset: np.ndarray,
        scale: Optional[str],
        center: Optional[str],
        centering_features: Optional[np.ndarray],
        scaling_features: Optional[np.ndarray],
    ) -> np.ndarray:
        # Calculate the parameters
        centers, scaling_factors = RowScaler._calculate_parameters(
            X_subset=X_subset,
            scale=scale,
            center=center,
            centering_features=centering_features,
            scaling_features=scaling_features,
        )

        # Apply transformations

        if center is not None:
            if centers is None:
                raise ValueError(
                    "`center` is specified but calculated `centers` were `None`."
                )
            X_subset -= np.expand_dims(centers, axis=-1)

        if scale is not None:
            if scaling_factors is None:
                raise ValueError(
                    "`scale` is specified but calculated `scaling_factors` were `None`."
                )
            X_subset /= np.expand_dims(scaling_factors, axis=-1)

        return X_subset

    @staticmethod
    def _calculate_parameters(
        X_subset: np.ndarray,
        scale: Optional[str],
        center: Optional[str],
        centering_features: Optional[np.ndarray],
        scaling_features: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Calculate scaling and centering parameters.
        """
        centers, scaling_factors = None, None

        if scale is not None:
            if scaling_features is not None:
                scaling_subset = scaling_features
            else:
                scaling_subset = X_subset.copy()

            if scale == "std":
                scaling_factors = np.std(scaling_subset, axis=-1)
            elif scale == "iqr":
                q75, q25 = np.percentile(scaling_subset, [75, 25], axis=-1)
                scaling_factors = q75 - q25
            elif scale == "mad":
                scaling_factors = np.mean(
                    np.abs(scaling_subset - np.mean(scaling_subset, axis=-1)), axis=-1
                )
            elif scale == "mean":
                scaling_factors = np.mean(scaling_subset, axis=-1)
            elif scale == "median":
                scaling_factors = np.median(scaling_subset, axis=-1)
            elif scale == "sum":
                scaling_factors = np.sum(scaling_subset, axis=-1)
            if any(scaling_factors == 0):
                warnings.warn("RowScaler: One or more scaling factors had the value 0.")

        if center is not None:
            if centering_features is not None:
                centering_subset = centering_features
            else:
                centering_subset = X_subset.copy()

            if center == "mean":
                centers = np.mean(centering_subset, axis=-1)
            elif center == "median":
                centers = np.median(centering_subset, axis=-1)

        return centers, scaling_factors

    def _more_tags(self):
        return {"allow_nan": False}
