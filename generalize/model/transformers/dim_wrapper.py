from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# TODO Perhaps find a more telling name?
class DimTransformerWrapper(BaseEstimator, TransformerMixin):
    _required_parameters = ["estimator_class"]

    def __init__(
        self, estimator_class: BaseEstimator, kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Wrapper for using 2D transformers on nD arrays (2D or larger) depending on the
        input data during `.fit()`. When 3D or larger, an estimator is created per index
        combination of the middle dimensions.

        Example
        -------
        >>> # Create pipeline with a standard scaler per feature set (second dimension in 3D data)
        >>> # This now handles both 2D and D>2 arrays
        >>> pipe = Pipeline(steps=[
        >>>     ("scaler_1", DimTransformerWrapper(StandardScaler)),
        >>> ])

        Parameters
        ----------
        estimator_class : uninitialized estimator (transformer) class
            The estimator class to apply to either the entire 2D data or
            to each index combination in the 2 - (n-1) dimensions separately.
        kwargs : dict
            Dict with keyword arguments for initializing the `estimator_class`.
            To set separate argument values for each of the feature sets,
            wrap the argument values in `ArgValueList`, which is a list that
            we can recognize as containing separate argument values.
            For arrays with more than 3 dimensions, you may wish
            to generate the indexings with `generate_loopable_indexings()`
            to know the expected order of the `ArgValueList` elements.
        """
        self.estimator_class = estimator_class
        self.kwargs = kwargs if kwargs is not None else {}

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this wrapper (not sub estimators, see `.get_estimator_params()`).

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
            if deep and hasattr(value, "get_params"):
                if key == "estimator_class":
                    # The estimator class will have the get_params() method
                    # but will never be instantiated under the name `estimator_class`
                    continue
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
        return out

    def get_estimator_params(self):
        """
        Get parameters for the sub estimators.

        Since `get_params()` can only include the parameters
        for the wrapping estimator, this method allows
        inspection of the sub estimators.
        """
        out = dict()
        if hasattr(self, "estimators_"):
            for idx, estimator in enumerate(self.estimators_):
                out[f"estimator_{idx}"] = estimator.get_params()
        elif hasattr(self, "estimator_"):
            out["estimator_0"] = self.estimator_.get_params()
        if "estimator_0" in out:
            out["Note"] = "`estimator_{idx}` is not directly settable."
        return out

    def set_params(self, **params) -> None:
        """
        Set parameters of each estimator.
        """
        try:
            check_is_fitted(self)

            # First update wrapper
            if "estimator_class" in params:
                raise ValueError("`estimator_class` cannot be changed after fitting.")
            self.__dict__.update(params)

            # Then update the sub estimators
            if not self.X_is_2d_:
                for estimator in self.estimators_:
                    estimator.set_params(**params)
            else:
                self.estimator_.set_params(**params)

        except NotFittedError:
            # Update wrapper
            self.__dict__.update(params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        fit_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize and fit the estimator(s) based on the shape of `X`.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, ..., n_features)
            The data used to fit the estimator(s) with.
            When >2D, an estimator is initialized and fitted per index combination
            in the 2 - (n-1) dimensions separately. E.g. when 3D, each
            index in the second dimension has its own estimator.
        y : {array-like} of shape (n_samples) or `None`
            Labels/targets for the samples.
            Some estimators might accept `None` and ignore this argument.
        fit_params : dict
            Keyword arguments for the fit method of the estimator(s).

        Returns
        -------
        self : object
            Wrapper with fitted estimator(s).
        """
        self.ndim_ = X.ndim
        self.shape_ = X.shape
        self.X_is_2d_ = self.ndim_ == 2
        assert self.ndim_ > 1, f"`X` must have 2 or more dimensions but had {X.ndim}."

        self.kwargs_to_get_by_idx_ = [
            key for key, val in self.kwargs.items() if isinstance(val, ArgValueList)
        ]

        # Get the number of feature sets
        # Which is the product of dimension sizes excluding
        # the first and last dimensions
        self.num_feature_sets_ = 1 if self.X_is_2d_ else np.prod(X.shape[1:-1])

        # Check that the keyword arguments with `ArgValueList` values
        # have the same length as the number of feature sets
        if self.kwargs_to_get_by_idx_:
            for kwarg in self.kwargs_to_get_by_idx_:
                if len(self.kwargs[kwarg]) != self.num_feature_sets_:
                    raise ValueError(
                        f"The keyword argument '{kwarg}' was an `ArgValueList` "
                        f"but did not have same length ({len(self.kwargs[kwarg])}) "
                        f"as the number of feature sets ({self.num_feature_sets_})."
                    )

        # Ensure `fit_params` is a dict
        if fit_params is None:
            fit_params = {}
        assert isinstance(fit_params, dict)

        # Initialize and fit estimators
        if not self.X_is_2d_:
            self.estimators_ = [
                self.estimator_class(
                    **_get_kwargs_by_index(
                        self.kwargs,
                        idx=feature_set,
                        non_constant_kwargs=self.kwargs_to_get_by_idx_,
                    )
                ).fit(X=X[indexing].reshape(X.shape[0], X.shape[-1]), y=y, **fit_params)
                for feature_set, indexing in enumerate(
                    generate_loopable_indexings(
                        shape=X.shape,
                        # Excluding first and last dims
                        dims=range(1, self.ndim_ - 1),
                    )
                )
            ]
        else:
            self.estimator_ = self.estimator_class(
                **_get_kwargs_by_index(
                    self.kwargs, idx=0, non_constant_kwargs=self.kwargs_to_get_by_idx_
                )
            ).fit(X=X, y=y, **fit_params)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transformation to data.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, ..., n_features)
            The data to transform.

        Returns
        -------
        `np.ndarray` of shape (n_samples, ..., n_transformed_features)
            Transformed array. The number of features can have been
            altered by the transformation.
        """
        check_is_fitted(self)

        # Avoid altering original X
        X = X.copy()

        if X.shape[1:] != self.shape_[1:]:
            raise ValueError(
                f"Transformer was fitted to an array with shape {self.shape_} "
                f"but `X` has shape {X.shape}. Only the first dimension is allowed to differ."
            )

        if not self.X_is_2d_:
            # Transform each subset of `X` and
            # stack to a 3D array
            X_tr = np.stack(
                [
                    self.estimators_[feature_set].transform(
                        X[indexing].reshape(X.shape[0], X.shape[-1])
                    )
                    for feature_set, indexing in enumerate(
                        generate_loopable_indexings(
                            shape=X.shape, dims=range(1, self.ndim_ - 1)
                        )
                    )
                ],
                axis=-2,
            )

            # If original `X` was >3 dimensions, reshape
            # such that everything but the last dimension
            # (which is allowed to differ post-transformation)
            # is the same as pre-transformation
            if X_tr.ndim != self.ndim_:
                X_tr = X_tr.reshape(X.shape[0], *self.shape_[1:-1], X_tr.shape[-1])

        else:
            X_tr = self.estimator_.transform(X=X)

        return X_tr

    def _more_tags(self):
        """
        These checks are needed when checking the estimator during tests.
        """
        return {"allow_nan": True, "no_validation": True}


class ArgValueList:
    def __init__(self, values: list) -> None:
        """
        Wraps a list of argument values. Allows us to to discern between
        argument values intended to be lists and lists of argument values.

        Parameters
        ----------
        values : list
            List of settings. Each element is a setting for that index.
        """
        self.values = values

    def __getitem__(self, idx: int) -> Any:
        return self.values[idx]

    def __len__(self):
        return len(self.values)


def _get_kwargs_by_index(kwargs: dict, idx: int, non_constant_kwargs: List[str]):
    """
    Get keyword argument for specific index.

    NOTE: Copies `kwargs` to avoid altering the original dict.
    """
    current_kwargs = kwargs.copy()
    if non_constant_kwargs:
        for kwarg in non_constant_kwargs:
            current_kwargs[kwarg] = current_kwargs[kwarg][idx]
    return current_kwargs


def generate_loopable_indexings(
    shape: Union[List[int], Tuple[int]],
    dims: Union[List[int], Tuple[int]],
    include_rest: bool = True,
) -> List[Tuple[slice]]:
    """
    Generator for creating all index combinations for the specified dimensions and
    converting to lists of slices for indexing the original array.

    Parameters
    ----------
    shape : list or tuple
        The shape of the array of which to loop across a set of dimensions.
    dims : list or tuple
        The dimensions to loop across.
    include_rest : bool
        Whether to include slice objects for the dimensions not in `dims`.
        Note that we sort `dims` first, why the slice objects
        order will be the same as the order of the sorted `dims`.

    Yields
    -------
    tuple with slices
        Tuple with slices for indexing the original array with
        index combinations for the specified dimensions.
        The dimensions not in `dims` will use all elements.
    """
    dims = sorted(dims)
    dims_indices = list(np.ndindex(tuple([shape[d] for d in dims])))
    for idx in dims_indices:
        yield _create_indexing(
            idx=idx, dims=dims, shape=shape, include_rest=include_rest
        )


def _create_indexing(
    idx: Tuple[int],
    dims: Union[List[int], Tuple[int]],
    shape: Union[List[int], Tuple[int]],
    include_rest: bool = True,
):
    """
    Create indexing (list of slices) with index combinations
    for the specified dimensions. The dimensions not in `dims`
    will use all elements (except when `include_rest` is `False`).
    """
    idx = list(idx)
    idx_plus_one = [i + 1 for i in idx]
    if include_rest:
        return tuple(
            [
                (
                    slice(0, shape[d])
                    if d not in dims
                    # Could be done with := but that is only py38+
                    else slice(idx.pop(0), idx_plus_one.pop(0))
                )
                for d in range(len(shape))
            ]
        )
    return tuple([slice(i, i + 1) for i in idx])
