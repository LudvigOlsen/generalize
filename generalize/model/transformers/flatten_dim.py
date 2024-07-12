

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import FunctionTransformer


class FlattenFeatureSetsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for flattening the feature sets dimension.
    This is the second last dimension when the dataset is a 3+d `numpy.ndarray`.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, inv_num_features: Optional[int] = None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, ..., n_feature_sets, n_features)
            The data to flatten the last two dimensions of.
        y : {array-like} of shape (n_samples) or `None`
            Ignored.
        inv_num_features : int or `None`
            The number of features to use for the inverse reshaping in `.inverse_transform()`. 
            When `None`, this is inferred from `X`.
        """
        self.inv_num_features_ = inv_num_features
        if self.inv_num_features_ is None:
            self.inv_num_features_ = X.shape[-1]

        return self

    def transform(self, X: np.ndarray):
        return _flatten_feature_sets(X)

    def inverse_transform(self, X: np.ndarray):
        return _reshape_to_feature_sets(X, self.inv_num_features_)


class UnflattenFeatureSetsTransformer(FunctionTransformer):

    def __init__(self, num_features: int):
        """
        FunctionTransformer for unflattening the features dimension into feature sets and features dimensions.
        The last dimension is reshaped into two dimensions.

        Parameters
        ----------
        num_features : int
            The number of features to have for each feature set after unflattening  
            (i.e. desired size of the last dimension).
            Should be divisible with the current number of features. 
            The new feature sets dimension will have size 
            `current_num_features / desired_num_features`.
        """
        super().__init__(
            func=_flatten_feature_sets,
            inverse_func=_reshape_to_feature_sets,
            inv_kw_args={"num_features": num_features})


def _flatten_feature_sets(x):
    return x.reshape(*x.shape[0:-2], -1)


def _reshape_to_feature_sets(x, num_features):
    return x.reshape(*x.shape[0:-1], -1, num_features)
