from typing import List
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
import numpy as np


class BaseIndexSelector(BaseEstimator, SelectorMixin):
    _required_parameters = ["feature_indices"]

    def __init__(self, feature_indices: List[int]):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        self.total_num_features_ = X.shape[-1]
        return self


class IndexFeatureSelector(BaseIndexSelector):
    def _get_support_mask(self):
        check_is_fitted(self)
        mask = np.full((self.total_num_features_), False)
        mask[self.feature_indices] = True
        return mask


class IndexFeatureRemover(BaseIndexSelector):
    def _get_support_mask(self):
        check_is_fitted(self)
        mask = np.full((self.total_num_features_), True)
        mask[self.feature_indices] = False
        return mask
