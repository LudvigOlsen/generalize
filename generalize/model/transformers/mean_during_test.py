from typing import Optional, List
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


class MeanDuringTest(BaseEstimator, TransformerMixin):
    def __init__(
        self, feature_indices: Optional[List[int]] = None, training_mode: bool = True
    ) -> None:
        self.feature_indices = feature_indices
        self.training_mode = training_mode

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        # Input validation
        X, y = check_X_y(X, y, ensure_2d=True)
        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=True)
        else:
            X = X.copy()

        self.used_feature_indices_ = self.feature_indices
        if self.feature_indices is None:
            self.used_feature_indices_ = list(range(X.shape[-1]))

        self.num_features_ = X.shape[-1]
        self.means_ = np.expand_dims(
            X[:, self.used_feature_indices_].mean(axis=0), axis=0
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, copy=True, ensure_2d=True, dtype=np.float64)
        if X.shape[-1] != self.num_features_:
            raise ValueError(
                f"Transformer was fitted on {self.num_features_} "
                f"features but `X` contains {X.shape[-1]} features."
            )
        if not self.training_mode:
            X[:, self.used_feature_indices_] = self.means_
        return X
