import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PCAByExplainedVariance(PCA):
    def __init__(self, target_variance: float = 1.0, **kwargs):
        super().__init__(n_components=None, **kwargs)
        self.target_variance = target_variance

    def fit(self, X, y=None):
        # Input validation
        X = check_array(X, ensure_2d=True, copy=True, accept_sparse=False)

        if X.dtype != np.float64:
            X = X.astype(np.float64, copy=True)

        # Save the number of features after aggregation
        self.num_input_features_ = X.shape[1]

        # Fit the PCA model with all components
        super().fit(X)

        if self.target_variance >= 1.0:
            # Get all components
            selected_n_components = self.n_components_
        else:
            # Calculate the cumulative explained variance
            cumulative_variance = np.cumsum(self.explained_variance_ratio_)

            # Find the number of components needed to explain at least p% variance
            # We can max find the number of components found by the super class though (sanity check)
            selected_n_components = min(
                self.n_components_,
                np.argmax(cumulative_variance >= self.target_variance) + 1,
            )

        # Adjust the components to the selected number
        self.n_components_ = selected_n_components
        self.components_ = self.components_[:selected_n_components]
        self.explained_variance_ = self.explained_variance_[:selected_n_components]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[
            :selected_n_components
        ]
        self.singular_values_ = self.singular_values_[:selected_n_components]

        return self

    def transform(self, X):
        # Check if fit has been called
        check_is_fitted(self, "num_input_features_")
        if not X.shape[1] == self.num_input_features_:
            raise ValueError(
                "`X` contained a different number of features than during `.fit()`."
            )

        # Input validation
        X = check_array(X, copy=True, ensure_2d=True, dtype=np.float64)

        return super().transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
