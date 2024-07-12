
import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted


class CorrelatedFeatureRemover(BaseEstimator, SelectorMixin):

    def __init__(self, threshold=0.9, select_by="avg_correlation_fast"):
        """
        A feature selector for removing correlated features.

        In order to reduce pairwise comparisons, we pre-sort the features by
            a) the number of correlations >= the threshold
            b) the greatest average absolute correlation to all other features

        Parameters
        ----------
        threshold: float
            The threshold at which (>=) one of the the features is randomly removed.
        select_by: str
            Method to select between features. One of:
                "random": 
                    Randomly select a feature
                "avg_correlation_fast":
                    Select the feature with the lowest average absolute correlation
                    to all the other features. In the 'fast' mode, the average
                    absolute correlation is calculated at once.
                "avg_correlation_exact": 
                    Select the feature with the lowest average absolute correlation
                    to all the other *remaining* features. In the 'exact' version,
                    we don't include the already removed features when calculating
                    the average absolute correlation. This is likely a bit 
                    slower but may lead to better results.
            The conceptual idea for this is taken from the `caret` R package,
            although their implementation may differ:
                https://github.com/topepo/caret/blob/0579b5dded32f5af9c4c6f4ef5a3898d3839c4d0/pkg/caret/R/findCorrelation.R
        """
        assert select_by in [
            "random", "avg_correlation_exact", "avg_correlation_fast"]
        self.threshold = threshold
        self.select_by = select_by

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if X.shape[0] < 2:
            raise ValueError("`X` contained only 1 sample but must have more than 1 samples.")
        if X.shape[1] < 2:
            # Single feature case
            self.remove_indices_ = []
            self.num_original_features_ = 1
            return self
        
        # List of indices to remove
        remove_indices_ = []
        self.num_original_features_ = X.shape[1]

        # Calculate the (absolute) correlation matrix
        abs_corr = np.abs(np.corrcoef(X, rowvar=False))

        # Calculate the average absolute correlations for each feature
        avg_abs_correlations = np.mean(abs_corr, axis=1)

        # Get the indices of features with at least one `>= threshold` correlations
        # sorted by a) the number of such too high correlations, and
        # b) the average absolute correlation to all other features
        sorting_indices = CorrelatedFeatureRemover._preorder_feature_indices(
            abs_corr=abs_corr,
            avg_abs_correlations=avg_abs_correlations,
            threshold=self.threshold
        )

        # Find the indices of correlated features to remove
        # Note that `sorting_indices` may have fewer indices 
        # than the total number of features
        for i in range(sorting_indices.size):
            for j in range(i + 1, sorting_indices.size):
                i_idx = sorting_indices[i]
                j_idx = sorting_indices[j]
                if (abs_corr[i_idx, j_idx] >= self.threshold
                    and i_idx not in remove_indices_
                        and j_idx not in remove_indices_):
                    remove_indices_.append(
                        CorrelatedFeatureRemover._select_to_remove(
                            abs_corr=abs_corr,
                            i=i_idx,
                            j=j_idx,
                            remove_indices=remove_indices_,
                            select_by=self.select_by,
                            avg_abs_correlations=avg_abs_correlations
                        )
                    )

        # Assign to self
        self.remove_indices_ = remove_indices_

        # Return the fitted transformer
        return self

    def _get_support_mask(self):
        """
        Used by `SelectorMixin` to provide the `transform()` method.
        """
        check_is_fitted(self)

        mask = np.full((self.num_original_features_,), True, dtype=bool)
        mask[self.remove_indices_] = False
        return mask

    @staticmethod
    def _preorder_feature_indices(abs_corr, avg_abs_correlations, threshold):

        # Count how many features each feature correlate too much with
        too_correlated_counts = np.sum(abs_corr >= threshold, axis=1)

        # Find the indices that sorts by greatest:
        #   a) count of too correlated features
        #   b) avg absolute correlation to all other features
        sorting_indices = np.flip(np.argsort(
            too_correlated_counts + avg_abs_correlations, 
            kind="stable"
        ))

        # Count how many features don't have any too high correlations
        num_never_too_correlated = np.argwhere(
            too_correlated_counts < 1.0).size

        # Remove those that are never too correlated, as
        # we don't need to check those
        if num_never_too_correlated > 0:
            # NOTE: indexing with [:-0] removes all elements
            sorting_indices = sorting_indices[:-num_never_too_correlated]

        return sorting_indices

    @staticmethod
    def _select_to_remove(i, j, select_by, abs_corr=None, remove_indices=None, avg_abs_correlations=None):
        """
        Chooses which of the two features to remove based on the `select_by` method.
        """
        if select_by == "random":
            return i if random.uniform(0, 1) < 0.5 else j
        else:
            if select_by == "avg_correlation_fast":
                # Since we've sorted by the average absolute correlations
                # `i`'s avg abs corr will be >= `j`'s, but when they have the same
                # we want to pick randomly, so we don't just return i
                avg_corr_i = avg_abs_correlations[i]
                avg_corr_j = avg_abs_correlations[j]

            else:
                # Exact method, where we only get the avg for the currently remaining columns
                # Compute the average correlation of each feature with all of the other features,
                # excluding the already removed features
                non_removed_indices = np.setdiff1d(
                    np.arange(abs_corr.shape[1]), remove_indices)
                avg_corr_i = np.mean(np.abs(abs_corr[i, non_removed_indices]))
                avg_corr_j = np.mean(np.abs(abs_corr[j, non_removed_indices]))

            if avg_corr_i > avg_corr_j:
                return i
            elif avg_corr_i < avg_corr_j:
                return j
            else:
                return CorrelatedFeatureRemover._select_to_remove(
                    i=i,
                    j=j,
                    select_by="random"
                )
