from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


class BaseTruncateExtremes(TransformerMixin, BaseEstimator, ABC):
    def __init__(
        self,
        by="all",
        copy: bool = True,
    ) -> None:
        """
        Base class for truncating extreme values.

        The subclass defines the method for calculating the value limits.
        Value limits can be found for either all data, per column or row-wise.

        Parameters
        ----------
        by
            Whether to extract value limits from all data,
            or separately per column or row.
            One of {'all', 'cols', 'rows'}.
            When `rows`, the value limits are extracted in `.fit()`.
        copy
            Whether to copy the input data during the
            `.transform()` method.
        """
        self.by = by
        self.copy = copy

        assert by in ["all", "rows", "cols"]

    def fit(self, X: Optional[np.ndarray], y: Optional[np.ndarray] = None):
        """Fit the transformer. Extracts the value limits.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features). Ignored.
        y : None
            Ignored.

        Returns
        -------
        self : object
        """
        # Reset internal state before fitting
        X, y = check_X_y(X, y, copy=False)
        self.num_features_ = X.shape[-1]
        if self.by == "all":
            self.lower_lim_, self.upper_lim_ = self.extract_limits(X)
        elif self.by == "cols":
            self.lower_lims_ = []
            self.upper_lims_ = []
            # TODO: Can't this be vectorized?
            for i in range(self.num_features_):
                lower_lim_, upper_lim_ = self.extract_limits(X[:, i])
                if lower_lim_ is not None:
                    self.lower_lims_.append(lower_lim_)
                if upper_lim_ is not None:
                    self.upper_lims_.append(upper_lim_)
        elif self.by == "rows":
            self.lower_lim_ = None
            self.upper_lim_ = None
        return self

    @abstractmethod
    def extract_limits(self, X: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Method for extracting value limits with the
        method decided by the subclass.
        """
        pass

    def transform(self, X: np.ndarray, copy: Optional[bool] = None):
        """Apply truncation based on fitted value limits.

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
        if self.by == "all":
            if self.lower_lim_ is not None:
                X[X < self.lower_lim_] = self.lower_lim_
            if self.upper_lim_ is not None:
                X[X > self.upper_lim_] = self.upper_lim_
        elif self.by == "cols":
            # TODO: Can't this be vectorized?
            for i in range(self.num_features_):
                if self.lower_lims_:
                    X[:, i][X[:, i] < self.lower_lims_[i]] = self.lower_lims_[i]
                if self.upper_lims_:
                    X[:, i][X[:, i] > self.upper_lims_[i]] = self.upper_lims_[i]
        elif self.by == "rows":
            # TODO: Can't this be vectorized?
            for i in range(X.shape[0]):
                lower_lim_, upper_lim_ = self.extract_limits(X[i, :])
                if lower_lim_ is not None:
                    X[i, :][X[i, :] < lower_lim_] = lower_lim_
                if upper_lim_ is not None:
                    X[i, :][X[i, :] > upper_lim_] = upper_lim_
        return X

    def _more_tags(self):
        return {"allow_nan": False}


class TruncateExtremesByPercentiles(BaseTruncateExtremes):
    def __init__(
        self,
        lower_percentile: Optional[float] = None,
        upper_percentile: Optional[float] = None,
        by="all",
        copy: bool = True,
    ):
        """
        Truncate values that are beyond a percentile in the fit data.

        Value limits can be found for either all data, per column or row-wise.

        Parameters
        ----------
        lower_percentile
            Percentile (0.-100.) to use as lower value limit.
            When `None`, no lower truncation is performed.
        upper_percentile
            Percentile (0.-100.) to use as upper value limit.
            When `None`, no upper truncation is performed.
        by
            Whether to extract value limits from all data,
            or separately per column or row.
            One of {'all', 'cols', 'rows'}.
            When `rows`, the value limits are extracted in `.fit()`.
        copy
            Whether to copy the input data during the
            `.transform()` method.
        """
        super().__init__(by=by, copy=copy)
        assert lower_percentile is None or lower_percentile >= 0.0
        assert upper_percentile is None or upper_percentile <= 100.0
        if (
            lower_percentile is not None
            and upper_percentile is not None
            and lower_percentile >= upper_percentile
        ):
            raise ValueError(
                "`lower_percentile` must be lower than "
                "`upper_percentile` (when both are specified)."
            )
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def extract_limits(self, X: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts value limits as percentile(s) based on
        `lower_percentile` and `upper_percentile`.

        Outputs may be `None`.
        """
        lower_lim_, upper_lim_ = (None, None)
        if self.lower_percentile is not None and self.upper_percentile is not None:
            lower_lim_, upper_lim_ = np.percentile(
                X, [self.lower_percentile, self.upper_percentile]
            )
        elif self.lower_percentile is not None:
            lower_lim_ = np.percentile(X, [self.lower_percentile])
        elif self.upper_percentile is not None:
            upper_lim_ = np.percentile(X, [self.upper_percentile])
        return lower_lim_, upper_lim_


class TruncateExtremesByIQR(BaseTruncateExtremes):
    def __init__(
        self,
        lower_num_iqr: Optional[float] = None,
        upper_num_iqr: Optional[float] = None,
        by="all",
        copy: bool = True,
    ):
        """
        Truncate values that are beyond a percentile in the fit data.

        Value limits can be found for either all data, per column or row-wise.

        Parameters
        ----------
        lower_num_iqr
            Number of IQRs below Q1 to consider the value limit.
            A good starting point may be 1.5.
            When `None`, no lower truncation is performed.
        upper_num_iqr
            Number of IQRs above Q3 to consider the value limit.
            A good starting point may be 1.5.
            When `None`, no upper truncation is performed.
        by
            Whether to extract value limits from all data,
            or separately per column or row.
            One of {'all', 'cols', 'rows'}.
            When `rows`, the value limits are extracted in `.fit()`.
        copy
            Whether to copy the input data during the
            `.transform()` method.
        """
        super().__init__(by=by, copy=copy)
        assert lower_num_iqr is None or lower_num_iqr >= 0
        assert upper_num_iqr is None or upper_num_iqr >= 0
        self.lower_num_iqr = lower_num_iqr
        self.upper_num_iqr = upper_num_iqr

    def extract_limits(self, X: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts value limits as a number of IQRs from Q1 and Q3,
        based on `lower_num_iqr` and `upper_num_iqr`.
        For the lower limit (when specified), the limit becomes `q1 - lower_num_iqr * iqr`.
        For the upper limit (when specified), the limit becomes `q3 + upper_num_iqr * iqr`.

        Outputs may be `None`.
        """
        q3, q1 = np.percentile(X, [75, 25])
        iqr = q3 - q1
        lower_lim_ = (
            q1 - self.lower_num_iqr * iqr if self.lower_num_iqr is not None else None
        )
        upper_lim_ = (
            q3 + self.upper_num_iqr * iqr if self.upper_num_iqr is not None else None
        )
        return lower_lim_, upper_lim_


class TruncateExtremesBySTD(BaseTruncateExtremes):
    def __init__(
        self,
        lower_num_std: Optional[float] = None,
        upper_num_std: Optional[float] = None,
        by="all",
        copy: bool = True,
    ):
        """
        Truncate values that are beyond a percentile in the fit data.

        Value limits can be found for either all data, per column or row-wise.

        Parameters
        ----------
        lower_num_std
            Number of standard deviations from the mean to consider the value limit.
            When `None`, no lower truncation is performed.
        upper_num_std
            Number of standard deviations from the mean to consider the value limit.
            When `None`, no upper truncation is performed.
        by
            Whether to extract value limits from all data,
            or separately per column or row.
            One of {'all', 'cols', 'rows'}.
            When `rows`, the value limits are extracted in `.fit()`.
        copy
            Whether to copy the input data during the
            `.transform()` method.
        """
        super().__init__(by=by, copy=copy)
        assert lower_num_std is None or lower_num_std > 0
        assert upper_num_std is None or upper_num_std > 0
        self.lower_num_std = lower_num_std
        self.upper_num_std = upper_num_std

    def extract_limits(self, X: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Extracts value limits as a number of standard deviations from the mean,
        based on `lower_num_std` and `upper_num_std`.
        For the lower limit (when specified), the limit becomes `mean - lower_num_std * iqr`.
        For the upper limit (when specified), the limit becomes `mean + upper_num_std * iqr`.

        Outputs may be `None`.
        """
        mean = np.mean(X)
        std = np.std(X)
        lower_lim_ = (
            mean - self.lower_num_std * std if self.lower_num_std is not None else None
        )
        upper_lim_ = (
            mean + self.upper_num_std * std if self.upper_num_std is not None else None
        )
        return lower_lim_, upper_lim_
