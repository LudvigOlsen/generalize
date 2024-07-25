"""
For handling a collection of ROC curves.
"""

import json
import pathlib
import copy
from typing import Callable, Dict, List, Optional, Union

# from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as auc_from_xy
from nattrs import nested_setattr, nested_hasattr, nested_getattr, nested_mutattr

from generalize.evaluate.confusion_matrices import _dict_to_str
from generalize.evaluate.prepare_inputs import BinaryPreparer

# TODO: Add method for finding threshold given a sensitivity


# @dataclass
# class ROCCurveDifferences:
#     """
#     Differences between two `ROCCurve` objects.

#     Parameters
#     ----------
#     fpr_diff
#         numpy.ndarray
#             Differences in False Positive Rates
#             I.e. `other.fpr - self.fpr` after interpolation.
#     tpr_diff
#         numpy.ndarray
#             Differences in True Positive Rates
#             I.e. `other.tpr - self.tpr` after interpolation.
#     thresholds
#         numpy.ndarray
#             Thresholds used for interpolation.
#             Defined as `np.linspace(0, 1, 1001)`.
#     auc_diff
#         float or None
#             Difference in AUC scores
#     """

#     fpr_diff: float
#     tpr_diff: float
#     thresholds: np.ndarray
#     auc_diff: float


class ROCCurve:
    def __init__(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray,
        auc: Optional[float] = None,
    ) -> None:
        """
        Container for Receiver Operating Characteristic (ROC) curve values.

        E.g. as calculated with `sklearn.metrics.roc_curve()`.

        Parameters
        ----------
        fpr : `numpy.ndarray`
            Increasing false positive rates such that element `i` is the
            false positive rate of predictions with `score >= thresholds[i]`.
        tpr : `numpy.ndarray`
            Increasing true positive rates such that element `i` is the
            true positive rate of predictions with `score >= thresholds[i]`.
        thresholds : `numpy.ndarray`
            Decreasing thresholds on the decision function used to compute fpr and tpr.
            `thresholds[0]` represents no instances being predicted and is arbitrarily
            set to `max(predicted_probabilities) + 1`.
        auc : float or `None`
            Area Under the Curve of this ROC curve.
        """
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
        self.auc = auc

        if not ROCCurve._check_array_order(fpr):
            raise ValueError("`fpr` must be sorted in ascending order.")
        if not ROCCurve._check_array_order(tpr):
            raise ValueError("`tpr` must be sorted in ascending order.")
        if not ROCCurve._check_array_order(thresholds, desc=True):
            raise ValueError("`thresholds` must be sorted in descending order.")

    def recalculate_auc(self):
        """
        Calculates AUC from `fpr` and `tpr` with `from sklearn.metrics import auc`.
        Overwrites the `self.auc` attribute.
        """
        self.auc = auc_from_xy(self.fpr, self.tpr)
        return self

    @staticmethod
    def _check_array_order(x, desc: bool = False) -> bool:
        if desc:
            return np.array_equal(x, np.sort(x)[::-1])
        return np.array_equal(x, np.sort(x))

    @staticmethod
    def from_data(
        targets: Union[list, np.ndarray],
        predicted_probabilities: Union[list, np.ndarray],
        sample_weight: Optional[Union[list, np.ndarray]] = None,
        positive: Union[int, str] = 1,
    ):
        """
        Create a ROC curve from a set of binary targets and probabilities.

        Wraps `sklearn.metrics.roc_curve()` and `sklearn.metrics.roc_auc_score()`.

        Parameters
        ----------
        targets : list or `numpy.ndarray`
            The true values.
        predicted_probabilities : list or `numpy.ndarray`
            The predicted probabilities.
        sample_weight : list or `numpy.ndarray` or `None`
            Sample weights.

        Returns
        -------
        ROCCurve
            A ROC curve object.
        """
        # Ensure format of targets
        targets = BinaryPreparer.prepare_targets(targets=targets)

        # Ensure probabilities are of second class
        predicted_probabilities = BinaryPreparer.prepare_probabilities(
            probabilities=predicted_probabilities
        )

        # Ensure weights have the right format
        sample_weight = BinaryPreparer.prepare_probabilities(
            probabilities=sample_weight
        )

        auc = roc_auc_score(
            y_true=targets, y_score=predicted_probabilities, sample_weight=sample_weight
        )

        roc = roc_curve(
            y_true=targets,
            y_score=predicted_probabilities,
            pos_label=positive,
            sample_weight=sample_weight,
        )

        return ROCCurve(fpr=roc[0], tpr=roc[1], thresholds=roc[2], auc=auc)

    def to_dict(self, copy: bool = False) -> Dict[str, Union[list, np.ndarray]]:
        """
        Convert `ROCCurve` object to a dictionary.

        Parameters
        ----------
        copy : bool
            Whether to make a copy of the values before
            assigning them to a dictionary.

        Returns
        -------
        dict
            Dictionary with the keys:
                `{'AUC', 'FPR', 'TPR', 'Thresholds'}`
        """
        get_from = self
        if copy:
            get_from = self.__copy__()
        return {
            "AUC": get_from.auc,
            "FPR": get_from.fpr,
            "TPR": get_from.tpr,
            "Thresholds": get_from.thresholds,
        }

    @staticmethod
    def from_dict(curve_dict, dtype: Optional[npt.DTypeLike] = None):
        """
        Create `ROCCurve` object from dictionary.

        Parameters
        ----------
        curve_dict : dict
            Dictionary with the following keys:
                `{'FPR', 'TPR', 'Thresholds', 'AUC' (optional)}`.
            `FPR`, `TPR`, and `Thresholds` elements must be array-like.
            When present, `AUC` must be a float or `numpy float` scalar or `None`.
        dtype : numpy.dtype
            Data type to enforce on the list/array values.
            Passed to `numpy.asarray(..., dtype=dtype)`.

        Returns
        -------
        ROCCurve
            A `ROCCurve` object with the values from the input dict.
        """
        return ROCCurve(
            fpr=np.asarray(curve_dict["FPR"], dtype=dtype),
            tpr=np.asarray(curve_dict["TPR"], dtype=dtype),
            thresholds=np.asarray(curve_dict["Thresholds"], dtype=dtype),
            auc=curve_dict.get("AUC", None),
        )

    # TODO: Unclear whether interpolating to thresholds
    #       gives meaningful representations of the two curves
    #       such that the difference is also meaningful
    # def __sub__(self, other) -> ROCCurveDifferences:
    #     """
    #     Subtract a ROC curve (TPR and FPR separately) from another.
    #     Applied linear interpolation to both curves and
    #     subtracts `other.fpr` from `self.fpr` and
    #     `other.tpr` from `self.tpr`.

    #     The outputs can be used to plot the difference
    #     in expected FPR and TPR at given thresholds
    #     for two ROC curves.

    #     Parameters
    #     ----------
    #     other: ROCCurve
    #         ROC curve to subtract from `self`.

    #     Returns
    #     -------
    #     ROCCurveDifferences
    #         With:
    #             numpy.ndarray
    #                 Differences in False Positive Rates
    #                 I.e. `other.fpr - self.fpr` after interpolation.
    #             numpy.ndarray
    #                 Differences in True Positive Rates
    #                 I.e. `other.tpr - self.tpr` after interpolation.
    #             numpy.ndarray
    #                 Thresholds used for interpolation.
    #                 Defined as `np.linspace(0, 1, 1001)`.
    #             float or None
    #                 Difference in AUC scores

    #     """
    #     num_points = 1001
    #     interpolated_roc_1 = self.interpolate(to=num_points, reference="thresholds")
    #     interpolated_roc_2 = other.interpolate(to=num_points, reference="thresholds")

    #     auc_diff = None
    #     if self.auc is not None and other.auc is not None:
    #         auc_diff = self.auc - other.auc

    #     return ROCCurveDifferences(
    #         fpr_diff=interpolated_roc_1.fpr - interpolated_roc_2.fpr,
    #         tpr_diff=interpolated_roc_1.tpr - interpolated_roc_2.tpr,
    #         thresholds=interpolated_roc_1.thresholds,
    #         auc_diff=auc_diff,
    #     )

    def interpolate(
        self,
        to: Union[int, List[float], np.ndarray] = 1001,
        reference: str = "fpr",
    ):
        """
        Apply linear interpolation.

        Parameters
        ----------
        to : int, list of floats, optional
            The grid to interpolate to (i.e. new values of `reference` array).
            Either as a number of points between 0-1,
            or a list of points (will be sorted in
            ascending order).
        reference : str, optional
            Reference for interpolation ('fpr', 'tpr', or 'thresholds'), by default 'fpr'.
            NOTE: thresholds don't seem to reproduce AUC scores
                  failed to find a good source on this.

        Returns
        -------
        ROCCurve
            A `ROCCurve` object with interpolated values.

        """
        reference = reference.lower()
        assert reference in ["fpr", "tpr", "thresholds"]
        # Define a common set of thresholds (equally spaced)

        if isinstance(to, int):
            to = np.linspace(0, 1, to)
        else:
            to = np.sort(to)

        if reference == "fpr":
            interpolated_tpr = np.interp(to, self.fpr, self.tpr)
            interpolated_thresholds = np.interp(to, self.fpr, self.thresholds)
            interpolated_auc = auc_from_xy(to, interpolated_tpr)
            return ROCCurve(
                fpr=to,
                tpr=interpolated_tpr,
                thresholds=interpolated_thresholds,
                auc=interpolated_auc,
            )

        elif reference == "tpr":
            interpolated_fpr = np.interp(to, self.tpr, self.fpr)
            interpolated_thresholds = np.interp(to, self.tpr, self.thresholds)
            interpolated_auc = auc_from_xy(interpolated_fpr, to)
            return ROCCurve(
                fpr=interpolated_fpr,
                tpr=to,
                thresholds=interpolated_thresholds,
                auc=interpolated_auc,
            )

        elif reference == "thresholds":
            interpolated_tpr = np.interp(to, self.thresholds[::-1], self.tpr[::-1])[
                ::-1
            ]
            interpolated_fpr = np.interp(to, self.thresholds[::-1], self.fpr[::-1])[
                ::-1
            ]
            interpolated_auc = auc_from_xy(interpolated_fpr, interpolated_tpr)
            return ROCCurve(
                fpr=interpolated_fpr,
                tpr=interpolated_tpr,
                thresholds=to[::-1],
                auc=interpolated_auc,
            )

    def __copy__(self):
        return ROCCurve(
            fpr=self.fpr.copy(),
            tpr=self.tpr.copy(),
            thresholds=self.thresholds.copy(),
            auc=self.auc,
        )

    def to_lists(self, copy: bool = True):
        """
        Convert all arrays to lists.

        Always returns a `ROCCurve` instance,
        so methods can be chained.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with the conversions
            or to convert the attributes of the existing object.

        Returns
        -------
        ROCCurve
            A roc curve instance.
            When `copy` is enabled, this is a copy of the original
            ROCCurve instance.
        """
        if isinstance(self.fpr, list):
            raise ValueError("The attributes are already lists.")
        if not hasattr(self.fpr, "tolist"):
            raise ValueError(
                "The attributes did not have the expected `.tolist()` method."
            )
        out = self if not copy else self.__copy__()
        out.fpr = out.fpr.tolist()
        out.tpr = out.tpr.tolist()
        out.thresholds = out.thresholds.tolist()
        return out

    def to_ndarrays(self, copy: bool = True, dtype: Optional[npt.DTypeLike] = None):
        """
        Convert all lists to `numpy.ndarray`s.

        **When to use**:
        Should only be used after arrays have been converted to lists
        (see `.to_lists()`).

        Always returns a `ROCCurve` instance,
        so methods can be chained.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with the conversions
            or to convert the attributes of the existing object.
        dtype : numpy dtype or `None`
            An optional dtype to convert the arrays to.

        Returns
        -------
        ROCCurve
            A roc curve instance.
            When `copy` is enabled, this is a copy of the original
            ROCCurve instance.
        """
        out = self if not copy else self.__copy__()
        out.fpr = np.asarray(out.fpr, dtype=dtype)
        out.tpr = np.asarray(out.tpr, dtype=dtype)
        out.thresholds = np.asarray(out.thresholds, dtype=dtype)
        return out

    @staticmethod
    def _check_param_bounds(x: float, x_name: str, min_=0.0, max_=1.0):
        if x < min_ or x > max_:
            raise ValueError(
                f"`{x_name}` must be between {min_} and {max_} "
                f"(including bounds) but was: {x}."
            )

    def get_threshold_at_specificity(
        self, above_specificity: float = 0.95, interpolate: bool = False
    ) -> Dict[str, float]:
        """
        Find first threshold and sensitivity where specificity is `> above_specificity`
        or use interpolation to get the interpolated threshold at `above_specificity`.

        Parameters
        ----------
        above_specificity : float
            Specificity above which to find the first threshold from a ROC curve.
        interpolate : bool
            Whether to use interpolation to get the threshold.
            Interpolates relative to the False Positive Rate (i.e. 1-specificity)
            and gets the interpolated threshold at the exact specificity
            (i.e. `above_specificity`).

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """

        ROCCurve._check_param_bounds(x=above_specificity, x_name="above_specificity")

        if interpolate:
            interpolated_roc = self.interpolate(
                to=[0, 1 - above_specificity, 1], reference="fpr"
            )
            return {
                "Threshold": interpolated_roc.thresholds[1],
                "Specificity": above_specificity,
                "Sensitivity": interpolated_roc.tpr[1],
            }

        else:
            specificities = 1 - self.fpr

            # Find first threshold where specifity is above `above_specificity`
            threshold_idx = np.argwhere(specificities > above_specificity)[-1][0]

            return {
                "Threshold": self.thresholds[threshold_idx],
                "Specificity": 1 - self.fpr[threshold_idx],
                "Sensitivity": self.tpr[threshold_idx],
            }

    def get_threshold_at_sensitivity(
        self, above_sensitivity: float = 0.95, interpolate: bool = False
    ) -> Dict[str, float]:
        """
        Find first threshold and specificity where sensitivity is `> above_sensitivity`
        or use interpolation to get the interpolated threshold at `above_sensitivity`.

        Parameters
        ----------
        above_sensitivity : float
            Sensitivity above which to find the first threshold from a ROC curve.
        interpolate : bool
            Whether to use interpolation to get the threshold.
            Interpolates relative to the sensitivities and gets the
            interpolated threshold at the exact sensitivity
            (i.e. `above_sensitivity`).

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """
        ROCCurve._check_param_bounds(x=above_sensitivity, x_name="above_sensitivity")

        if interpolate:
            interpolated_roc = self.interpolate(
                to=[0, above_sensitivity, 1], reference="tpr"
            )
            return {
                "Threshold": interpolated_roc.thresholds[1],
                "Specificity": interpolated_roc.fpr[1],
                "Sensitivity": above_sensitivity,
            }

        else:
            sensitivities = self.tpr

            # Find first threshold where specifity is above `above_specificity`
            threshold_idx = np.argwhere(sensitivities > above_sensitivity)[0][0]

            return {
                "Threshold": self.thresholds[threshold_idx],
                "Specificity": 1 - self.fpr[threshold_idx],
                "Sensitivity": self.tpr[threshold_idx],
            }

    def get_nearest_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Find the nearest existing threshold.

        Parameters
        ----------
        threshold : float
            Threshold to find nearest threshold of from a ROC curve.

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """

        ROCCurve._check_param_bounds(x=threshold, x_name="threshold")

        # Find first threshold where specifity is above `above_specificity`
        threshold_idx = np.abs(self.thresholds - threshold).argmin()

        return {
            "Threshold": self.thresholds[threshold_idx],
            "Specificity": 1 - self.fpr[threshold_idx],
            "Sensitivity": self.tpr[threshold_idx],
        }

    def get_interpolated_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Get the specificity and sensitivity (and threshold) for a specific
        threshold by interpolating relative to the thresholds.

        Parameters
        ----------
        threshold : float
            Threshold to find via interpolation.

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """
        ROCCurve._check_param_bounds(x=threshold, x_name="threshold")

        interpolated_roc = self.interpolate(
            to=[0, threshold, 1], reference="thresholds"
        )
        return {
            "Threshold": threshold,
            "Specificity": interpolated_roc.fpr[1],
            "Sensitivity": interpolated_roc.tpr[1],
        }

    def get_threshold_at_max_j(self, interpolate: bool = False) -> Dict[str, float]:
        """
        Find first threshold where Youden's J statistic is at its max.

        Youden's J statistic is defined as
            `J = sensitivity + Specificity - 1`
        and can thus be written as:
            `J = TPR - FPR`

        Parameters
        ----------
        interpolate : bool
            Whether to use interpolation to get the threshold.
            Interpolates relative to the False Positive Rate to
            get a (possibly higher) resolution of `1001` points
            (between 0 and 1) and gets the interpolated threshold
            at the maximum Youden's J. The idea is, that this
            might get a more precise expected threshold to use
            on new data.

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """
        if interpolate:
            interpolated_roc = self.interpolate(reference="fpr")
            return ROCCurve._get_threshold_at_max_j(interpolated_roc)
        else:
            return ROCCurve._get_threshold_at_max_j(self)

    @staticmethod
    def _get_threshold_at_max_j(roc) -> Dict[str, float]:
        """
        Find first threshold where Youden's J statistic is at its max.

        Youden's J statistic is defined as
            `J = sensitivity + Specificity - 1`
        and can thus be written as:
            `J = TPR - FPR`

        Returns
        -------
            dict
                Dictionary with threshold, specificity, and sensitivity.
        """

        # Calculate Youden's J statistic
        j = roc.tpr - roc.fpr

        # Get lowest threshold with max J
        threshold_idx = sorted(np.where(j == np.max(j))[0])[-1]

        return {
            "Threshold": roc.thresholds[threshold_idx],
            "Specificity": 1 - roc.fpr[threshold_idx],
            "Sensitivity": roc.tpr[threshold_idx],
        }

    def to_printable_dict(
        self, max_array_elems: Optional[int] = 3, decimals: Optional[int] = 4
    ) -> dict:
        """
        Convert to a printable dictionary. NOTE: May not contain all values.

        Parameters
        ----------
        max_array_elems : int or None
            How many elements to keep per array. When too many,
            a final element with "..." is added to indicate
            that not all elements are shown.
        decimals : int
            Decimals to round values to.

        Returns
        -------
        dict
            Dictionary with formatted elements, ready for printing.
        """
        d = self.to_dict(copy=True)
        return ROCCurve.make_dict_printable(
            d=d, max_array_elems=max_array_elems, decimals=decimals
        )

    @staticmethod
    def make_dict_printable(
        d: dict, max_array_elems: Optional[int] = 3, decimals: Optional[int] = 4
    ) -> dict:
        """
        Convert a dictionary version of a `ROCCurve` object to a
        printable dictionary. NOTE: May not contain all values.

        Parameters
        ----------
        d : dict
            Dictionary with the following keys:
            `{'FPR', 'TPR', 'Thresholds', 'AUC' (optional)}`
        max_array_elems : int or None
            How many elements to keep per array. When too many,
            a final element with "..." is added to indicate
            that not all elements are shown.
        decimals : int
            Decimals to round values to.

        Returns
        -------
        dict
            Dictionary with formatted elements, ready for printing.
        """

        def formatter(arr):
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            too_long = max_array_elems is not None and len(arr) > max_array_elems
            if too_long:
                arr = arr[:max_array_elems]
            # arr = np.round(arr, decimals=decimals)
            arr = arr.tolist()
            # Converting to list after np.round introduces
            # new rounding error decimals for some reason
            # so we apply round on the list instead
            if decimals is not None:
                arr = [round(x, decimals) for x in arr]
            if too_long:
                arr.append("...")
            return arr

        # Add to new dict with array lengths in the keys
        new_d = {}
        if "AUC" in d and d["AUC"] is not None:
            new_d["AUC"] = np.round(d["AUC"], decimals=decimals)
        new_d[f"FPR ({len(d['FPR'])})"] = formatter(d["FPR"])
        new_d[f"TPR ({len(d['TPR'])})"] = formatter(d["TPR"])
        new_d[f"Thresholds ({len(d['Thresholds'])})"] = formatter(d["Thresholds"])
        return new_d

    @staticmethod
    def printable_dict_to_str(d) -> str:
        """
        Convert a printable dictionary (as made with `.make_dict_printable()`)
        to a string.

        Parameters
        ----------
        d : dict
            Printable dictionary as made with `.make_dict_printable()`.

        Returns
        -------
        str
        """
        strings = ["ROC Curve:"]
        strings += [f"{key}: {val}" for key, val in d.items() if val is not None]
        return "\n  ".join(strings)

    def __str__(self):
        d = self.to_printable_dict(max_array_elems=None, decimals=4)
        return ROCCurve.printable_dict_to_str(d)


class ROCCurves:
    def __init__(
        self,
    ) -> None:
        """
        Container for `ROCCurve` objects.
        """
        self.paths = []
        self._curves = {}

    def add(self, path: str, roc_curve: ROCCurve, replace: bool = False):
        """
        Add ROCCurve object to the container.

        Parameters
        ----------
        path : str
            Path of dot-separated dict keys to assign `roc_curve` at.
            This allows the `ROCCurves` container to have a nested structure.
            For non-nested structures, this can be thought of simply as a dict key.
            See the `attr` argument and examples in `nattrs.nested_setattr()`.
        roc_curve : ROCCurve
            ROC curve object.
        replace : bool
            Whether to replace existing `ROCCurve` objects or
            raise a `ValueError` when the path already exists.

        Returns
        -------
        ROCCurves
            Self, allowing chaining of methods.

        Examples
        --------

        Example of nested structure. Note: Fill in `ROCCurve()` arguments before running it.
        >>> container = ROCCurves()
        >>> container.add("a.b.c", ROCCurve(...))
        >>> container.add("a.b.d", ROCCurve(...))
        >>> print(container)
        ROC Curve Collection:
            ROC Curves:
                {
                "a": {
                    "b": {
                        "c": {
                            "AUC": ..., "FPR (x)": [...], "TPR (x)": [...], "Thresholds (x)": [...]
                        },
                        "d": {
                            "AUC": ..., "FPR (x)": [...], "TPR (x)": [...], "Thresholds (x)": [...]
                    }
                    }
                }
            }
        """
        if not replace and nested_hasattr(self._curves, path):
            raise ValueError(f"A `ROCCurve` already exists at path: {path}.")
        nested_setattr(obj=self._curves, attr=path, value=roc_curve, make_missing=True)
        if path not in self.paths:
            self.paths.append(path)

        return self

    def get(self, path: str) -> ROCCurve:
        """
        Get a `ROCCurve` object from a given path.

        Returns
        -------
        ROCCurve
            The `ROCCurve` object asked for.
        """
        return nested_getattr(obj=self._curves, attr=path)

    def __copy__(self):
        out = ROCCurves()
        for path in self.paths:
            out.add(path=path, roc_curve=self.get(path=path).__copy__())
        return out

    def save(self, file_path: Union[str, pathlib.PurePath]):
        """
        Save collection to disk.

        NOTE: The `numpy.ndarray`s are converted to lists when
        saved via json, meaning that the types are not saved.
        Upon loading, the types are assumed to be `numpy.float32`,
        which could lead to different dtypes than the original
        arrays.

        Parameters
        ----------
        file_path : str or `pathlib.Path`
            Path to save collection at. Should end with `.json`.

        Returns
        -------
        ROCCurves
            Self, allowing chaining of methods.
        """
        if str(file_path)[-5:] != ".json":
            raise ValueError("`file_path` must have the extension `.json`.")

        # TODO Fix saving+restoring of dtypes for arrays (Note: Aren't they always floats?)
        out = self.to_lists(copy=True).to_dicts(copy=False).to_dict()

        # Write to desk
        with open(str(file_path), "w") as outfile:
            json.dump(out, outfile)

        return self

    @staticmethod
    def load(file_path: Union[str, pathlib.PurePath]):
        """
        Load a ROC curve collection from disk and convert
        to a `ROCCurves` instance.

        Assumes values (`fpr`, `tpr`, `thresholds`) are `numpy.float32`.

        Returns
        -------
        ROCCurves
            The loaded collection.
            Allows chaining of methods.
        """
        with open(str(file_path), "r") as infile:
            d = json.load(infile)
        if sorted(d.keys()) != ["Curves", "Paths"]:
            raise RuntimeError(
                f"Json file did not have the required top-level keys: {file_path}"
            )
        new_collection = ROCCurves()
        new_collection.paths = d["Paths"]
        new_collection._curves = d["Curves"]
        new_collection.to_roc_curves(dtype=np.float32, copy=False)
        return new_collection

    @staticmethod
    def merge(collections: dict, path_prefix: str = ""):
        """
        Merge multiple `ROCCurves` collections.
        The paths in each collection are prefixed by `<path_prefix>.<collection name>.`
        why all paths are unique during merging and no overwriting will happen.

        Parameters
        ----------
        collections : dict of `ROCCurves` collections
            Mapping of collection names to `ROCCurves`.
        path_prefix : str
            Prefix to add for all paths in the new collection.

        Returns
        -------
        ROCCurves
            A new collection containing all the collections.
            Allows chaining of methods.
        """
        if len(path_prefix) > 0 and path_prefix[-1] != ".":
            # Add final dot
            path_prefix = path_prefix + "."

        # Create new collection
        new_collection = ROCCurves()

        for coll_name, coll in collections.items():
            # Ensure column name is a string (could be integer)
            coll_name = str(coll_name)
            coll_prefix = f"{path_prefix}{coll_name}"
            if "." in coll_name:
                raise ValueError(
                    "A key in `collections` contained a dot ('.'). "
                    f"This is not allowed. The key was: `{coll_name}`."
                )
            nested_setattr(
                new_collection._curves,
                attr=coll_prefix,
                value=coll._curves,
                make_missing=True,
            )
            new_collection.paths += [coll_prefix + "." + path for path in coll.paths]

        return new_collection

    def get_average_roc_curve(
        self,
        paths: List[str],
        num_points: int = 1001,
        weights: Optional[List[float]] = None,
    ) -> ROCCurve:
        """
        Get the average ROC curves for a given set of paths.

        Uses linear interpolation and vertical (weighted) averaging.

        Parameters
        ----------
        num_points
            Number of points to use during interpolation.
        """
        if weights is not None and len(weights) != len(paths):
            raise ValueError(
                "When supplying `weights`, there must be exactly one weight per path."
            )

        # Create ROC dict
        rocs = {path: self.get(path) for path in paths}

        # Create weights dict
        weights = (
            None
            if weights is None
            else {path: weight for path, weight in zip(paths, weights)}
        )

        # Calculate average ROC curves
        return ROCCurves.average_roc_curves(
            rocs,
            num_points=num_points,
            weights=weights,
        )

    @staticmethod
    def average_roc_curves(
        roc_dict: Dict[str, ROCCurve],
        num_points: int = 1001,
        weights: Optional[Dict[str, float]] = None,
    ) -> ROCCurve:
        """
        Average a set of ROC curves using linear interpolation and
        vertical (weighted) averaging.

        Interpolation is performed relative to the
        False Positive Rates.
        """
        # Initialize lists to hold interpolated TPR and threshold values
        thresholds_interp_list = []
        tpr_interp_list = []

        # We use fpr as reference during interpolation
        common_fpr = None

        # Extract weights or use uniform weights if not provided
        if weights is None:
            weights = np.ones(len(roc_dict))
        else:
            # Ensure order and convert to array
            weights = np.array([weights[curve_name] for curve_name in roc_dict.keys()])
            # Normalize weights to mean=1
            weights /= np.mean(weights)

        for roc_, weight in zip(roc_dict.values(), weights):
            interpolated_roc = roc_.interpolate(to=num_points, reference="fpr")

            tpr_interp_list.append(interpolated_roc.tpr * weight)
            thresholds_interp_list.append(interpolated_roc.thresholds * weight)

            if common_fpr is None:
                common_fpr = interpolated_roc.fpr

        # Calculate the weighted average TPR and FPR
        mean_tpr = np.sum(tpr_interp_list, axis=0) / np.sum(weights)
        mean_thresholds = np.sum(thresholds_interp_list, axis=0) / np.sum(weights)

        # Reverse arrays to be consistent with rest of ROC Curves
        return ROCCurve(
            fpr=common_fpr,
            tpr=mean_tpr,
            thresholds=mean_thresholds,
            auc=None,
        ).recalculate_auc()

    def to_dict(self, copy: bool = True) -> dict:
        """
        Convert object to dictionary with curves and paths.

        Parameters
        ----------
        copy : bool
            Whether to return a copy of the curves and paths.

        Returns
        -------
        dict
            Dictionary with curves and paths.
        """
        return {
            "Curves": self._curves.copy() if copy else self._curves,
            "Paths": self.paths.copy() if copy else self.paths,
        }

    def to_lists(self, copy: bool = True):
        """
        Convert all `numpy.ndarray`s to lists.

        Always returns a `ROCCurves` instance,
        so methods can be chained.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with converted curves
            or to convert the curves of the existing object.

        Returns
        -------
        ROCCurves
            A collection of `ROCCurve` objects where values are stored in lists.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ROCCurves._to_lists(out._curves, paths=self.paths)
        return out

    def to_dicts(self, copy: bool = True):
        """
        Convert all `ROCCurve` objects to dictionaries.

        Always returns a `ROCCurves` instance,
        so methods can be chained.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with converted curves
            or to convert the curves of the existing object.

        Returns
        -------
        ROCCurves
            A collection of dicts with the values from the `ROCCurve` objects.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ROCCurves._to_dicts(out._curves, paths=self.paths)
        return out

    def to_roc_curves(self, dtype: Optional[npt.DTypeLike] = None, copy: bool = True):
        """
        Convert all dicts to `ROCCurve`s.

        **When to use**:
        Should only be used after a conversion to dicts with `.to_dicts()`.

        Always returns a `ROCCurves` instance,
        so methods can be chained.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type to enforce on the list/array values.
            Passed to `numpy.asarray(..., dtype=dtype)`.
        copy : bool
            Whether to return a copy with converted curves
            or to convert the curves of the existing object.

        Returns
        -------
        ROCCurves
            A collection of `ROCCurve` objects.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ROCCurves._to_roc_curves(out._curves, paths=self.paths, dtype=dtype)
        return out

    def to_ndarrays(self, dtype: Optional[npt.DTypeLike] = None, copy: bool = True):
        """
        Convert all lists to `numpy.ndarray`s.

        **When to use**:
        Should only be used after a conversion to lists, as they
        are `numpy.ndarray`s by default.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type to enforce on the list/array values.
            Passed to `numpy.asarray(..., dtype=dtype)`.
        copy : bool
            Whether to return a copy with converted curves
            or to convert the curves of the existing object.

        Returns
        -------
        ROCCurves
            A collection of `ROCCurve` objects where values are stored in `numpy.ndarray`s.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ROCCurves._to_ndarrays(out._curves, paths=self.paths, dtype=dtype)
        return out

    @staticmethod
    def _to_lists(curves: dict, paths: List[str]) -> None:
        # TODO: This does not work after we changed to copy arg
        ROCCurves._mutate_curves(
            curves=curves,
            paths=paths,
            fn=lambda x: x.to_lists(copy=False),
            is_inplace_fn=True,
        )

    @staticmethod
    def _to_dicts(curves: dict, paths: List[str]) -> None:
        ROCCurves._mutate_curves(curves=curves, paths=paths, fn=lambda x: x.to_dict())

    @staticmethod
    def _to_ndarrays(curves: dict, paths: List[str], dtype=None) -> None:
        ROCCurves._mutate_curves(
            curves=curves,
            paths=paths,
            fn=lambda x: x.to_ndarrays(copy=False, dtype=dtype),
            is_inplace_fn=True,
        )

    @staticmethod
    def _to_roc_curves(
        curves: dict, paths: List[str], dtype: Optional[npt.DTypeLike] = None
    ) -> None:
        ROCCurves._mutate_curves(
            curves=curves, paths=paths, fn=lambda x: ROCCurve.from_dict(x, dtype=dtype)
        )

    @staticmethod
    def _to_strings(curves: dict, paths: List[str]) -> None:
        ROCCurves._mutate_curves(curves=curves, paths=paths, fn=lambda x: str(x))

    @staticmethod
    def _to_printable_dicts(
        curves: dict,
        paths: List[str],
        max_array_elems: Optional[int] = 3,
        decimals: Optional[int] = 4,
    ) -> None:
        def convert_to_printable_dict(x):
            if isinstance(x, dict):
                return ROCCurve.make_dict_printable(
                    d=x, max_array_elems=max_array_elems, decimals=decimals
                )
            else:
                return x.to_printable_dict(
                    max_array_elems=max_array_elems, decimals=decimals
                )

        ROCCurves._mutate_curves(
            curves=curves, paths=paths, fn=lambda x: convert_to_printable_dict(x)
        )

    @staticmethod
    def _mutate_curves(
        curves: dict, paths: List[str], fn: Callable, is_inplace_fn: bool = False
    ) -> None:
        for path in paths:
            nested_mutattr(obj=curves, attr=path, fn=fn, is_inplace_fn=is_inplace_fn)

    def __str__(self) -> str:
        strings = ["ROC Curve Collection:"]
        curves = copy.deepcopy(self._curves)
        ROCCurves._to_printable_dicts(curves, paths=self.paths)
        curves_string = _dict_to_str(curves)
        strings += [f"ROC Curves:\n    {curves_string}"]
        return "\n  ".join(strings)
