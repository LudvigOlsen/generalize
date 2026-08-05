import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from generalize.model.transformers.index_feature_selector import (
    IndexFeatureSelector,
    IndexFeatureRemover,
)
from generalize.model.transformers.dim_wrapper import DimTransformerWrapper


def test_index_feature_selectors():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # SELECT

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "select_first_10",
            DimTransformerWrapper(
                IndexFeatureSelector,
                kwargs={"feature_indices": range(10)},
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    # No change during training mode
    x_tr = pipe.transform(x)
    assert_array_almost_equal(x[:, :10], x_tr)

    # REMOVE

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "remove_first_10",
            DimTransformerWrapper(
                IndexFeatureRemover,
                kwargs={"feature_indices": range(10)},
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    # No change during training mode
    x_tr = pipe.transform(x)
    assert_array_almost_equal(x[:, 10:], x_tr)


def test_index_feature_selectors_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(IndexFeatureSelector(feature_indices=[0]))
    check_estimator(IndexFeatureRemover(feature_indices=[0]))


def test_index_feature_selectors_allow_nan():
    """NaNs pass through selection for downstream branch-specific handling."""
    x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([0, 1])

    selected = IndexFeatureSelector(feature_indices=[0, 1]).fit_transform(x, y)
    removed = IndexFeatureRemover(feature_indices=[2]).fit_transform(x, y)

    assert_array_almost_equal(selected, x[:, :2])
    assert_array_almost_equal(removed, x[:, :2])
