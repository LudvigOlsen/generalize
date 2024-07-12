import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator


from generalize.model.transformers.dim_wrapper import DimTransformerWrapper
from generalize.model.transformers.truncate_extremes import (
    TruncateExtremesByPercentiles,
    TruncateExtremesByIQR,
    TruncateExtremesBySTD,
)


def test_truncate_extremes_by_percentiles_transformer_simple():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "truncate_extremes",
            DimTransformerWrapper(
                TruncateExtremesByPercentiles,
                kwargs={
                    "lower_percentile": 20.0,
                    "upper_percentile": 80.0,
                    "by": "all",
                },
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    x_tr = pipe.transform(x)

    # We should have same output shape
    assert x_tr.shape == x.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    assert_raises(AssertionError, assert_array_almost_equal, x_tr, x)

    assert ((x_tr.min(), x_tr.max()) == np.percentile(x, [20.0, 80.0])).all()


def test_truncate_extremes_by_IQR_transformer_simple():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "truncate_extremes",
            DimTransformerWrapper(
                TruncateExtremesByIQR,
                kwargs={
                    "lower_num_iqr": 0.5,  # Ensure low enough to affect data
                    "upper_num_iqr": 0.5,
                    "by": "all",
                },
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    x_tr = pipe.transform(x)

    # We should have same output shape
    assert x_tr.shape == x.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    assert_raises(AssertionError, assert_array_almost_equal, x_tr, x)

    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    assert [x_tr.min(), x_tr.max()] == [q1 - 0.5 * iqr, q3 + 0.5 * iqr]


def test_truncate_extremes_by_std_transformer_simple():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "truncate_extremes",
            DimTransformerWrapper(
                TruncateExtremesBySTD,
                kwargs={
                    "lower_num_std": 0.5,  # Ensure low enough to affect data
                    "upper_num_std": 0.5,
                    "by": "all",
                },
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    x_tr = pipe.transform(x)

    # We should have same output shape
    assert x_tr.shape == x.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    assert_raises(AssertionError, assert_array_almost_equal, x_tr, x)

    mean = np.mean(x)
    std = np.std(x)
    assert [x_tr.min(), x_tr.max()] == [mean - 0.5 * std, mean + 0.5 * std]


def test_truncate_extremes_by_percentile_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(
        TruncateExtremesByPercentiles(lower_percentile=1.0, upper_percentile=99.0)
    )


def test_truncate_extremes_by_IQR_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(TruncateExtremesByIQR(lower_num_iqr=1.5, upper_num_iqr=1.5))


def test_truncate_extremes_by_std_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(TruncateExtremesBySTD(lower_num_std=2.5, upper_num_std=2.5))
