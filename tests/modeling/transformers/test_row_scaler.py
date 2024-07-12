import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LogisticRegression

from generalize.model.transformers.dim_wrapper import DimTransformerWrapper
from generalize.model.transformers.row_scaler import RowScaler
from generalize.model.cross_validate.grid_search import SeededGridSearchCV


def test_row_scaler_transformer_simple():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(RowScaler, kwargs={"center": "mean", "scale": "std"}),
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

    assert np.round(np.mean(np.std(x_tr, axis=-1)), decimals=10) == 1.0


def test_row_scaler_transformer_feature_groups():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "center": ["mean", "mean"],
                    "scale": ["std", "std"],
                    "feature_groups": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
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

    assert np.round(np.mean(np.mean(x_tr[:, :5], axis=-1)), decimals=10) == 0.0
    assert np.round(np.mean(np.mean(x_tr[:, 5:10], axis=-1)), decimals=10) == 0.0
    assert (
        np.round(np.mean(np.mean(x_tr[:, 10:15], axis=-1)), decimals=10) == 0.0639923315
    )  # Not centered!

    assert np.round(np.mean(np.std(x_tr[:, :5], axis=-1)), decimals=10) == 1.0
    assert np.round(np.mean(np.std(x_tr[:, 5:10], axis=-1)), decimals=10) == 1.0
    assert (
        np.round(np.mean(np.std(x_tr[:, 10:15], axis=-1)), decimals=10) == 0.7613152488
    )  # Not scaled!

    # SAME - but with single string arguments for center and scale

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "center": "mean",
                    "scale": "std",
                    "feature_groups": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
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

    assert np.round(np.mean(np.mean(x_tr[:, :5], axis=-1)), decimals=10) == 0.0
    assert np.round(np.mean(np.mean(x_tr[:, 5:10], axis=-1)), decimals=10) == 0.0
    assert (
        np.round(np.mean(np.mean(x_tr[:, 10:15], axis=-1)), decimals=10) == 0.0639923315
    )  # Not centered!

    assert np.round(np.mean(np.std(x_tr[:, :5], axis=-1)), decimals=10) == 1.0
    assert np.round(np.mean(np.std(x_tr[:, 5:10], axis=-1)), decimals=10) == 1.0
    assert (
        np.round(np.mean(np.std(x_tr[:, 10:15], axis=-1)), decimals=10) == 0.7613152488
    )  # Not scaled!


def test_row_scaler_transformer_center_features():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # "Centering" only
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "center": "mean",
                    "scale": None,
                    "feature_groups": [
                        [2, 3, 4, 5],
                        [6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                    ],
                    "center_by_features": [0, 1, -1],
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

    assert_array_almost_equal(x_tr[:, :2], x[:, :2])  # No change
    assert_array_almost_equal(x_tr[:, 2:6], x[:, 2:6] - x[:, [0]])  # Centered by f0
    assert_array_almost_equal(x_tr[:, 6:10], x[:, 6:10] - x[:, [1]])  # Centered by f1
    assert (
        np.round(np.mean(np.mean(x_tr[:, 10:15], axis=-1)), decimals=10) == 0.0
    )  # Centered by mean

    # Add scaling
    # Pipeline steps
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "center": "mean",
                    "scale": "std",
                    "feature_groups": [
                        [2, 3, 4, 5],
                        [6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                    ],
                    "center_by_features": [0, 1, -1],
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

    assert_array_almost_equal(x_tr[:, :2], x[:, :2])  # No change
    assert np.round(np.mean(np.std(x_tr[:, 2:6], axis=-1)), decimals=10) == 1.0
    assert np.round(np.mean(np.std(x_tr[:, 6:10], axis=-1)), decimals=10) == 1.0
    assert np.round(np.mean(np.std(x_tr[:, 10:15], axis=-1)), decimals=10) == 1.0

    # SAME - but with single center by feature

    # Pipeline steps
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "scale": "std",
                    "center_by_features": 0,
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

    assert_array_almost_equal(x_tr[:, :1], x[:, :1])  # No change
    exp_rest = x[:, 1:15] - np.expand_dims(x[:, 0], axis=-1)
    exp_rest /= np.expand_dims(np.std(x[:, 1:15], axis=-1), axis=-1)
    assert_array_almost_equal(x_tr[:, 1:15], exp_rest)  # Centered by f0

    assert np.round(np.mean(np.std(x_tr[:, 1:15], axis=-1)), decimals=10) == 1.0

    # Removing centering features

    # Pipeline steps
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs={
                    "scale": "std",
                    "center_by_features": 0,
                    "rm_center_by_features": True,
                },
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    x_tr = pipe.transform(x)

    # We should have one feature less
    assert x_tr.shape[-1] + 1 == x.shape[-1]


def test_row_scaler_transformer_in_gridsearch():
    # Data
    np.random.seed(4)
    x = np.random.normal(size=(50, 15)) + 0.1
    y = np.random.choice([0, 1], size=50)
    x[y == 1, 0:3] += 0.5

    print(y)

    scaler_kwargs = {"center": None, "scale": "std"}

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "scale_rows",
            DimTransformerWrapper(
                RowScaler,
                kwargs=scaler_kwargs,
            ),
        ),
        (
            "model",
            LogisticRegression(
                penalty="l1",
                solver="saga",
                tol=0.0001,
                max_iter=1000,
            ),
        ),
    ]

    pipe = Pipeline(steps=steps)

    grid = SeededGridSearchCV(pipe, cv=3, param_grid={"model__C": [0.1, 1]}, seed=3)

    grid.fit(X=x, y=y)

    out = grid.predict(x[:5])

    assert_array_almost_equal(out, np.array([1, 1, 0, 1, 1]))


def test_row_scaler_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(RowScaler(center="mean", scale="std"))
