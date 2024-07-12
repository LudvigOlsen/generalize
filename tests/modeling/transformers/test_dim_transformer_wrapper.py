import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from generalize.model.transformers.dim_wrapper import DimTransformerWrapper


def test_dim_transformer_wrapper():

    # Data
    np.random.seed(1)
    x_2d = np.random.normal(size=(10, 5))
    x_3d = np.random.normal(size=(10, 3, 5))
    x_4d = np.random.normal(size=(10, 3, 2, 5))
    y = np.random.choice([0, 1], size=10)

    # Pipeline steps
    steps = [
        ("scaler_1", DimTransformerWrapper(StandardScaler)),
        ("scaler_2", DimTransformerWrapper(StandardScaler)),
    ]

    #### Three-dimensional data ####

    pipe = Pipeline(steps=steps)
    pipe.fit(x_3d, y)

    x_3d_tr = pipe.transform(x_3d)

    assert x_3d_tr.shape == x_3d.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    assert_raises(
        AssertionError,
        assert_array_almost_equal,
        x_3d.mean(axis=0),
        x_3d_tr.mean(axis=0),
    )

    # Test that means are 0 for each feature for each feature set
    assert_array_almost_equal(
        x_3d_tr.mean(axis=0), np.zeros(shape=(3, 5), dtype=np.float64), decimal=14
    )

    # Test that standard deviations are 1 for each feature for each feature set
    assert_array_almost_equal(
        x_3d_tr.std(axis=0), np.ones(shape=(3, 5), dtype=np.float64), decimal=14
    )

    #### Two-dimensional data ####

    pipe = Pipeline(steps=steps)
    pipe.fit(x_2d, y)

    x_2d_tr = pipe.transform(x_2d)

    assert x_2d_tr.shape == x_2d.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    assert_raises(
        AssertionError,
        assert_array_almost_equal,
        x_2d.mean(axis=0),
        x_2d_tr.mean(axis=0),
    )

    # Test that means are 0 for each feature
    assert_array_almost_equal(
        x_2d_tr.mean(axis=0), np.zeros(shape=(5), dtype=np.float64), decimal=14
    )

    # Test that standard deviations are 1 for each feature
    assert_array_almost_equal(
        x_2d_tr.std(axis=0), np.ones(shape=(5), dtype=np.float64), decimal=14
    )

    #### Four-dimensional data ####

    pipe = Pipeline(steps=steps)
    pipe.fit(x_4d, y)

    x_4d_tr = pipe.transform(x_4d)

    assert x_4d_tr.shape == x_4d.shape

    # Check that something happened
    # (That the two arrays are not almost equal!)
    for d2 in range(x_4d.shape[2]):
        assert_raises(
            AssertionError,
            assert_array_almost_equal,
            x_4d[:, :, d2, :].mean(axis=0),
            x_4d_tr[:, :, d2, :].mean(axis=0),
        )

    # Test that means are 0 for each feature for each
    # feature set (combinations of dim 1 and 2)

    for idx in [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]:
        assert_array_almost_equal(
            x_4d_tr[:, idx[0], idx[1], :].mean(axis=0),
            np.zeros(shape=(5,), dtype=np.float64),
            decimal=14,
        )

        # Test that standard deviations are 1 for each feature for each feature set
        assert_array_almost_equal(
            x_4d_tr[:, idx[0], idx[1], :].std(axis=0),
            np.ones(shape=(5,), dtype=np.float64),
            decimal=14,
        )


# TODO Test the kwargs and ArgValueList stuff


def test_dim_transformer_wrapper_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(DimTransformerWrapper(StandardScaler))
