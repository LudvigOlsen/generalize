import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from generalize.model.transformers.dim_wrapper import DimTransformerWrapper
from generalize.model.transformers.mean_during_test import MeanDuringTest


def test_mean_during_test_transformer_simple():
    # Data
    np.random.seed(1)
    x = np.random.normal(size=(35, 15))
    y = np.random.choice([0, 1], size=35)

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "mean_during_test",
            DimTransformerWrapper(
                MeanDuringTest,
                kwargs={"feature_indices": [0, 1, 2], "training_mode": True},
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    # No change during training mode
    x_tr = pipe.transform(x)
    assert_array_almost_equal(x, x_tr)

    # Set to test mode
    pipe.named_steps["mean_during_test"].set_params(training_mode=False)

    # No change during training mode
    x_tr_2 = pipe.transform(x)
    x_exp = x.copy()
    x_exp[:, [0, 1, 2]] = np.expand_dims(x_exp[:, [0, 1, 2]].mean(axis=0), axis=0)
    assert_array_almost_equal(x_tr_2, x_exp)


def test_mean_during_test_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(MeanDuringTest(feature_indices=[0], training_mode=False))
    check_estimator(MeanDuringTest(feature_indices=[0], training_mode=True))
