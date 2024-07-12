import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from generalize.model.transformers.dim_wrapper import DimTransformerWrapper
from generalize.model.transformers.correlated_feature_remover import (
    CorrelatedFeatureRemover,
)


def test_correlated_feature_remover_transformer_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(CorrelatedFeatureRemover())


def test_correlated_feature_remover_transformer_simple():

    # Data
    np.random.seed(1)
    x = np.random.normal(size=(15, 5))
    y = np.random.choice([0, 1], size=15)

    # Check correlations between features
    print("Pre-selection correlations: \n", np.corrcoef(x, rowvar=False))

    # Pipeline steps
    # Simple version first
    steps = [
        (
            "correlated_feature_remover",
            DimTransformerWrapper(
                CorrelatedFeatureRemover,
                kwargs={"threshold": 0.3, "select_by": "avg_correlation_fast"},
            ),
        )
    ]

    pipe = Pipeline(steps=steps)
    pipe.fit(x, y)

    assert pipe.steps[0][1].estimator_.remove_indices_ == [1, 3]

    x_tr = pipe.transform(x)

    # Check correlations between features
    post_corrs = np.corrcoef(x_tr, rowvar=False)
    print("Post-selection correlations: \n", post_corrs)

    # Test that all non-diagonal correlations are below the threshold
    assert np.max(np.abs(post_corrs)[~np.eye(post_corrs.shape[0], dtype=bool)]) < 0.3

    # We removed 2 features
    assert x_tr.shape == (15, 3)
