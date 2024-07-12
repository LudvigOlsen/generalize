import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from generalize.model.transformers.pca_by_explained_variance import (
    PCAByExplainedVariance,
)


def test_pca_by_variance_initialization():
    """Test the initialization and attributes of the PrepareMegabins class."""
    estimator = PCAByExplainedVariance(target_variance=0.8)
    assert estimator.target_variance == 0.8


def test_pca_by_variance_fit_dimensions():
    """Test fitting to ensure dimensions and learned attributes are set correctly."""
    np.random.seed(1)
    X = np.random.rand(10, 100)  # 10 samples, 100 features
    y = np.random.choice([0, 1], size=10)
    estimator = PCAByExplainedVariance(target_variance=0.8)
    estimator.fit(X, y)

    # Check if PCA components are fitted
    print(estimator.n_components_)
    print(
        estimator.explained_variance_ratio_, estimator.explained_variance_ratio_.sum()
    )
    assert estimator.n_components_ == 7
    assert hasattr(estimator, "num_input_features_")


def test_pca_by_variance_transform_not_fitted():
    """Ensure that transform raises an error if called before fit."""
    np.random.seed(1)
    X = np.random.rand(10, 100)
    estimator = PCAByExplainedVariance(target_variance=0.8)
    with pytest.raises(NotFittedError):
        estimator.transform(X)


def test_pca_by_variance_passes_check_estimator():
    # Check that the transformer passes standard estimator checks
    check_estimator(PCAByExplainedVariance(target_variance=0.8))
