import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from generalize.utils.normalization import (
    standardize_features_3D,
    standardize_features_2D,
)


def calculate_tree_feature_importance(
    x: np.ndarray,
    y: np.ndarray,
    task: str = "regression",
    standardize_cols: bool = True,
    seed: int = 1,
) -> np.ndarray:
    """
    Calculate random forest feature importances.
    A basic random forest is fitted to the data, using all features at once.
    The tree object contains a feature importance score, which we return.

    Parameters
    ----------
    y: 1D `numpy.ndarray`
        Target values.
    x: 3D `numpy.ndarray`
        Data to calculate feature importances for.
        Expected shape: (num_samples, features).
    standardize_cols
        Whether to standardize the feature before fitting its model.
    seed
        Random seed for the random forest to ensure reproducibility.

    Returns
    -------
    `numpy.ndarray`
        Feature importances.
    """
    if standardize_cols:
        standardize_fn = (
            standardize_features_2D if x.ndim == 2 else standardize_features_3D
        )
        x = standardize_fn(x.copy())

    # Assign model function
    if task == "regression":
        model = RandomForestRegressor(random_state=seed)
    elif task == "classification":
        model = RandomForestClassifier(random_state=seed)

    # Fit model
    model.fit(x[:, :], y)

    return model.feature_importances_
