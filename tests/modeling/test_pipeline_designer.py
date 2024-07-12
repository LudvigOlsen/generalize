import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from generalize.model.pipeline.pipeline_designer import PipelineDesigner
from generalize.model.transformers.pca_by_explained_variance import (
    PCAByExplainedVariance,
)


def test_pipeline_designer():
    np.random.seed(1)
    x = np.random.normal(size=(5, 10))
    y = np.random.choice((0, 1), size=(5))

    # Create a pipeline designer
    designer = PipelineDesigner()

    # Add a few different arbitrary steps
    # Including a split of the features
    designer.add_step(
        "standardize_rows",
        "scale_rows",
        add_dim_transformer_wrapper=True,
        kwargs={"center": "median"},
    )
    designer.split(
        "confounders_features",
        split_indices={"confounders": [0, 1, 2, 3], "features": [4, 5, 6, 7, 8, 9]},
    )
    designer.add_step(
        "features__standardize_cols",
        "scale_features",
        add_dim_transformer_wrapper=True,
        kwargs=None,
    )
    designer.add_step(
        "confounders__standardize_cols",
        "scale_features",
        add_dim_transformer_wrapper=True,
        kwargs=None,
    )
    designer.add_step(
        "features__pca_by_variance",
        PCAByExplainedVariance,
        add_dim_transformer_wrapper=True,
        kwargs={"target_variance": 0.9},
    )
    designer.collect()
    designer.add_step(
        "restandardize_rows",
        "scale_rows",
        add_dim_transformer_wrapper=True,
        kwargs={"center": "mean"},
    )

    steps = designer.build()
    pipe = Pipeline(steps)
    print(pipe)

    pipe.fit(x, y)
    x_transformed = pipe.transform(x)

    assert x_transformed.shape == x.shape
    print(set(pipe.named_steps))
    # Test pipeline flow
    step_paths = set(
        [
            "standardize_rows",
            "confounders_features__confounders__select_features",
            "confounders_features__features__select_features",
            "confounders_features__features__standardize_cols",
            "confounders_features__confounders__standardize_cols",
            "confounders_features__features__pca_by_variance",
            "restandardize_rows",
        ]
    )
    print(set(_get_estimator_names(pipe)))
    assert step_paths == set(_get_estimator_names(pipe))


def test_pipeline_designer_with_empty_split():
    np.random.seed(1)
    x = np.random.normal(size=(5, 10))
    y = np.random.choice((0, 1), size=(5))

    # Create a pipeline designer
    designer = PipelineDesigner()

    # Add a few different arbitrary steps
    # Including a split of the features
    designer.add_step(
        "standardize_rows",
        "scale_rows",
        add_dim_transformer_wrapper=True,
        kwargs={"center": "median"},
    )
    designer.split(
        "confounders_features",
        split_indices={"confounders": [], "features": range(10)},
    )
    designer.add_step(
        "features__standardize_cols",
        "scale_features",
        add_dim_transformer_wrapper=True,
        kwargs=None,
    )
    # A step that won't be included due to the lack of confounders
    designer.add_step(
        "confounders__standardize_cols",
        "scale_features",
        add_dim_transformer_wrapper=True,
        kwargs=None,
    )
    designer.add_step(
        "features__pca_by_variance",
        PCAByExplainedVariance,
        add_dim_transformer_wrapper=True,
        kwargs={"target_variance": 0.9},
    )
    designer.collect()
    designer.add_step(
        "restandardize_rows",
        "scale_rows",
        add_dim_transformer_wrapper=True,
        kwargs={"center": "mean"},
    )

    steps = designer.build()
    pipe = Pipeline(steps)
    print(pipe)

    pipe.fit(x, y)
    x_transformed = pipe.transform(x)

    assert x_transformed.shape == x.shape
    print(set(pipe.named_steps))
    # Test pipeline flow
    # No confounder steps
    step_paths = set(
        [
            "standardize_rows",
            "confounders_features__features__select_features",
            "confounders_features__features__standardize_cols",
            "confounders_features__features__pca_by_variance",
            "restandardize_rows",
        ]
    )
    print(set(_get_estimator_names(pipe)))
    assert step_paths == set(_get_estimator_names(pipe))


def _get_estimator_names(estimator, parent_name=""):
    """
    Recursively extract names of all estimators from a scikit-learn pipeline or feature union.

    Parameters
    ----------
    estimator: The pipeline, feature union, or estimator from which to extract names.
    parent_name: Internal use for recursive calls to track the full path of names.

    Returns
    -------
    A list of strings, each representing the full path to an estimator within the structure.
    """
    estimator_names = []

    # Check if the estimator is a Pipeline.
    if isinstance(estimator, Pipeline):
        for step_name, step_estimator in estimator.steps:
            full_name = f"{parent_name}__{step_name}" if parent_name else step_name
            estimator_names.extend(_get_estimator_names(step_estimator, full_name))

    # Check if the estimator is a FeatureUnion.
    elif isinstance(estimator, FeatureUnion):
        for transformer_name, transformer in estimator.transformer_list:
            full_name = (
                f"{parent_name}__{transformer_name}"
                if parent_name
                else transformer_name
            )
            estimator_names.extend(_get_estimator_names(transformer, full_name))

    # Base case: the estimator is neither a Pipeline nor a FeatureUnion.
    else:
        if parent_name:  # Only add the estimator if it's part of a composite structure.
            estimator_names.append(parent_name)

    return estimator_names
