import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from sklearn.linear_model import LogisticRegression

from generalize.model.pipeline.pipelines import create_pipeline
from generalize.model.transformers.dim_wrapper import DimTransformerWrapper
from generalize.model.transformers.mean_during_test import MeanDuringTest
from generalize.model.transformers.pca_by_explained_variance import (
    PCAByExplainedVariance,
)


def test_create_pipeline_3d():
    np.random.seed(1)
    x3d = np.random.normal(size=(5, 3, 10))
    y = np.random.choice((0, 1), size=(5))

    pipe = create_pipeline(
        model=None,
        flatten_feature_sets=True,
        add_channel_dim=True,
        transformers=[
            (
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            )
        ],
    )

    print(pipe)

    x3d_tr = pipe.fit_transform(X=x3d, y=y)

    print(x3d_tr.shape)

    # Max n_samples components during PCA so 3x5
    assert x3d_tr.shape == (5, 1, 3*5)

    pipe = create_pipeline(
        model=LogisticRegression(),
        flatten_feature_sets=True,
        add_channel_dim=False,
        transformers=[
            (
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            )
        ],
    )

    print(pipe)

    preds = pipe.fit(X=x3d, y=y).predict(X=x3d)

    print(preds.shape)

    assert preds.shape == (5,)


def test_create_pipeline_2d():
    np.random.seed(1)
    x2d = np.random.normal(size=(5, 10))
    y = np.random.choice((0, 1), size=(5))

    pipe = create_pipeline(
        model=None,
        flatten_feature_sets=False,
        add_channel_dim=True,
        transformers=[
            (
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            )
        ],
    )

    print(pipe)

    x2d_tr = pipe.fit_transform(X=x2d, y=y)

    print(x2d_tr.shape)

    # max(num samples, num features)
    assert x2d_tr.shape == (5, 1, 5)

    pipe = create_pipeline(
        model=LogisticRegression(),
        flatten_feature_sets=False,
        add_channel_dim=False,
        transformers=[
            (
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            )
        ],
    )

    print(pipe)

    preds = pipe.fit(X=x2d, y=y).predict(X=x2d)

    print(preds.shape)

    assert preds.shape == (5,)


def test_create_pipeline_weighting_by_groups():
    np.random.seed(1)
    x2d = np.random.normal(size=(25, 10))
    y = np.random.choice((0, 1), size=(25))
    groups = np.random.choice((0, 1, 2, 3), size=(25))

    pipe = create_pipeline(
        model=LogisticRegression(),
        flatten_feature_sets=False,
        add_channel_dim=False,
        weight_loss_by_groups=True,
        transformers=[
            (
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            )
        ],
    )

    print(pipe)

    preds = pipe.fit(X=x2d, y=y, groups=groups).predict(X=x2d)

    print(preds.shape)

    assert preds.shape == (25,)

    with pytest.raises(ValueError):
        # Cannot add sample weight by groups
        # when no model is specified
        pipe = create_pipeline(
            model=None,
            flatten_feature_sets=False,
            add_channel_dim=True,
            weight_loss_by_groups=True,
            transformers=[
                (
                    "pca_by_variance",
                    DimTransformerWrapper(
                        PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                    ),
                )
            ],
        )


def test_create_pipeline_train_test_transformers():
    np.random.seed(1)
    x2d = np.random.normal(size=(5, 5))
    y = np.random.choice((0, 1), size=(5))

    pipe = create_pipeline(
        model=None,
        flatten_feature_sets=False,
        add_channel_dim=False,
        transformers=[
            (
                # Test non-train-test transformers are not affected
                "pca_by_variance",
                DimTransformerWrapper(
                    PCAByExplainedVariance, kwargs={"target_variance": 1.0}
                ),
            ),
            (
                "mean_during_test",
                DimTransformerWrapper(
                    MeanDuringTest,
                    kwargs={"feature_indices": [0, 1], "training_mode": True},
                ),
            ),
        ],
        train_test_transformers=["mean_during_test"],
    )

    print(pipe)

    x2d_tr = pipe.fit_transform(X=x2d, y=y)
    assert_array_almost_equal(x2d_tr.shape, x2d.shape)

    print(pipe.get_params())

    assert not pipe.named_steps["mean_during_test"].get_estimator_params()[
        "estimator_0"
    ]["training_mode"]
    print(x2d_tr)

    assert_array_almost_equal(
        x2d_tr[:, [0, 1]], x2d[:, [0, 1]] * 0
    )  # It's zero due to the standardization


# TODO: Confounders are now handled in `transformers`. Worth testing though!
# def test_create_pipeline_with_confounders():
#     np.random.seed(1)
#     x2d = np.random.normal(size=(5, 5))
#     confounders = np.random.normal(size=(5, 2))
#     y = np.random.choice((0, 1), size=(5))

#     pipe = create_pipeline(
#         model=None,
#         # num_confounders=confounders.shape[-1],
#         flatten_feature_sets=False,
#         add_channel_dim=True,
#         transformers=[
#             (
#                 # Test non-train-test transformers are not affected
#                 "distance_to_mean",
#                 DimTransformerWrapper(
#                     DistanceToMeanTransformer, kwargs={"replace": True}
#                 ),
#             ),
#         ],
#     )

#     print(pipe)

#     x_and_confounders = np.concatenate([confounders, x2d], axis=-1)
#     x_and_confounders_tr = pipe.fit_transform(X=x_and_confounders, y=y)
#     assert_array_almost_equal(
#         # `add_channel_dim` should have added a singleton dimension
#         x_and_confounders_tr.shape,
#         np.expand_dims(x_and_confounders, axis=1).shape,
#     )

#     print(pipe.get_params())

#     assert (
#         not dict(pipe.named_steps["features"].transformer_list)["confounders"]
#         .named_steps["mean_during_test"]
#         .get_estimator_params()["estimator_0"]["training_mode"]
#     )
#     print(x_and_confounders_tr)

#     assert_array_almost_equal(
#         x_and_confounders_tr[:, 0, [0, 1]], x_and_confounders[:, [0, 1]] * 0
#     )  # It's zero due to the standardization

#     print("\nWITH MODEL\n")

#     # Ensure the modeling part works after the feature split/concat
#     pipe = create_pipeline(
#         model=LogisticRegression(),
#         num_confounders=confounders.shape[-1],
#         flatten_feature_sets=False,
#         add_channel_dim=False,
#         transformers=[
#             (
#                 "distance_to_mean",
#                 DimTransformerWrapper(
#                     DistanceToMeanTransformer, kwargs={"replace": True}
#                 ),
#             )
#         ],
#     )

#     print()
#     print(pipe)

#     preds = pipe.fit(X=x_and_confounders, y=y).predict(X=x_and_confounders)

#     print()
#     print(pipe.get_params())

#     assert (
#         len(pipe.named_steps["model"].coef_.flatten()) == x_and_confounders.shape[-1]
#     ), "confounders did not get model coefficients"

#     print(preds.shape)

#     assert preds.shape == (5,)

#     assert (
#         not dict(pipe.named_steps["features"].transformer_list)["confounders"]
#         .named_steps["mean_during_test"]
#         .get_estimator_params()["estimator_0"]["training_mode"]
#     )
