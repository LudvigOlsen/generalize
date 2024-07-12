import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression

from generalize.model.full_model import train_full_model


# TODO: Doesn't work - regression doesn't seem to work currently
def test_train_full_model_deterministic(tmp_path):
    x_lo = np.zeros(shape=(30, 5), dtype=np.float32)
    x_lo[:, 4] = np.arange(30) / 1000
    x_hi = np.zeros(shape=(30, 5), dtype=np.float32)
    x_hi.fill(0.9)
    x_hi[:, 4] += np.arange(30) / 1000
    x = np.vstack([x_lo, x_hi])

    y = x.sum(axis=-1)

    class SumRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, a=None):
            self.a = a

        def fit(self, X, y, **fit_params):
            self.is_fitted_ = True
            warnings.warn(
                "`SumRegressor` warning catching test warning.", ConvergenceWarning
            )
            # `fit` should always return `self`
            return self

        def predict(self, X=None):
            return X.sum(axis=-1)

    model = SumRegressor()
    grid = {"model__a": [1, 2, 3]}

    # Ignore "divide by zero" warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        model_out = train_full_model(
            x=x,
            y=y,
            model=model,
            grid=grid,
            k=3,
            metric="neg_mean_squared_error",
            task="regression",
            num_jobs=1,
        )

    print(model_out)

    # These should be identical, if the returned
    # predictions have the correct order
    np.testing.assert_array_equal(model_out["Predictions"], y)

    # Since they are identical, we should get MAE and RMSE of 0
    assert all(model_out["Evaluation"]["Scores"]["MAE"] == pd.Series([0.0]))

    assert (
        str(model_out["Warnings"][0].message)
        == "`SumRegressor` warning catching test warning."
    )

    #### Pre-defined folds ####

    print("\n---------------------------------------------\n")

    # We should be able to call the folds what we want
    # Here: train_only(1), 2, 4
    # Where the wrapper `train_only()` means "always in train set"
    folds = np.asarray([["train_only(1)"] * 20, [str(2)] * 20, [str(4)] * 20]).flatten()

    # Ignore "divide by zero" warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        model_out = train_full_model(
            x=x,
            y=y,
            model=model,
            grid=grid,
            split=folds,
            metric="neg_mean_squared_error",
            task="regression",
            num_jobs=1,
        )

    print(model_out)

    # These should be identical, if the returned
    # predictions have the correct order

    np.testing.assert_array_equal(model_out["Predictions"], y[20:])
    np.testing.assert_array_equal(model_out["Indices"], range(20, 60))
    np.testing.assert_array_equal(model_out["Split"], folds[20:])

    #### Pre-defined folds AND groups ####

    print("\n---------------------------------------------\n")

    # We should be able to call the folds what we want
    # Here: train_only(1), 2, 4
    # Where the wrapper `train_only()` means "always in train set"
    folds = np.asarray([["train_only(1)"] * 20, [str(2)] * 20, [str(4)] * 20]).flatten()
    groups = np.sort(np.concatenate([np.arange(len(folds) // 4) for _ in range(4)]))
    groups[np.isin(groups, [8, 9])] *= -1
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # Ignore "divide by zero" warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        model_out = train_full_model(
            x=x,
            y=y,
            model=model,
            groups=groups,
            grid=grid,
            split=folds,
            metric="neg_mean_squared_error",
            task="regression",
            num_jobs=1,
        )

    print(model_out)

    # These should be identical, if the returned
    # predictions have the correct order

    np.testing.assert_array_equal(
        len(model_out["Predictions"]), 40 - 8
    )  # We set 2x4 to train only
    np.testing.assert_array_equal(
        len(model_out["Indices"]), 40 - 8
    )  # We set 2x4 to train only
    np.testing.assert_array_equal(
        len(model_out["Split"]), 40 - 8
    )  # We set 2x4 to train only


@ignore_warnings(category=ConvergenceWarning)
def test_train_full_model_multiclass_classification(
    xy_mc_classification, LogisticRegressionClassifierPartial, tmp_path
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_mc_classification["x"]
    y = xy_mc_classification["y"]
    num_samples = xy_mc_classification["num_samples"]

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_mc"]

    model_out = train_full_model(
        x=x,
        y=y,
        model=model,
        grid=grid,
        k=3,
        metric="balanced_accuracy",
        task="multiclass_classification",
        num_jobs=1,
        seed=seed,
    )

    print("Finished model training")
    print(model_out)

    raise

    # # Testing small subset of results
    # # as evaluate() is (TODO) tested elsewhere

    # # Testing summarized results
    # print(cv_out["Evaluation"]["Summary"])
    # assert (
    #     cv_out["Evaluation"]["Summary"]["What"]
    #     == "Summary of 2 evaluations from multiclass classification."
    # )
    # assert (
    #     np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5)
    #     == np.round(pd.Series([0.96, 0.0000, 0.96, 0.96, 0.000000]), decimals=5)
    # ).all()
    # assert (
    #     np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Macro AUC"], decimals=5)
    #     == np.round(
    #         pd.Series([0.994196, 0.002994, 0.992079, 0.996313, 0.000000]), decimals=5
    #     )
    # ).all()

    # # Total (summed) confusion matrix
    # # Sums to 2 x num_samples == 100

    # # print(cv_out["Evaluation"]["Summary"]["Confusion Matrix"])
    # np.testing.assert_equal(
    #     cv_out["Evaluation"]["Summary"]["Confusion Matrix"].get_counts(),
    #     np.array([[34, 2, 0], [1, 20, 1], [0, 2, 40]]),
    # )
    # assert (
    #     np.sum(cv_out["Evaluation"]["Summary"]["Confusion Matrix"].get_counts())
    #     == 2 * num_samples
    # )

    # # Testing repetition evaluations
    # print(cv_out["Evaluation"]["Evaluations"])
    # assert cv_out["Evaluation"]["Evaluations"]["What"] == "Combined evaluations."
    # assert (
    #     len(cv_out["Evaluation"]["Evaluations"]["Scores"]) == 2
    # ), "Number of evaluations does not match number of repetitions."

    # # print(cv_out["Evaluation"]["Evaluations"]['Confusion Matrices'])
    # expected_conf_mats = {
    #     "Repetition": {
    #         "0": [[17, 1, 0], [0, 10, 1], [0, 1, 20]],
    #         "1": [[17, 1, 0], [1, 10, 0], [0, 1, 20]],
    #     }
    # }
    # for rep in range(2):
    #     np.testing.assert_equal(
    #         cv_out["Evaluation"]["Evaluations"]["Confusion Matrices"].get(
    #             f"Repetition.{rep}"
    #         ),
    #         expected_conf_mats["Repetition"][str(rep)],
    #     )

    # assert (
    #     cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["Class"]
    #     == pd.Series([0, 1, 2, 0, 1, 2])
    # ).all()
    # print(cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["TN"])
    # assert (
    #     cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["TN"]
    #     == pd.Series([32, 37, 28, 31, 37, 29])
    # ).all()
    # assert (
    #     np.round(
    #         cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["Balanced Accuracy"],
    #         decimals=5,
    #     )
    #     == np.round(
    #         pd.Series([0.972222, 0.928904, 0.958949, 0.956597, 0.928904, 0.976190]),
    #         decimals=5,
    #     )
    # ).all()

    # # Testing repetition predictions
    # print(cv_out["Outer Predictions"])
    # assert (
    #     len(cv_out["Outer Predictions"]) == 2
    # ), "Number of prediction sets does not match number of repetitions."
    # assert cv_out["Outer Predictions"][0].shape == (num_samples, 3)
    # np.testing.assert_almost_equal(
    #     cv_out["Outer Predictions"][0][4, 1:3], np.array([0.00752, 0.99248]), decimal=4
    # )

    # print(cv_out["Inner Results"][0])
    # print(cv_out["Inner Results"][0].columns)

    # # We cannot know the order of results a priori, due to the
    # # parallel saving to disk during runtime
    # # so we check the sorted values

    # # print(np.round(sorted(cv_out["Inner Results"][0]
    # #       ["split0_test_score"].tolist()), decimals=4).tolist())
    # assert all(
    #     np.round(
    #         sorted(cv_out["Inner Results"][0]["split0_test_score"].tolist()), decimals=4
    #     )
    #     == np.round(
    #         sorted(
    #             [
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.3333,
    #                 0.7667,
    #                 0.8222,
    #                 0.8889,
    #                 0.8889,
    #                 0.8889,
    #                 0.8889,
    #                 0.8889,
    #                 0.8889,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9167,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 0.9333,
    #                 1.0,
    #                 1.0,
    #                 1.0,
    #             ]
    #         ),
    #         decimals=4,
    #     )
    # )

    # print(
    #     np.round(
    #         cv_out["Inner Results"][0]["param_model__C"].tolist(), decimals=6
    #     ).tolist()
    # )
    # assert all(
    #     np.round(
    #         sorted(cv_out["Inner Results"][0]["param_model__C"].tolist()), decimals=4
    #     )
    #     == np.round(
    #         sorted(
    #             [
    #                 0.0001,
    #                 0.000825,
    #                 0.006813,
    #                 0.056234,
    #                 0.464159,
    #                 3.831187,
    #                 31.622777,
    #                 261.015722,
    #                 2154.43469,
    #                 17782.7941,
    #                 146779.926762,
    #                 1211527.658629,
    #                 10000000.0,
    #                 0.0001,
    #                 0.000825,
    #                 0.006813,
    #                 0.056234,
    #                 0.464159,
    #                 3.831187,
    #                 31.622777,
    #                 261.015722,
    #                 2154.43469,
    #                 17782.7941,
    #                 146779.926762,
    #                 1211527.658629,
    #                 10000000.0,
    #                 0.0001,
    #                 0.000825,
    #                 0.006813,
    #                 0.056234,
    #                 0.464159,
    #                 3.831187,
    #                 31.622777,
    #                 261.015722,
    #                 2154.43469,
    #                 17782.7941,
    #                 146779.926762,
    #                 1211527.658629,
    #                 10000000.0,
    #             ]
    #         ),
    #         decimals=4,
    #     )
    # )

    # assert np.unique(
    #     cv_out["Inner Results"][0]["outer_split (unordered)"]
    # ).tolist() == ["A", "B", "C"]


@ignore_warnings(category=ConvergenceWarning)
def test_train_full_model_binary_classification_weighting_per_split(
    xy_binary_classification_xl,
    UnbalancedLogisticRegressionClassifierPartial,
    create_splits_fn,
):
    # Unit tests to check the expected number of elements
    # in the output when adding groups with some train_only indicators

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    splits = np.sort(
        create_splits_fn(
            num_samples=num_samples,
            num_splits=5,
            id_names=["train_only(-1)", "ts2", "ts3", "ts4", "ts5"],
        )
    )  # Weird split names are for maintaining order
    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups[np.isin(groups, range(40, 50))] *= -1  # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # Randomly remove some elements to cause class and group imbalances
    rm_indices = np.random.choice(range(len(groups)), replace=False, size=11)
    splits = np.delete(splits, rm_indices, axis=0)
    groups = np.delete(groups, rm_indices, axis=0)
    x = np.delete(x, rm_indices, axis=0)
    y = np.delete(y, rm_indices, axis=0)

    # print([(s,g) for s,g in zip(splits, groups)])

    # Init model object
    model = UnbalancedLogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = UnbalancedLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

    modeling_out_1 = train_full_model(
        x=x,
        y=y,
        model=model,
        groups=groups,
        grid=grid,
        split=splits,
        positive=1,
        metric="balanced_accuracy",
        task="binary_classification",
        num_jobs=1,
        weight_loss_by_groups=False,  # NOTE: Not in the first test, easier to test
        weight_loss_by_class=True,
        weight_per_split=True,
    )

    print("Modeling 1: ", modeling_out_1)

    # With group weighting

    modeling_out_2 = train_full_model(
        x=x,
        y=y,
        model=model,
        groups=groups,
        grid=grid,
        split=splits,
        positive=1,
        metric="balanced_accuracy",
        task="binary_classification",
        num_jobs=1,
        weight_loss_by_groups=True,  # enabled
        weight_loss_by_class=True,
        weight_per_split=True,
    )

    print("Modeling 2: ", modeling_out_2)

    # *Without* splits per group (no group loss either)

    modeling_out_3 = train_full_model(
        x=x,
        y=y,
        model=model,
        groups=groups,
        grid=grid,
        split=splits,
        positive=1,
        metric="balanced_accuracy",
        task="binary_classification",
        num_jobs=1,
        weight_loss_by_groups=False,  # disabled
        weight_loss_by_class=True,
        weight_per_split=False,  # disabled
    )

    print("Modeling 3: ", modeling_out_3)

    print(
        modeling_out_1["Predictions"][:6].tolist(),
        modeling_out_2["Predictions"][:6].tolist(),
        modeling_out_3["Predictions"][:6].tolist(),
    )
    # Test that predictions differ for models with group-weighted loss
    assert any(modeling_out_1["Predictions"] != modeling_out_2["Predictions"])
    # Test that predictions differ for models with per-split weighting
    assert any(modeling_out_1["Predictions"] != modeling_out_3["Predictions"])
    assert_array_almost_equal(
        modeling_out_1["Predictions"][:6],
        np.asarray(
            [0.9940932, 0.8725881, 0.12918971, 0.14601526, 0.11770537, 0.98887163]
        ),
    )
    assert_array_almost_equal(
        modeling_out_2["Predictions"][:6],
        np.asarray(
            [
                0.9939877986907959,
                0.8719323873519897,
                0.13315576314926147,
                0.13463085889816284,
                0.11462106555700302,
                0.9872577786445618,
            ]
        ),
    )
    assert_array_almost_equal(
        modeling_out_3["Predictions"][:6],
        np.asarray(
            [
                0.9942026138305664,
                0.8753448724746704,
                0.12921789288520813,
                0.15208053588867188,
                0.11671973764896393,
                0.9892414808273315,
            ]
        ),
    )

    # Check that sample weights are passed through
    # Note: Fails on purpose
    test_sample_weights = False
    if test_sample_weights:
        print("\nLabel imbalances: ")
        print(
            "  NOTE: Sample weights are normalized in pipeline so these won't match completely but be somewhat close"
        )
        expected_weightings = {}
        for _split in np.unique(splits):
            expected_weightings[_split] = (
                np.unique(y[splits == _split], return_counts=True)[1]
                / np.unique(y[splits == _split], return_counts=True)[1].sum()
                * 2
            )
            print(
                "  ",
                _split,
                ": ",
                np.unique(y[splits == _split], return_counts=True),
                expected_weightings[_split],
                "\n  For: ",
                groups[splits == _split],
                "\n",
            )

        class LogisticRegression2(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    print("Sample Weights: ", sample_weight)
                return super().fit(X=X, y=y, sample_weight=sample_weight)

        model = LogisticRegression2(
            **{
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": None,
            },
            random_state=seed,
        )

        # Without group weighting
        print("\nClass but no group weighting:")
        train_full_model(
            x=x,
            y=y,
            model=model,
            groups=groups,
            grid=grid,
            split=splits,
            positive=1,
            metric="balanced_accuracy",
            task="binary_classification",
            num_jobs=1,
            weight_loss_by_groups=False,  # disabled
            weight_loss_by_class=True,
            weight_per_split=True,  # disabled
        )

        # Both
        print("\nBoth group and class weighting:")
        train_full_model(
            x=x,
            y=y,
            model=model,
            groups=groups,
            grid=grid,
            split=splits,
            positive=1,
            metric="balanced_accuracy",
            task="binary_classification",
            num_jobs=1,
            weight_loss_by_groups=True,  # disabled
            weight_loss_by_class=True,
            weight_per_split=True,  # disabled
        )

        assert False, "Check that sample weights were printed and disable test"
