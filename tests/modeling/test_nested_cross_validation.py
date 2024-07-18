
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from generalize.model.cross_validate.nested_cross_validation import nested_cross_validate
from generalize.model.cross_validate.grid_search import make_lowest_c_refit_strategy


# Models and data (xy) fixtures are located in modeling/conftest.py

# TODO Test first that the predictions etc. are returned in the right order
# so we know that targets and predictions are in the same order!
# E.g. make a simple dataset and make a deterministic model (e.g. x < 0.5 = 0 else 1)
# and check the or

#@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_deterministic(tmp_path):

    x_lo = np.zeros(shape=(30, 5), dtype=np.float32)
    x_lo[:, 4] = np.arange(30) / 1000
    x_hi = np.zeros(shape=(30, 5), dtype=np.float32)
    x_hi.fill(0.9)
    x_hi[:, 4] += np.arange(30) / 1000
    x = np.vstack(
        [x_lo, x_hi]
    )

    y = x.sum(axis=-1)

    class SumRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, a=None):
            self.a = a

        def fit(self, X, y, sample_weight=None):
            self.is_fitted_ = True
            warnings.warn(
                "`SumRegressor` warning catching test warning.",
                ConvergenceWarning
            )
            # `fit` should always return `self`
            return self

        def predict(self, X=None):
            return X.sum(axis=-1)

    model = SumRegressor()
    grid = {"model__a": [1, 2, 3]}

    # Ignore "divide by zero" warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_out = nested_cross_validate(
            x=x, y=y, model=model, grid=grid,
            k_outer=3, k_inner=3, inner_metric='neg_mean_squared_error',
            task='regression', reps=2, num_jobs=1,
            tmp_path=tmp_path,
        )

    print(cv_out)

    # These should be identical, if the returned
    # predictions have the correct order
    for rep in range(2):
        np.testing.assert_array_equal(cv_out["Outer Predictions"][rep], y)

    # Since they are identical, we should get MAE and RMSE of 0
    assert all(cv_out["Evaluation"]["Summary"]["Scores"]["MAE"] ==
               pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]))

    #### Pre-defined outer folds ####

    print("\n---------------------------------------------\n")

    # We should be able to call the folds what we want
    # Here: train_only(1), 2, 4
    # where the `train_only` wrapper means "always in train set"
    outer_folds = np.asarray(
        [["train_only(1)"] * 20, [str(2)] * 20, [str(4)] * 20]
    ).flatten()

    # Ignore "divide by zero" warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_out = nested_cross_validate(
            x=x, y=y, model=model, grid=grid,
            k_inner=3, outer_split=outer_folds,
            inner_metric='neg_mean_squared_error',
            task='regression', reps=2, num_jobs=1,
            tmp_path=tmp_path, cv_error_score="raise",
        )

    # These should be identical, if the returned
    # predictions have the correct order
    for rep in range(2):
        np.testing.assert_array_equal(
            cv_out["Outer Predictions"][rep], y[20:])
        np.testing.assert_array_equal(
            cv_out["Outer Indices"][rep], range(20, 60))
        np.testing.assert_array_equal(
            cv_out["Outer Splits"][rep],
            np.asarray([[str(2)] * 20, [str(4)] * 20]).flatten())

    # # TODO Test that we catch convergence warnings!


    raise

@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_multiclass_classification(xy_mc_classification, LogisticRegressionClassifierPartial, tmp_path):

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

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        k_outer=3, k_inner=3, inner_metric='balanced_accuracy',
        task='multiclass_classification', reps=2, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")
    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results
    print(cv_out["Evaluation"]["Summary"])
    assert cv_out["Evaluation"]["Summary"]["What"] == "Summary of 2 evaluations from multiclass classification."
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.96, 0.0000, 0.96, 0.96, 0.000000
    ]), decimals=5)).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Macro AUC"], decimals=5) == np.round(pd.Series([
        0.994196, 0.002994, 0.992079, 0.996313, 0.000000
    ]), decimals=5)).all()

    # Total (summed) confusion matrix
    # Sums to 2 x num_samples == 100

    #print(cv_out["Evaluation"]["Summary"]["Confusion Matrix"])
    np.testing.assert_equal(
        cv_out["Evaluation"]["Summary"]["Confusion Matrix"].get_counts(),
        np.array([[34, 2, 0],
                  [1, 20, 1],
                  [0, 2, 40]])
    )
    assert np.sum(
        cv_out["Evaluation"]["Summary"]["Confusion Matrix"].get_counts()
    ) == 2 * num_samples

    # Testing repetition evaluations
    print(cv_out["Evaluation"]["Evaluations"])
    assert cv_out["Evaluation"]["Evaluations"]["What"] == "Combined evaluations."
    assert len(cv_out["Evaluation"]["Evaluations"]["Scores"]) == 2, \
        "Number of evaluations does not match number of repetitions."

    #print(cv_out["Evaluation"]["Evaluations"]['Confusion Matrices'])
    expected_conf_mats = {
        "Repetition": {
            "0": [[17, 1, 0], [0, 10, 1], [0, 1, 20]],
            "1": [[17, 1, 0], [1, 10, 0], [0, 1, 20]]
        }
    }
    for rep in range(2):
        np.testing.assert_equal(
            cv_out["Evaluation"]["Evaluations"]['Confusion Matrices'].get(
                f"Repetition.{rep}"),
            expected_conf_mats["Repetition"][str(rep)]
        )

    assert (cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["Class"]
            == pd.Series([0, 1, 2, 0, 1, 2])).all()
    print(cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["TN"])
    assert (cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["TN"] ==
            pd.Series([32, 37, 28, 31, 37, 29])).all()
    assert (np.round(cv_out["Evaluation"]["Evaluations"]["One-vs-All"]["Balanced Accuracy"], decimals=5) == np.round(pd.Series([
        0.972222, 0.928904, 0.958949, 0.956597, 0.928904, 0.976190
    ]), decimals=5)).all()

    # Testing repetition predictions
    print(cv_out["Outer Predictions"])
    assert len(cv_out["Outer Predictions"]) == 2, \
        "Number of prediction sets does not match number of repetitions."
    assert cv_out["Outer Predictions"][0].shape == (num_samples, 3)
    np.testing.assert_almost_equal(
        cv_out["Outer Predictions"][0][4, 1:3],
        np.array([
            0.00752, 0.99248
        ]),
        decimal=4
    )

    print(cv_out["Inner Results"][0])
    print(cv_out["Inner Results"][0].columns)

    # We cannot know the order of results a priori, due to the
    # parallel saving to disk during runtime
    # so we check the sorted values

    # print(np.round(sorted(cv_out["Inner Results"][0]
    #       ["split0_test_score"].tolist()), decimals=4).tolist())
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["split0_test_score"].tolist()), decimals=4) ==
        np.round(sorted(
            [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333,
             0.3333, 0.3333, 0.7667, 0.8222, 0.8889, 0.8889, 0.8889, 0.8889, 0.8889, 0.8889,
             0.9167, 0.9167, 0.9167, 0.9167, 0.9167, 0.9167, 0.9167, 0.9167, 0.9333, 0.9333,
             0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 1.0, 1.0, 1.0]

        ),
            decimals=4)
    )

    print(np.round(cv_out["Inner Results"][0]
          ["param_model__C"].tolist(), decimals=6).tolist())
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["param_model__C"].tolist()), decimals=4) ==
        np.round(sorted(
            [0.0001, 0.000825, 0.006813, 0.056234, 0.464159, 3.831187, 31.622777, 261.015722,
             2154.43469, 17782.7941, 146779.926762, 1211527.658629, 10000000.0, 0.0001, 0.000825,
             0.006813, 0.056234, 0.464159, 3.831187, 31.622777, 261.015722, 2154.43469, 17782.7941,
             146779.926762, 1211527.658629, 10000000.0, 0.0001, 0.000825, 0.006813, 0.056234,
             0.464159, 3.831187, 31.622777, 261.015722, 2154.43469, 17782.7941, 146779.926762,
             1211527.658629, 10000000.0]),
            decimals=4)
    )

    assert (np.unique(cv_out["Inner Results"][0]["outer_split (unordered)"]).tolist() ==
            ['A', 'B', 'C'])


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_svm_multiclass_classification(xy_mc_classification, SVMClassifierPartial, tmp_path):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_mc_classification["x"]
    y = xy_mc_classification["y"]

    # Init model object
    model = SVMClassifierPartial["model"](random_state=seed)
    grid = SVMClassifierPartial["grid"]

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        k_outer=3, k_inner=3, inner_metric='balanced_accuracy',
        task='multiclass_classification', reps=2, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")
    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results
    print(cv_out["Evaluation"]["Summary"])
    assert cv_out["Evaluation"]["Summary"]["What"] == "Summary of 2 evaluations from multiclass classification."
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.98, 0.009428, 0.973333, 0.986667, 0.0
    ]), decimals=5)).all()
    # Count of NaNs
    assert cv_out["Evaluation"]["Summary"]["Scores"]["Macro AUC"][4] == 2.0


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification(xy_binary_classification, LogisticRegressionClassifierPartial, tmp_path):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]
    num_samples = xy_binary_classification["num_samples"]
    num_reps = 2

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_binary"]

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_outer=3, k_inner=3, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results

    assert cv_out["Evaluation"]["Summary"]["What"] == "Evaluation summaries from binary classification."
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Threshold Version'] == pd.Series([
        "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold", "High Specificity Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
        "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Measure'] == pd.Series([
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
    ])).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.97, 0.014142, 0.96, 0.98, 0.00,
        0.97, 0.014142, 0.96, 0.98, 0.00,
        0.95, 0.014142, 0.94, 0.96, 0.000000,
    ]), decimals=5)).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["AUC"], decimals=5) == np.round(pd.Series([
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
    ]), decimals=5)).all()

    # Total (summed) confusion matrices
    #print(cv_out["Evaluation"]["Summary"]["Confusion Matrices"])
    expected_conf_mats = {
        "Threshold Version": {
            "0_5 Threshold": np.array([[47, 3], [2, 48]]),
            "High Specificity Threshold": np.array([[49, 1], [2, 48]]),
            "Max_ J Threshold": np.array([[50, 0], [3, 47]])
        }
    }
    for thresh in ["High Specificity Threshold", "Max_ J Threshold", "0_5 Threshold"]:
        np.testing.assert_almost_equal(
            cv_out["Evaluation"]["Summary"]["Confusion Matrices"].get(
                f"Threshold Version.{thresh}"),
            expected_conf_mats["Threshold Version"][thresh]
        )

    # Testing repetition evaluations

    # Three thresholds times 2 repetitions == 6 rows
    assert len(cv_out["Evaluation"]["Evaluations"]["Scores"]) == 6, \
        "Number of evaluations does not match number of repetitions."
    assert cv_out["Evaluation"]["Evaluations"]["What"] == "Combined evaluations from 2 prediction sets and 3 thresholds from binary classification."
    # print(cv_out["Evaluation"]["Evaluations"]['Confusion Matrices'])
    expected_conf_mats = {
        "Threshold Version": {
            "0_5 Threshold": {
                "Repetition": {
                    "0": [[23, 2], [1, 24]], "1": [[24, 1], [1, 24]]
                }
            }, "High Specificity Threshold": {
                "Repetition": {
                    "0": [[25, 0], [1, 24]], "1": [[24, 1], [1, 24]]
                }
            }, "Max_ J Threshold": {
                "Repetition": {
                    "0": [[25, 0], [1, 24]], "1": [[25, 0], [2, 23]]
                }
            }
        }
    }
    for thresh in ["High Specificity Threshold", "Max_ J Threshold", "0_5 Threshold"]:
        for rep in range(num_reps):
            np.testing.assert_almost_equal(
                cv_out["Evaluation"]["Evaluations"]["Confusion Matrices"].get(
                    f"Threshold Version.{thresh}.Repetition.{rep}"),
                expected_conf_mats["Threshold Version"][thresh]["Repetition"][str(
                    rep)]
            )

    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Accuracy"] == pd.Series(
        [0.98, 0.96, 0.98, 0.96, 0.94, 0.96])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Repetition"] ==
            pd.Series([0, 1, 0, 1, 0, 1])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]['Threshold Version'] == pd.Series([
        "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
        "0.5 Threshold", "0.5 Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Positive Class"]
            == pd.Series([1, 1, 1, 1, 1, 1])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Num Classes"] ==
            pd.Series([2, 2, 2, 2, 2, 2])).all()

    # Testing repetition predictions
    assert len(cv_out["Outer Predictions"]) == 2, \
        "Number of prediction sets does not match number of repetitions."
    assert cv_out["Outer Predictions"][0].shape == (num_samples,)
    print(cv_out["Outer Predictions"][0][4])
    np.testing.assert_almost_equal(
        cv_out["Outer Predictions"][0][4],
        np.array([0.9999997]),
        decimal=5
    )

    np.testing.assert_almost_equal(
        cv_out['Outer Splits'][0],
        np.array([1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 0, 2, 2,
                  1, 1, 0, 1, 1, 2, 0, 2, 0, 2, 1, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0,
                  2, 1, 2, 2, 2, 0]),
        decimal=5
    )
    np.testing.assert_almost_equal(
        cv_out['Outer Splits'][1],
        np.array([0, 1, 0, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 1, 1, 2, 0, 0, 1, 1, 1, 0,
                  2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 2, 0, 1, 0, 0, 2, 2, 0, 0, 1, 0,
                  2, 0, 1, 2, 1, 1]),
        decimal=5
    )

    print(cv_out["Inner Results"][0].columns)
    # We cannot know the order of results a priori, due to the
    # parallel saving to disk during runtime
    # so we check the sorted values
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["split0_test_score"].tolist()), decimals=4) ==
        np.round(sorted([0.5, 0.5, 0.5, 0.9166666, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
                         0.9166666, 0.9166666, 0.9166666, 0.9166666, 0.9166666, 0.9166666,
                         0.9166666, 0.5, 0.5, 0.5, 0.9166666, 0.9166666, 1.0, 1.0, 1.0, 1.0, 1.0]),
                 decimals=4)
    )

    # Due to weird rounding (likely during saving to disk), we round to only 2 decimals
    # even though we would have preferred higher resolution
    assert all(np.round(sorted(cv_out["Inner Results"][0]["param_model__C"].tolist()), decimals=2) ==
               np.round(sorted(
                   [0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0,
                    0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0,
                    0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0]),
        decimals=2))

    assert (np.unique(cv_out["Inner Results"][0]["outer_split (unordered)"]).tolist() ==
            ['A', 'B', 'C'])
    
    # Using a custom refit strategy

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_outer=3, k_inner=3, inner_metric='balanced_accuracy', 
        refit=make_lowest_c_refit_strategy(c_name="model__C", score_name="balanced_accuracy"),
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print(cv_out["Inner Results"][0])
    assert False, "Remove this"


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_rf_binary_classification(xy_binary_classification, RandomForestClassifierPartial, tmp_path):

    # Simple regression test to ensure it runs

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]

    # Init model object
    model = RandomForestClassifierPartial["model"](random_state=seed)
    grid = RandomForestClassifierPartial["grid"]

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_outer=3, k_inner=3, inner_metric='balanced_accuracy',
        task='binary_classification', reps=2, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results

    assert cv_out["Evaluation"]["Summary"]["What"] == "Evaluation summaries from binary classification."
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Threshold Version'] == pd.Series([
        "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold", "High Specificity Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
        "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Measure'] == pd.Series([
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
    ])).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.96, 0.028284, 0.94, 0.98, 0.00,
        0.96, 0.028284, 0.94, 0.98, 0.00,
        0.94, 0.028284, 0.92, 0.96, 0.00,
    ]), decimals=5)).all()

    


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_svm_binary_classification(xy_binary_classification, SVMClassifierPartial, tmp_path):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]
    # num_samples = xy_binary_classification["num_samples"]

    # Init model object
    model = SVMClassifierPartial["model"](random_state=seed)
    grid = SVMClassifierPartial["grid"]

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_outer=3, k_inner=3, inner_metric='balanced_accuracy',
        task='binary_classification', reps=2, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results

    assert cv_out["Evaluation"]["Summary"]["What"] == "Evaluation summaries from binary classification."
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Threshold Version'] == pd.Series([
        "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold", "High Specificity Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
        "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Measure'] == pd.Series([
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
        "Average", "SD", "Min", "Max", "# NaNs",
    ])).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.96, 0.0, 0.96, 0.96, 0.0,
        0.96, 0.0, 0.96, 0.96, 0.0,
        0.96, 0.0, 0.96, 0.96, 0.0,
    ]), decimals=5)).all()



@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_regression(xy_regression, LassoLinearRegressionPartial, tmp_path):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    x = xy_regression["x"]
    y = xy_regression["y"]
    num_samples = xy_regression["num_samples"]

    seed = 15
    np.random.seed(seed)

    # Init model object
    model = LassoLinearRegressionPartial["model"](random_state=seed)
    grid = LassoLinearRegressionPartial["grid"]

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        k_outer=3, k_inner=3, inner_metric='neg_root_mean_squared_error',
        task='regression', reps=2, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")
    print(cv_out.keys())
    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results

    assert cv_out["Evaluation"]["Summary"]["What"] == "Summary of 2 evaluations from regression."
    print(cv_out["Evaluation"]["Summary"]["Scores"])
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["RMSE"], decimals=5) == np.round(pd.Series([
        0.201546, 0.015016, 0.190929, 0.212164, 0.000000
    ]), decimals=5)).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["MAE"], decimals=5) == np.round(pd.Series([
        0.158782, 0.006225, 0.154380, 0.163183, 0.000000
    ]), decimals=5)).all()

    # Testing repetition evaluations
    print(cv_out["Evaluation"]["Evaluations"]["Scores"])
    assert len(cv_out["Evaluation"]["Evaluations"]["Scores"]) == 2, \
        "Number of evaluations does not match number of repetitions."
    assert cv_out["Evaluation"]["Evaluations"]["What"] == "Combined evaluations."
    assert all(np.round(cv_out["Evaluation"]["Evaluations"]["Scores"]["RMSE"], decimals=4) == np.round(
        pd.Series([0.212164, 0.190929], dtype=np.float32), decimals=4))
    assert (np.round(cv_out["Evaluation"]["Evaluations"]["Scores"]["MAE"], decimals=4) == np.round(
        pd.Series([0.163183, 0.154380], dtype=np.float32), decimals=4)).all()

    # Testing repetition predictions
    assert len(cv_out["Outer Predictions"]) == 2, \
        "Number of prediction sets does not match number of repetitions."
    assert cv_out["Outer Predictions"][0].shape == (num_samples,)
    np.testing.assert_almost_equal(
        cv_out["Outer Predictions"][0][4:7],
        np.array([
            0.92863, 0.30982, 0.62884
        ]),
        decimal=4
    )

    print(cv_out["Inner Results"][0])
    print(cv_out["Inner Results"][0].columns)

    # We cannot know the order of results a priori, due to the
    # parallel saving to disk during runtime
    # so we check the sorted values

    # print(np.round(sorted(cv_out["Inner Results"][0]
    #       ["split0_test_score"].tolist()), decimals=4).tolist())
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["split0_test_score"].tolist()), decimals=4) ==
        np.round(sorted(
            [-0.3929, -0.3929, -0.3929, -0.3854, -0.3854, -0.3854, -0.3646, -0.3369,
             -0.3369, -0.3369, -0.3212, -0.3154, -0.278, -0.2608, -0.2415, -0.2348,
             -0.2336, -0.2316, -0.2311, -0.231, -0.231, -0.231, -0.231, -0.2164,
             -0.2112, -0.2112, -0.2111, -0.2108, -0.2102, -0.2093, -0.2091, -0.2077,
             -0.2004, -0.1965, -0.1954, -0.195, -0.1949, -0.1948, -0.1948]
        ),
            decimals=4)
    )

    # print(np.round(cv_out["Inner Results"][0]
    #       ["param_model__alpha"].tolist(), decimals=6).tolist())
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["param_model__alpha"].tolist()), decimals=5) ==
        np.round(sorted(
            [0.0001, 0.000223, 0.000497, 0.001107, 0.002466, 0.005496, 0.012247,
             0.027293, 0.060822, 0.13554, 0.302048, 0.673106, 1.5, 0.0001, 0.000223,
             0.000497, 0.001107, 0.002466, 0.005496, 0.012247, 0.027293, 0.060822,
             0.13554, 0.302048, 0.673106, 1.5, 0.0001, 0.000223, 0.000497, 0.001107,
             0.002466, 0.005496, 0.012247, 0.027293, 0.060822, 0.13554, 0.302048, 0.673106, 1.5]),
            decimals=5)
    )

    assert (np.unique(cv_out["Inner Results"][0]["outer_split (unordered)"]).tolist() ==
            ['A', 'B', 'C'])

@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_by_splits(xy_binary_classification_xl, LogisticRegressionClassifierPartial, tmp_path, create_splits_fn):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]
    num_reps = 2

    splits = create_splits_fn(num_samples=num_samples, num_splits=3, id_names=["A", "B", "C"])

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_binary"]
    
    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_inner=3, outer_split=splits, 
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out.keys())
    print(cv_out["Evaluation"].keys())
    print(cv_out["Evaluation"]["Summary"])
    
    # With train_only datasets

    splits = create_splits_fn(num_samples=num_samples, num_splits=3, id_names=["A", "B", "train_only(C)"])

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_inner=3, outer_split=splits, 
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out.keys())
    print(cv_out["Evaluation"].keys())
    print(cv_out["Evaluation"]["Summary"])
    raise
    print(cv_out)

    # Testing small subset of results
    # as evaluate() is (TODO) tested elsewhere

    # Testing summarized results

    assert cv_out["Evaluation"]["Summary"]["What"] == "Summary of 2 *summaries* from binary classification."
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Threshold Version'] == pd.Series([
        "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold", "0.5 Threshold",
        "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold", "High Specificity Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Summary"]["Scores"]['Measure'] == pd.Series([
        "Average", "SD", "Min", "Max", "# Total NaNs",
        "Average", "SD", "Min", "Max", "# Total NaNs",
        "Average", "SD", "Min", "Max", "# Total NaNs",
    ])).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["Accuracy"], decimals=5) == np.round(pd.Series([
        0.95, 0.014142, 0.94, 0.96, 0.000000,
        0.97, 0.014142, 0.96, 0.98, 0.00,
        0.97, 0.014142, 0.96, 0.98, 0.00,
    ]), decimals=5)).all()
    assert (np.round(cv_out["Evaluation"]["Summary"]["Scores"]["AUC"], decimals=5) == np.round(pd.Series([
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
        0.984, 0.009051, 0.9776, 0.9904, 0.00,
    ]), decimals=5)).all()

    # Total (summed) confusion matrices
    #print(cv_out["Evaluation"]["Summary"]["Confusion Matrices"])
    expected_conf_mats = {
        "Threshold Version": {
            "0_5 Threshold": np.array([[47, 3], [2, 48]]),
            "High Specificity Threshold": np.array([[49, 1], [2, 48]]),
            "Max_ J Threshold": np.array([[50, 0], [3, 47]])
        }
    }
    for thresh in ["0_5 Threshold", "High Specificity Threshold", "Max_ J Threshold"]:
        np.testing.assert_almost_equal(
            cv_out["Evaluation"]["Summary"]["Confusion Matrices"].get(
                f"Threshold Version.{thresh}"),
            expected_conf_mats["Threshold Version"][thresh]
        )

    # Testing repetition evaluations

    # Three thresholds times 2 repetitions == 6 rows
    assert len(cv_out["Evaluation"]["Evaluations"]["Scores"]) == 6, \
        "Number of evaluations does not match number of repetitions."
    assert cv_out["Evaluation"]["Evaluations"]["What"] == "Combined evaluations from 2 prediction sets and 3 thresholds from binary classification."
    # print(cv_out["Evaluation"]["Evaluations"]['Confusion Matrices'])
    expected_conf_mats = {
        "Threshold Version": {
            "0_5 Threshold": {
                "Repetition": {
                    "0": [[23, 2], [1, 24]], "1": [[24, 1], [1, 24]]
                }
            }, "High Specificity Threshold": {
                "Repetition": {
                    "0": [[25, 0], [1, 24]], "1": [[24, 1], [1, 24]]
                }
            }, "Max_ J Threshold": {
                "Repetition": {
                    "0": [[25, 0], [1, 24]], "1": [[25, 0], [2, 23]]
                }
            }
        }
    }
    for thresh in ["0_5 Threshold", "High Specificity Threshold", "Max_ J Threshold"]:
        for rep in range(num_reps):
            np.testing.assert_almost_equal(
                cv_out["Evaluation"]["Evaluations"]["Confusion Matrices"].get(
                    f"Threshold Version.{thresh}.Repetition.{rep}"),
                expected_conf_mats["Threshold Version"][thresh]["Repetition"][str(
                    rep)]
            )

    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Accuracy"] == pd.Series(
        [0.94, 0.96, 0.98, 0.96, 0.98, 0.96])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Repetition"] ==
            pd.Series([0, 1, 0, 1, 0, 1])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]['Threshold Version'] == pd.Series([
        "0.5 Threshold", "0.5 Threshold",
        "Max. J Threshold", "Max. J Threshold",
        "High Specificity Threshold", "High Specificity Threshold",
    ])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Positive Class"]
            == pd.Series([1, 1, 1, 1, 1, 1])).all()
    assert (cv_out["Evaluation"]["Evaluations"]["Scores"]["Num Classes"] ==
            pd.Series([2, 2, 2, 2, 2, 2])).all()

    # Testing repetition predictions
    assert len(cv_out["Outer Predictions"]) == 2, \
        "Number of prediction sets does not match number of repetitions."
    assert cv_out["Outer Predictions"][0].shape == (num_samples,)
    print(cv_out["Outer Predictions"][0][4])
    np.testing.assert_almost_equal(
        cv_out["Outer Predictions"][0][4],
        np.array([0.9999997]),
        decimal=5
    )

    np.testing.assert_almost_equal(
        cv_out['Outer Splits'][0],
        np.array([1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 1, 1, 0, 2, 2,
                  1, 1, 0, 1, 1, 2, 0, 2, 0, 2, 1, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0,
                  2, 1, 2, 2, 2, 0]),
        decimal=5
    )
    np.testing.assert_almost_equal(
        cv_out['Outer Splits'][1],
        np.array([0, 1, 0, 0, 1, 2, 0, 2, 2, 1, 0, 1, 1, 1, 1, 2, 0, 0, 1, 1, 1, 0,
                  2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 2, 0, 1, 0, 0, 2, 2, 0, 0, 1, 0,
                  2, 0, 1, 2, 1, 1]),
        decimal=5
    )

    print(cv_out["Inner Results"][0].columns)
    # We cannot know the order of results a priori, due to the
    # parallel saving to disk during runtime
    # so we check the sorted values
    assert all(
        np.round(sorted(cv_out["Inner Results"][0]["split0_test_score"].tolist()), decimals=4) ==
        np.round(sorted([0.5, 0.5, 0.5, 0.9166666, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
                         0.9166666, 0.9166666, 0.9166666, 0.9166666, 0.9166666, 0.9166666,
                         0.9166666, 0.5, 0.5, 0.5, 0.9166666, 0.9166666, 1.0, 1.0, 1.0, 1.0, 1.0]),
                 decimals=4)
    )

    # Due to weird rounding (likely during saving to disk), we round to only 2 decimals
    # even though we would have preferred higher resolution
    assert all(np.round(sorted(cv_out["Inner Results"][0]["param_model__C"].tolist()), decimals=2) ==
               np.round(sorted(
                   [0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0,
                    0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0,
                    0.001, 0.00774, 0.05995, 0.46416, 3.59381, 27.82559, 215.44347, 1668.10054, 12915.49665, 100000.0]),
        decimals=2))

    assert (np.unique(cv_out["Inner Results"][0]["outer_split (unordered)"]).tolist() ==
            ['A', 'B', 'C'])


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_with_confounders(xy_binary_classification_xl, LogisticRegressionClassifierPartial, tmp_path, create_splits_fn):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]
    num_reps = 2

    splits = create_splits_fn(num_samples=num_samples, num_splits=3, id_names=["A", "B", "C"])

    from sklearn.preprocessing import OneHotEncoder
    
    # One-hot-encode splits
    enc = OneHotEncoder() 
    # Note: Sparse output
    confounders = enc.fit_transform(np.expand_dims(splits, axis=1)).toarray()

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_binary"]
    
    np.random.seed(seed)
    cv_out_with = nested_cross_validate(
        x=x, y=y, confounders=confounders, model=model, grid=grid,
        positive=1, k_inner=3, outer_split=splits, 
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print(cv_out_with.keys())
    print(cv_out_with["Evaluation"].keys())
    print(cv_out_with["Evaluation"]["Summary"])
    
    np.random.seed(seed)
    cv_out_without = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_inner=3, outer_split=splits, 
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    # Test it makes a difference in predictions
    assert (cv_out_with['Outer Predictions'][0] != cv_out_without['Outer Predictions'][0]).any()

    assert False, "Needs more rigorous testing"


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_with_pca(xy_binary_classification_xl, LogisticRegressionClassifierPartial, tmp_path):

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    # num_samples = xy_binary_classification_xl["num_samples"]
    num_reps = 2

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_binary"]

    transformers = [
        ("pca", PCA(n_components=5))
    ]

    seed = 15

    cv_out_with_pca = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_inner=3, transformers=transformers,
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    seed = 15

    cv_out_without_pca = nested_cross_validate(
        x=x, y=y, model=model, grid=grid,
        positive=1, k_inner=3,
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path
    )

    print("Finished cv")

    print("With PCA")
    print(cv_out_with_pca.keys())
    print(cv_out_with_pca["Evaluation"].keys())
    print(cv_out_with_pca["Evaluation"]["Summary"])

    print("Without PCA")
    print(cv_out_without_pca.keys())
    print(cv_out_without_pca["Evaluation"].keys())
    print(cv_out_without_pca["Evaluation"]["Summary"])

    # Tests that they differ
    assert not (
        cv_out_with_pca["Evaluation"]["Summary"]["Scores"]["Accuracy"] == 
        cv_out_without_pca["Evaluation"]["Summary"]["Scores"]["Accuracy"]
    ).all()
    
    # Test that the PCA was applied the same way (at least that the same metrics were calculated)
    # print(cv_out_with_pca["Evaluation"]["Summary"]["Scores"]["Accuracy"])
    np.testing.assert_almost_equal(cv_out_with_pca["Evaluation"]["Summary"]["Scores"]["Accuracy"], [
        0.993333,0.009428,0.986667,1.000000,0.000000,
        0.993333,0.009428,0.986667,1.000000,0.000000,
        0.983333,0.004714,0.980000,0.986667,0.000000,
    ], decimal=4)


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_with_groups(xy_binary_classification_xl, LogisticRegressionClassifierPartial, tmp_path, create_splits_fn):

    # Unit tests to check the expected number of elements
    # in the output when adding groups with some train_only indicators

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    num_reps = 2

    splits = np.sort(create_splits_fn(num_samples=num_samples, num_splits=5, id_names=["train_only(-1)", "ts2", "ts3", "ts4", "ts5"])) # Weird split names are for maintaining order
    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups[np.isin(groups, range(40, 50))] *= -1 # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # print([(s,g) for s,g in zip(splits, groups)])

    # Init model object
    model = LogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = LogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=3, k_outer=3,
        eval_by_split=False, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, 
        num_jobs=1, seed=seed, tmp_path=tmp_path, 
        cv_error_score="raise"
    )

    # print(cv_out)

    # We set 10 groups of 3 to train_only
    assert len(cv_out['Outer Predictions'][0]) == num_samples-10*3
    assert len(cv_out['Outer Indices'][0]) == num_samples-10*3
    assert len(cv_out['Outer Splits'][0]) == num_samples-10*3
    assert len(cv_out['Outer Groups'][0]) == num_samples-10*3
    assert len(cv_out['Outer Targets'][0]) == num_samples-10*3

    # None of the train-only indices should be present in outer indices
    assert not set(cv_out['Outer Indices'][0]).intersection(set(np.where(groups == -1)[0]))
    
    # We should just have 3 splits as that's what was specified
    assert set(cv_out['Outer Splits'][0]) == {0,1,2}

    assert len(cv_out["Inner Results"][0]) == 30 # 3 * 10 hparam combos

    initial_bal_acc = cv_out["Evaluation"]["Summary"]["Scores"]["Balanced Accuracy"]
    assert_array_almost_equal(
        initial_bal_acc,
        [  
            0.9758064516129032, 0.0, 0.9758064516129032, 0.9758064516129032, 0.0, 
            0.9709399332591769, 0.006095748113677107, 0.9666295884315907, 0.9752502780867631, 0.0,
            0.9671857619577309, 0.0, 0.9671857619577309, 0.9671857619577309, 0.0, 
        ]
    )

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(cv_out["Inner Results"][0])

    # WITH outer splits

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=3, k_outer=3, outer_split=splits,
        eval_by_split=True, inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, num_jobs=1, seed=seed,
        tmp_path=tmp_path, cv_error_score="raise"
    )

    # print(cv_out)

    # The first outer split is train-only 
    # We further set 10 groups of 3 to train_only
    # So we expect num samples - 30 - 30
    assert len(cv_out['Outer Predictions'][0]) == num_samples - 10*3 - num_samples / 5
    assert len(cv_out['Outer Indices'][0]) == num_samples - 10*3 - num_samples / 5
    assert len(cv_out['Outer Splits'][0]) == num_samples -10*3 - num_samples / 5

    # None of the train-only indices should be present in outer indices
    assert not set(cv_out['Outer Indices'][0]).intersection(set(np.where(groups == -1)[0]))
    
    # 'D' should not be in outer splits (are all train-only)
    assert set(cv_out['Outer Splits'][0]) == {'ts2', 'ts3', 'ts4'}

    # Average probabilities per group

    cv_out = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups, 
        weight_loss_by_groups=True,
        positive=1, k_inner=3, k_outer=3,
        eval_by_split=False, aggregate_by_groups=True, 
        inner_metric='balanced_accuracy',
        task='binary_classification', reps=num_reps, 
        num_jobs=1, seed=seed, tmp_path=tmp_path, 
        cv_error_score="raise"
    )

    # print(cv_out)

    # We set 10 groups of 3 to train_only
    assert len(cv_out['Outer Predictions'][0]) == num_samples-10*3
    assert len(cv_out['Outer Indices'][0]) == num_samples-10*3
    assert len(cv_out['Outer Splits'][0]) == num_samples-10*3
    assert len(cv_out['Outer Groups']) == num_samples-10*3
    assert len(cv_out['Outer Targets']) == num_samples-10*3

    # None of the train-only indices should be present in outer indices
    assert not set(cv_out['Outer Indices'][0]).intersection(set(np.where(groups == -1)[0]))
    
    # We should just have 3 splits as that's what was specified
    assert set(cv_out['Outer Splits'][0]) == {0,1,2}

    assert len(cv_out["Inner Results"][0]) == 30 # 3 * 10 hparam combos

    print(cv_out["Evaluation"]["Summary"]["Scores"]["Balanced Accuracy"].tolist())

    # Test not != initial_bal_acc
    assert_array_almost_equal(
        cv_out["Evaluation"]["Summary"]["Scores"]["Balanced Accuracy"],
        [
            0.7416879795396419, 0.0054253717226589475, 0.7378516624040921, 0.7455242966751918, 
            0.0, 0.7781329923273657, 0.01537188654753367, 0.7672634271099744, 0.789002557544757, 
            0.0, 0.6521739130434783, 0.03074377309506726, 0.6304347826086957, 0.6739130434782609, 0.0
        ]
    )

    # Check that sample weights are passed through 
    # Note: Fails on purpose
    test_sample_weights = False
    if test_sample_weights:

        class LogisticRegression2(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    print("Sample Weights: ", sample_weight)
                return super().fit(X=X,y=y, sample_weight=sample_weight)
            
        model = LogisticRegression2(
            **{
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": "balanced"
            }, random_state=seed)
            
        cv_out = nested_cross_validate(
            x=x, y=y, model=model, grid=grid, groups=groups, 
            weight_loss_by_groups=True,
            positive=1, k_inner=3, k_outer=3,
            eval_by_split=False, aggregate_by_groups=True,  
            inner_metric='balanced_accuracy',
            task='binary_classification', reps=1, 
            num_jobs=1, seed=seed, tmp_path=tmp_path, 
            cv_error_score="raise"
        )

        assert False, "Check that sample weights were printed and disable test"


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_train_only_paths(xy_binary_classification_xl, SavingLogisticRegressionClassifierPartial, tmp_path):

    # Unit tests to check the number of samples passed to fit and predict
    # are correct when there are groups and train_only indicators

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    #num_reps = 2

    # Set train_only groups
    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups[np.isin(groups, range(40, 50))] *= -1 # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # print([(s,g) for s,g in zip(splits, groups)])

    # Init model object
    model = SavingLogisticRegressionClassifierPartial["model"](random_state=seed, save_path=tmp_path)
    grid = SavingLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

     # Init model object
    model = SavingLogisticRegressionClassifierPartial["model"](random_state=seed, save_path=tmp_path)
    
    # Single C only
    grid = SavingLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

    cv_out = nested_cross_validate(  # noqa: F841
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=2, k_outer=2, reps=1, # k_inner=3, k_outer=3,
        eval_by_split=False, inner_metric='balanced_accuracy',
        task='binary_classification', # reps=num_reps, 
        num_jobs=1, seed=seed,
        tmp_path=tmp_path, cv_error_score="raise"
    )

    # Test that the model didn't predict on train_only in *outer* loop
    outer_test_X_paths = list(tmp_path.glob("**/predict_proba/X.npy"))
    outer_random_dirs = [path.parent.parent.name for path in outer_test_X_paths]
    print(outer_test_X_paths)
    assert len(outer_test_X_paths) == 2
    test_X_total_length = sum([len(np.load(path)) for path in outer_test_X_paths])
    assert test_X_total_length == (num_samples - 30), (
        "model predicted on the wrong number of samples: "
        f"{test_X_total_length} != {(num_samples - 30)}"
    )

    all_test_X_paths = list(tmp_path.glob("**/predict/X.npy"))
    print(outer_random_dirs, all_test_X_paths)
    exp_num_all_X_paths = 2 * 2 + 2 # inner * outer + outer
    assert len(all_test_X_paths) == exp_num_all_X_paths

    inner_test_X_paths = [
        path for path in all_test_X_paths 
        if path.parent.parent.name not in outer_random_dirs
    ]
    exp_num_inner_X_paths = 2 * 2 # inner * outer
    assert len(inner_test_X_paths) == exp_num_inner_X_paths
    inner_test_X_total_length = sum([len(np.load(path)) for path in inner_test_X_paths])
    assert inner_test_X_total_length == (num_samples - 30), (
        "model predicted on the wrong number of samples: "
        f"{inner_test_X_total_length} != {(num_samples - 30)}"
    )

    # Test that the model didn't predict on train_only in *inner* loop
    # Note: predict() is called many times (186; perhaps during optimization?)
    # so we can't rely on that
    all_train_y_paths = list(tmp_path.glob("**/fit/y.npy"))
    exp_num_y_paths = 2 * 2 + 2 # inner * outer + outer
    assert len(all_train_y_paths) == exp_num_y_paths


@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_splits_train_only_groups_paths(xy_binary_classification_xl, SavingLogisticRegressionClassifierPartial, tmp_path, create_splits_fn):

    # Unit tests to check that the correct samples are passed
    # to fit and predict when there's a combination of train_only
    # indicators in groups and outer_split - also tests
    # that groups are respected during folding

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    # num_reps = 2

    splits = np.sort(create_splits_fn(num_samples=num_samples, num_splits=5, id_names=["train_only(-1)", "ts2", "ts3", "ts4", "ts5"])) # Weird split names are for maintaining order
    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups[np.isin(groups, range(40, 50))] *= -1 # All of "D" split
    exp_train_only = groups.copy() # See below - must be copied before string conversion
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])
    
    # Set first feature to the expected train only indicators and groups otherwise
    # to allow checking the grouping
    exp_train_only[np.isin(exp_train_only, range(0, 10))] = -1
    x[:, 0] = exp_train_only
        
    # print([(s,g) for s,g in zip(splits, groups)])

    # Init model object
    model = SavingLogisticRegressionClassifierPartial["model"](random_state=seed, save_path=tmp_path)
    grid = SavingLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

     # Init model object
    model = SavingLogisticRegressionClassifierPartial["model"](random_state=seed, save_path=tmp_path)
    
    # Single C only
    grid = SavingLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

    cv_out = nested_cross_validate(  # noqa: F841
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=2, reps=1, outer_split=splits,
        eval_by_split=False, inner_metric='balanced_accuracy',
        task='binary_classification', 
        num_jobs=1, seed=seed,
        tmp_path=tmp_path, cv_error_score="raise"
    )

    # Test number of files

    # Test that the model didn't predict on train_only in *outer* loop
    outer_test_X_paths = list(tmp_path.glob("**/predict_proba/X.npy"))
    outer_random_dirs = [path.parent.parent.name for path in outer_test_X_paths]
    print(outer_test_X_paths)
    assert len(outer_test_X_paths) == 3
    test_Xs = [np.load(path) for path in outer_test_X_paths]
    test_X_total_length = sum([len(x_) for x_ in test_Xs])
    assert test_X_total_length == (num_samples - 60), (
        "model predicted on the wrong number of samples: "
        f"{test_X_total_length} != {(num_samples - 60)}"
    )

    all_test_X_paths = list(tmp_path.glob("**/predict/X.npy"))
    print(outer_random_dirs, all_test_X_paths)
    exp_num_all_X_paths = 2 * 3 + 3 # inner * outer + outer
    assert len(all_test_X_paths) == exp_num_all_X_paths

    inner_test_X_paths = [
        path for path in all_test_X_paths 
        if path.parent.parent.name not in outer_random_dirs
    ]
    exp_num_inner_X_paths = 2 * 3 # inner * outer
    assert len(inner_test_X_paths) == exp_num_inner_X_paths
    inner_test_X_total_length = sum([len(np.load(path)) for path in inner_test_X_paths])
    # There are 6 (3 outer * 2 inner) inner train/test iterations
    # 60 are always in train, so only 60 are tested on in inner cv per 
    # outer iteration (as 30 are test in outer)
    # giving us 3*60=180 total samples tested on in inner cv
    assert inner_test_X_total_length == 180, (
        "model predicted on the wrong number of samples: "
        f"{inner_test_X_total_length} != {180}"
    )

    # Test that the model didn't predict on train_only in *inner* loop
    # Note: predict() is called many times (186; perhaps during optimization?)
    # so we can't rely on that
    all_train_y_paths = list(tmp_path.glob("**/fit/y.npy"))
    exp_num_y_paths = 2 * 3 + 3 # inner * outer + outer
    assert len(all_train_y_paths) == exp_num_y_paths

    # Test that the correct groups are in the fitting data
    test_Xs_groups = [x_[:, 0] for x_ in test_Xs]
    for grouping in test_Xs_groups:
        assert (grouping >= 0).all(), (
            "Found train_only groups in the outer test data: "
            f"{grouping}"
        )
        counts = np.unique(grouping.flatten(), return_counts=True)[1]
        assert (counts == 3).all(), \
            "Not all groups had all 3 observations in the test data."

@ignore_warnings(category=ConvergenceWarning)
def test_nested_cross_validate_binary_classification_weighting_per_split(xy_binary_classification_xl, UnbalancedLogisticRegressionClassifierPartial, tmp_path, create_splits_fn):

    # Unit tests to check the expected number of elements
    # in the output when adding groups with some train_only indicators

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    num_reps = 2

    splits = np.sort(create_splits_fn(num_samples=num_samples, num_splits=5, id_names=["train_only(-1)", "ts2", "ts3", "ts4", "ts5"])) # Weird split names are for maintaining order
    groups = np.sort(np.concatenate([np.arange(num_samples // 3) for _ in range(3)]))
    groups[np.isin(groups, range(40, 50))] *= -1 # All of "D" split
    groups = np.asarray([f"train_only({-g})" if g < 0 else str(g) for g in groups])

    # Randomly remove some elements to cause class and group imbalances
    rm_indices = np.random.choice(range(len(groups)), replace=False, size=11)
    splits=np.delete(splits, rm_indices, axis=0)
    groups=np.delete(groups, rm_indices, axis=0)
    x=np.delete(x, rm_indices, axis=0)
    y=np.delete(y, rm_indices, axis=0)

    # print([(s,g) for s,g in zip(splits, groups)])

    # Init model object
    model = UnbalancedLogisticRegressionClassifierPartial["model"](random_state=seed)
    grid = UnbalancedLogisticRegressionClassifierPartial["grid_binary"]

    seed = 15

    cv_out_1 = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=3, outer_split=splits,
        eval_by_split=True, 
        inner_metric='balanced_accuracy',
        weight_loss_by_groups=False, # NOTE: Not in the first test, easier to test
        weight_loss_by_class=True,
        weight_per_split=True,
        task='binary_classification', reps=num_reps, 
        num_jobs=1, seed=seed, tmp_path=tmp_path, 
        cv_error_score="raise"
    )

    print("CV1: ",cv_out_1)

    # With group weighting

    cv_out_2 = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=3, outer_split=splits,
        eval_by_split=True, 
        inner_metric='balanced_accuracy',
        weight_loss_by_groups=True,
        weight_loss_by_class=True,
        weight_per_split=True,
        task='binary_classification', reps=num_reps, 
        num_jobs=1, seed=seed, tmp_path=tmp_path, 
        cv_error_score="raise"
    )

    print("CV2: ",cv_out_2)

    # *Without* splits per group (no group loss either)

    cv_out_3 = nested_cross_validate(
        x=x, y=y, model=model, grid=grid, groups=groups,
        positive=1, k_inner=3, outer_split=splits,
        eval_by_split=True, 
        inner_metric='balanced_accuracy',
        weight_loss_by_groups=False, # NOTE: Not in the first test, easier to test
        weight_loss_by_class=True,
        weight_per_split=False,
        task='binary_classification', reps=num_reps, 
        num_jobs=1, seed=seed, tmp_path=tmp_path, 
        cv_error_score="raise"
    )

    print("CV3: ", cv_out_3)

    print(cv_out_1["Outer Predictions"][0][:6].tolist(), cv_out_2["Outer Predictions"][0][:6].tolist(), cv_out_3["Outer Predictions"][0][:6].tolist())
    # Test that predictions differ for models with group-weighted loss
    assert any(cv_out_1["Outer Predictions"][0] != cv_out_2["Outer Predictions"][0])
    # Test that predictions differ for models with per-split weighting
    assert any(cv_out_1["Outer Predictions"][0] != cv_out_3["Outer Predictions"][0])
    assert_array_almost_equal(cv_out_1["Outer Predictions"][0][:6], np.asarray([0.06628229, 0.08877924, 0.06218732, 0.998869, 0.99940634, 0.93594605]))
    assert_array_almost_equal(cv_out_2["Outer Predictions"][0][:6], np.asarray([0.99940139, 0.93667513, 0.06915096, 0.07947106, 0.05905978, 0.998629]))
    assert_array_almost_equal(cv_out_3["Outer Predictions"][0][:6], np.asarray([0.9994131,  0.93834025, 0.06838495, 0.09622831, 0.06066705, 0.9988845]))


    # Check that sample weights are passed through 
    # Note: Fails on purpose
    test_sample_weights = False
    if test_sample_weights:

        print("\nLabel imbalances: ")
        print("  NOTE: Sample weights are normalized in pipeline so these won't match completely but be somewhat close")
        expected_weightings = {}
        for _split in np.unique(splits):
            expected_weightings[_split] = np.unique(y[splits == _split], return_counts=True)[1]/np.unique(y[splits == _split], return_counts=True)[1].sum()*2
            print("  ", _split, ": ", np.unique(y[splits == _split], return_counts=True), expected_weightings[_split], "\n  For: ", groups[splits == _split], "\n")

        class LogisticRegression2(LogisticRegression):
            def fit(self, X, y, sample_weight=None):
                if sample_weight is not None:
                    print("Sample Weights: ", sample_weight)
                return super().fit(X=X,y=y, sample_weight=sample_weight)
            
        model = LogisticRegression2(
            **{
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": None
            }, random_state=seed)

        # Without group weighting
        print("\nClass but no group weighting:")
        nested_cross_validate(
            x=x, y=y, model=model, grid=grid, groups=groups, 
            weight_loss_by_groups=False,
            weight_loss_by_class=True,
            weight_per_split=True,
            positive=1, k_inner=2, outer_split=splits,
            eval_by_split=False, aggregate_by_groups=False,  
            inner_metric='balanced_accuracy',
            task='binary_classification', reps=1, 
            num_jobs=1, seed=seed, tmp_path=tmp_path, 
            cv_error_score="raise"
        )

        # Both
        print("\nBoth group and class weighting:")
        nested_cross_validate(
            x=x, y=y, model=model, grid=grid, groups=groups, 
            weight_loss_by_groups=True,
            weight_loss_by_class=True,
            weight_per_split=True,
            positive=1, k_inner=2, outer_split=splits,
            eval_by_split=False, aggregate_by_groups=True,  
            inner_metric='balanced_accuracy',
            task='binary_classification', reps=1, 
            num_jobs=1, seed=seed, tmp_path=tmp_path, 
            cv_error_score="raise"
        )

        assert False, "Check that sample weights were printed and disable test"
