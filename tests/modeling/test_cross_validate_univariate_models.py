import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from generalize.model.univariate.cross_validate_univariate_models import (
    cross_validate_univariate_models,
)


@ignore_warnings(category=ConvergenceWarning)
def test_univariate_cross_validate_univariate_models_binary_classification(
    xy_binary_classification,
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]
    num_samples = xy_binary_classification["num_samples"]
    num_reps = 2

    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        k=3,
        task="classification",
        reps=num_reps,
        standardize_cols=True,
        weight_loss_by_class=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    print("Finished cv")

    print(cv_out)

    # print(cv_out["Accuracy"].tolist())
    assert_array_almost_equal([0.72, 0.86, 0.89, 0.79, 0.81], cv_out["Accuracy"])
    assert_array_almost_equal(range(5), cv_out["Feature Index"])
    assert list(cv_out.columns) == [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "TP",
        "FP",
        "TN",
        "FN",
        "AUC",
        "Num Classes",
        "Positive Class",
        "Num Repetitions",
        "Feature Index",
    ]

    # Only 1 rep
    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        k=3,
        task="classification",
        reps=1,
        standardize_cols=True,
        weight_loss_by_class=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    assert_array_almost_equal([0.72, 0.86, 0.88, 0.78, 0.82], cv_out["Accuracy"])


@ignore_warnings(category=ConvergenceWarning)
def test_univariate_cross_validate_univariate_models_binary_classification_by_splits(
    xy_binary_classification_xl,
    create_splits_fn,
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification_xl["x"]
    y = xy_binary_classification_xl["y"]
    num_samples = xy_binary_classification_xl["num_samples"]

    splits = create_splits_fn(
        num_samples=num_samples, num_splits=3, id_names=["A", "B", "C"]
    )

    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        split=splits,
        eval_by_split=True,
        task="classification",
        reps=2,
        standardize_cols=True,
        weight_loss_by_class=True,
        weight_per_split=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    print("Finished cv")

    print(cv_out)

    # print(cv_out["Accuracy"].tolist())
    assert_array_almost_equal(
        [0.82, 0.793333, 0.853333, 0.753333, 0.833333], cv_out["Accuracy"]
    )
    assert_array_almost_equal(range(5), cv_out["Feature Index"])
    print(list(cv_out.columns))
    assert list(cv_out.columns) == [
        "AUC_A",
        "AUC_B",
        "AUC_C",
        "Accuracy_A",
        "Accuracy_B",
        "Accuracy_C",
        "Balanced Accuracy_A",
        "Balanced Accuracy_B",
        "Balanced Accuracy_C",
        "F1_A",
        "F1_B",
        "F1_C",
        "FN_A",
        "FN_B",
        "FN_C",
        "FP_A",
        "FP_B",
        "FP_C",
        "NPV_A",
        "NPV_B",
        "NPV_C",
        "PPV_A",
        "PPV_B",
        "PPV_C",
        "Sensitivity_A",
        "Sensitivity_B",
        "Sensitivity_C",
        "Specificity_A",
        "Specificity_B",
        "Specificity_C",
        "TN_A",
        "TN_B",
        "TN_C",
        "TP_A",
        "TP_B",
        "TP_C",
        "Threshold_A",
        "Threshold_B",
        "Threshold_C",
        "AUC",
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "FN",
        "FP",
        "NPV",
        "PPV",
        "Sensitivity",
        "Specificity",
        "TN",
        "TP",
        "Num Classes",
        "Positive Class",
        "Num Repetitions",
        "Feature Index",
    ]

    # Only 1 rep
    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        split=splits,
        eval_by_split=True,
        task="classification",
        reps=1,
        standardize_cols=True,
        weight_loss_by_class=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    assert_array_almost_equal(
        [0.82, 0.793333, 0.853333, 0.753333, 0.833333], cv_out["Accuracy"]
    )
    assert_array_almost_equal(range(5), cv_out["Feature Index"])
    print(list(cv_out.columns))
    assert list(cv_out.columns) == [
        "AUC_A",
        "AUC_B",
        "AUC_C",
        "Accuracy_A",
        "Accuracy_B",
        "Accuracy_C",
        "Balanced Accuracy_A",
        "Balanced Accuracy_B",
        "Balanced Accuracy_C",
        "F1_A",
        "F1_B",
        "F1_C",
        "FN_A",
        "FN_B",
        "FN_C",
        "FP_A",
        "FP_B",
        "FP_C",
        "NPV_A",
        "NPV_B",
        "NPV_C",
        "PPV_A",
        "PPV_B",
        "PPV_C",
        "Sensitivity_A",
        "Sensitivity_B",
        "Sensitivity_C",
        "Specificity_A",
        "Specificity_B",
        "Specificity_C",
        "TN_A",
        "TN_B",
        "TN_C",
        "TP_A",
        "TP_B",
        "TP_C",
        "Threshold_A",
        "Threshold_B",
        "Threshold_C",
        "AUC",
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "FN",
        "FP",
        "NPV",
        "PPV",
        "Sensitivity",
        "Specificity",
        "TN",
        "TP",
        "Num Classes",
        "Positive Class",
        "Num Repetitions",
        "Feature Index",
    ]


@ignore_warnings(category=ConvergenceWarning)
def test_univariate_cross_validate_univariate_models_multiclass_classification(
    xy_mc_classification,
):
    raise NotImplementedError("Multiclass univariate CV is not currently implemented")

    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_mc_classification["x"]
    y = xy_mc_classification["y"]
    num_reps = 2

    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        k=3,
        task="classification",
        reps=num_reps,
        standardize_cols=True,
        weight_loss_by_class=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    print("Finished cv")

    print(cv_out)
    assert False

    # print(cv_out["Accuracy"].tolist())
    assert_array_almost_equal([0.73, 0.87, 0.9, 0.78, 0.82], cv_out["Accuracy"])
    assert_array_almost_equal(range(5), cv_out["Feature Index"])
    assert list(cv_out.columns) == [
        "Accuracy",
        "Balanced Accuracy",
        "F1",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "TP",
        "FP",
        "TN",
        "FN",
        "AUC",
        "Num Repetitions",
        "Feature Index",
    ]

    # Only 1 rep
    cv_out = cross_validate_univariate_models(
        x=x,
        y=y,
        positive_label=1,
        k=3,
        task="classification",
        reps=1,
        standardize_cols=True,
        num_jobs=1,
        seed=seed,
        add_info_cols=True,
    )

    assert_array_almost_equal([0.74, 0.86, 0.88, 0.76, 0.82], cv_out["Accuracy"])
