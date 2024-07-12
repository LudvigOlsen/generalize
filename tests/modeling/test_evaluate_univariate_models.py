import numpy as np
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from generalize.model.univariate.evaluate_univariate_models import (
    evaluate_univariate_models,
)


@ignore_warnings(category=ConvergenceWarning)
def test_evaluate_univariate_models_binary_classification(
    xy_binary_classification,
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]

    evaluations, _ = evaluate_univariate_models(
        x=x,
        y=y,
        task="classification",
        names=["A", "B", "C", "D", "E"],
        k=3,
        positive_label=1,
    )

    print("Finished cv")

    pd.set_option("display.max_columns", None)
    print(evaluations)

    # print(evaluations.to_dict())

    expected_evaluations = pd.DataFrame(
        {
            "name": {0: "E", 1: "A", 2: "D", 3: "B", 4: "C"},
            "Feature Set": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "intercept": {
                0: 0.00019933251350379359,
                1: 0.003407838828131782,
                2: -0.08862073021084887,
                3: 0.2638078477906434,
                4: -0.03623329305292816,
            },
            "slope": {
                0: 2.1742711270622586,
                1: 1.5991262156820405,
                2: 2.509606675470246,
                3: 3.1059862649173047,
                4: 4.159415275879349,
            },
            "std_err": {
                0: 0.5783896130439601,
                1: 0.4766175568062183,
                2: 0.7564250989491418,
                3: 0.9429032299147799,
                4: 1.2884742362545987,
            },
            "p_value": {
                0: 0.0008523530939088999,
                1: 0.003966013213832722,
                2: 0.004537770718548503,
                3: 0.0049374506486803225,
                4: 0.006229221903006162,
            },
            "neg_log10_p_value": {
                0: 3.069380458086232,
                1: 2.401645843160198,
                2: 2.3431574516385645,
                3: 2.3064972322491135,
                4: 2.205566198017405,
            },
            "significant": {0: True, 1: True, 2: True, 3: True, 4: True},
            "R2": {
                0: 0.40103510322189406,
                1: 0.2712646252915959,
                2: 0.40155440711987467,
                3: 0.49610649295883624,
                4: 0.6313631137719539,
            },
            "Accuracy": {0: 0.82, 1: 0.74, 2: 0.8, 3: 0.86, 4: 0.9},
            "Balanced Accuracy": {
                0: 0.8200000000000001,
                1: 0.74,
                2: 0.8,
                3: 0.86,
                4: 0.9,
            },
            "F1": {
                0: 0.8085106382978724,
                1: 0.7719298245614036,
                2: 0.7916666666666667,
                3: 0.8627450980392156,
                4: 0.888888888888889,
            },
            "Sensitivity": {0: 0.76, 1: 0.88, 2: 0.76, 3: 0.88, 4: 0.8},
            "Specificity": {0: 0.88, 1: 0.6, 2: 0.84, 3: 0.84, 4: 1.0},
            "PPV": {
                0: 0.8636363636363636,
                1: 0.6875,
                2: 0.8260869565217391,
                3: 0.8461538461538461,
                4: 1.0,
            },
            "NPV": {
                0: 0.7857142857142857,
                1: 0.8333333333333334,
                2: 0.7777777777777778,
                3: 0.875,
                4: 0.8333333333333334,
            },
            "TP": {0: 19, 1: 22, 2: 19, 3: 22, 4: 20},
            "FP": {0: 3, 1: 10, 2: 4, 3: 4, 4: 0},
            "TN": {0: 22, 1: 15, 2: 21, 3: 21, 4: 25},
            "FN": {0: 6, 1: 3, 2: 6, 3: 3, 4: 5},
            "AUC": {
                0: 0.8896000000000001,
                1: 0.7776000000000001,
                2: 0.8720000000000001,
                3: 0.9119999999999999,
                4: 0.9616,
            },
            "Positive Class": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
            "Num Classes": {0: 2, 1: 2, 2: 2, 3: 2, 4: 2},
            "tree_importance": {
                0: 0.07605095692895723,
                1: 0.040151460886448775,
                2: 0.12427293047639679,
                3: 0.16348593485424373,
                4: 0.5960387168539535,
            },
            "group": {0: 2, 1: 2, 2: 2, 3: 2, 4: 2},
            "significance_threshold": {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05},
            "num_tests": {0: 5, 1: 5, 2: 5, 3: 5, 4: 5},
            "abs_slope": {
                0: 2.1742711270622586,
                1: 1.5991262156820405,
                2: 2.509606675470246,
                3: 3.1059862649173047,
                4: 4.159415275879349,
            },
            "array_index": {0: 4, 1: 0, 2: 3, 3: 1, 4: 2},
        }
    )

    for col in expected_evaluations.columns:
        assert (evaluations[col] == expected_evaluations[col]).all()


# TODO: Test with regression
