import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from generalize.model.univariate.fit_statsmodels_model import (
    fit_statsmodels_model,
    fit_statsmodels_univariate_models,
)


@ignore_warnings(category=ConvergenceWarning)
def test_fit_statsmodels_model_binary_classification(
    xy_binary_classification,
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]

    modeling_out = fit_statsmodels_model(
        x=(x[:, 0] - np.mean(x[:, 0])) / np.std(x[:, 0]),
        y=y,
        task="classification",
        add_intercept=True,
        rm_missing=False,
    )

    print("Finished cv")

    print(modeling_out)

    params, pvalues, rsquared, slope_std_err = modeling_out
    assert_array_almost_equal(params, [0.00340784, 1.59912622])
    assert_array_almost_equal(pvalues, [9.92122087e-01, 7.93202643e-04])
    assert_array_almost_equal(rsquared, 0.2712646252915959)
    assert_array_almost_equal(slope_std_err, 0.4766175568062183)

    # TODO Test with multiple predictors (got an error previously)


@ignore_warnings(category=ConvergenceWarning)
def test_fit_statsmodels_univariate_models_binary_classification(
    xy_binary_classification,
):
    # Mainly regression tests to ensure it runs
    # Then evaluation should be tested elsewhere

    seed = 15
    np.random.seed(seed)

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]

    modeling_out = fit_statsmodels_univariate_models(
        x=x,
        y=y,
        task="classification",
        standardize_cols=True,
        add_intercept=True,
        rm_missing=False,
        df_out=True,
    )

    print("Finished cv")

    print(modeling_out)

    assert modeling_out.to_dict() == {
        "Intercept": {
            0: 0.003407838828131782,
            1: 0.2638078477906434,
            2: -0.03623329305292816,
            3: -0.08862073021084887,
            4: 0.00019933251350379359,
        },
        "Slope": {
            0: 1.5991262156820405,
            1: 3.1059862649173047,
            2: 4.159415275879349,
            3: 2.509606675470246,
            4: 2.1742711270622586,
        },
        "P(Intercept)": {
            0: 0.9921220867178183,
            1: 0.5474593804006052,
            2: 0.9427977040090961,
            3: 0.8187347447903474,
            4: 0.9995911868318506,
        },
        "P(Slope)": {
            0: 0.0007932026427665445,
            1: 0.0009874901297360645,
            2: 0.0012458443806012325,
            3: 0.0009075541437097006,
            4: 0.00017047061878177998,
        },
        "R2": {
            0: 0.2712646252915959,
            1: 0.49610649295883624,
            2: 0.6313631137719539,
            3: 0.40155440711987467,
            4: 0.40103510322189406,
        },
        "SE(Slope)": {
            0: 0.4766175568062183,
            1: 0.9429032299147799,
            2: 1.2884742362545987,
            3: 0.7564250989491418,
            4: 0.5783896130439601,
        },
        "Model Function": {
            0: "statsmodels.Logit",
            1: "statsmodels.Logit",
            2: "statsmodels.Logit",
            3: "statsmodels.Logit",
            4: "statsmodels.Logit",
        },
    }

    params, pvalues, rsquareds, slope_std_errs = fit_statsmodels_univariate_models(
        x=x,
        y=y,
        task="classification",
        standardize_cols=True,
        add_intercept=True,
        rm_missing=False,
        df_out=False,
    )

    print("Finished cv")

    print(
        params.tolist(), pvalues.tolist(), rsquareds.tolist(), slope_std_errs.tolist()
    )
    assert_array_almost_equal(
        params,
        [
            [0.003407838828131782, 1.5991262156820405],
            [0.2638078477906434, 3.1059862649173047],
            [-0.03623329305292816, 4.159415275879349],
            [-0.08862073021084887, 2.509606675470246],
            [0.00019933251350379359, 2.1742711270622586],
        ],
    )
    assert_array_almost_equal(
        pvalues,
        [
            [0.9921220867178183, 0.0007932026427665445],
            [0.5474593804006052, 0.0009874901297360645],
            [0.9427977040090961, 0.0012458443806012325],
            [0.8187347447903474, 0.0009075541437097006],
            [0.9995911868318506, 0.00017047061878177998],
        ],
    )
    assert_array_almost_equal(
        rsquareds,
        [
            0.2712646252915959,
            0.49610649295883624,
            0.6313631137719539,
            0.40155440711987467,
            0.40103510322189406,
        ],
    )
    assert_array_almost_equal(
        slope_std_errs,
        [
            0.4766175568062183,
            0.9429032299147799,
            1.2884742362545987,
            0.7564250989491418,
            0.5783896130439601,
        ],
    )


# TODO: Test with regression
