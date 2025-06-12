import random
import numpy as np
import pandas as pd

from generalize.evaluate.roc_curves import ROCCurves, ROCCurve


def test_roc_curves_set_get():
    seed = 15
    np.random.seed(seed)
    random.seed(seed)

    roc_coll = ROCCurves()

    for rep in range(3):
        for split in range(2):
            for thresh in ["a", "b"]:
                fpr = np.sort(np.random.uniform(size=(10)))
                tpr = np.sort(np.random.uniform(size=(10)))
                roc_coll.add(
                    path=f"rep.{rep}.split.{split}.threshold.{thresh}",
                    roc_curve=ROCCurve(
                        fpr=fpr,
                        tpr=tpr,
                        thresholds=np.linspace(num=10, start=1, stop=0),
                        auc=None,
                    ).recalculate_auc(),
                )

    print(roc_coll._curves)

    roc_coll_dicts = roc_coll.to_dicts()
    print(roc_coll_dicts)

    expected_stdout = """
    ROC Curve Collection:
    ROC Curves:
        {
        "rep": {
            "0": {
            "split": {
                "0": {
                "threshold": {
                    "a": {
                    "AUC": 0.5031, "FPR (10)": [0.0544, 0.1117, 0.1789, "..."], "TPR (10)": [0.0394, 0.0467, 0.1672, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.4473, "FPR (10)": [0.1094, 0.1416, 0.3728, "..."], "TPR (10)": [0.299, 0.3643, 0.3904, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }, "1": {
                "threshold": {
                    "a": {
                    "AUC": 0.4088, "FPR (10)": [0.0292, 0.0308, 0.0568, "..."], "TPR (10)": [0.0239, 0.1954, 0.2688, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.3316, "FPR (10)": [0.0977, 0.3713, 0.4253, "..."], "TPR (10)": [0.012, 0.1068, 0.2198, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }
            }
            }, "1": {
            "split": {
                "0": {
                "threshold": {
                    "a": {
                    "AUC": 0.4101, "FPR (10)": [0.1276, 0.2014, 0.2953, "..."], "TPR (10)": [0.0016, 0.0717, 0.2007, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.3238, "FPR (10)": [0.015, 0.1393, 0.5463, "..."], "TPR (10)": [0.009, 0.1214, 0.124, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }, "1": {
                "threshold": {
                    "a": {
                    "AUC": 0.4914, "FPR (10)": [0.0962, 0.1292, 0.2154, "..."], "TPR (10)": [0.2282, 0.365, 0.37, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.4006, "FPR (10)": [0.066, 0.1154, 0.1595, "..."], "TPR (10)": [0.0361, 0.0993, 0.1215, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }
            }
            }, "2": {
            "split": {
                "0": {
                "threshold": {
                    "a": {
                    "AUC": 0.4235, "FPR (10)": [0.079, 0.167, 0.1847, "..."], "TPR (10)": [0.0475, 0.1016, 0.2174, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.4193, "FPR (10)": [0.0309, 0.1332, 0.1984, "..."], "TPR (10)": [0.1461, 0.1585, 0.1679, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }, "1": {
                "threshold": {
                    "a": {
                    "AUC": 0.3534, "FPR (10)": [0.0073, 0.2092, 0.2683, "..."], "TPR (10)": [0.2052, 0.2094, 0.2622, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }, "b": {
                    "AUC": 0.4259, "FPR (10)": [0.02, 0.0261, 0.2253, "..."], "TPR (10)": [0.0159, 0.1444, 0.4084, "..."], "Thresholds (10)": [1.0, 0.8889, 0.7778, "..."]
                    }
                }
                }
            }
            }
        }
    }
    """
    # Without any whitespace

    def remove_whitespace(s):
        return "".join(s.split())

    assert remove_whitespace(expected_stdout) == remove_whitespace(
        str(roc_coll_dicts)
    ), "The printing of ROC collection changed."

    # print(conf_coll.paths)
    assert roc_coll.paths == [
        "rep.0.split.0.threshold.a",
        "rep.0.split.0.threshold.b",
        "rep.0.split.1.threshold.a",
        "rep.0.split.1.threshold.b",
        "rep.1.split.0.threshold.a",
        "rep.1.split.0.threshold.b",
        "rep.1.split.1.threshold.a",
        "rep.1.split.1.threshold.b",
        "rep.2.split.0.threshold.a",
        "rep.2.split.0.threshold.b",
        "rep.2.split.1.threshold.a",
        "rep.2.split.1.threshold.b",
    ]


def test_roc_curves_save_load(tmp_path):
    seed = 15
    np.random.seed(seed)
    random.seed(seed)

    roc_coll = ROCCurves()

    for rep in range(3):
        for split in range(2):
            fpr = np.sort(np.random.uniform(size=(10)))
            tpr = np.sort(np.random.uniform(size=(10)))
            roc_coll.add(
                path=f"rep.{rep}.split.{split}",
                roc_curve=ROCCurve(
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=np.linspace(num=10, start=1, stop=0),
                    auc=None,
                ).recalculate_auc(),
            )

    print("Before saving")
    print(roc_coll)

    roc_coll.save(file_path=tmp_path / "tmp_roc_curves.json")
    loaded_roc_coll = roc_coll.load(file_path=tmp_path / "tmp_roc_curves.json")

    print("Loaded")
    print(loaded_roc_coll)

    # Without any whitespace
    def remove_whitespace(s):
        return "".join(s.split())

    assert remove_whitespace(str(roc_coll)) == remove_whitespace(
        str(loaded_roc_coll)
    ), "The printing of ROC collection changed after save+load."

    assert roc_coll.paths == loaded_roc_coll.paths


def test_roc_curve_ops():
    seed = 15
    np.random.seed(seed)
    random.seed(seed)

    roc_coll = ROCCurves()

    for rep in range(3):
        for split in range(2):
            fpr = np.sort(np.random.uniform(size=(10)))
            tpr = np.sort(np.random.uniform(size=(10)))
            roc_coll.add(
                path=f"rep.{rep}.split.{split}",
                roc_curve=ROCCurve(
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=np.linspace(num=10, start=1, stop=0),
                    auc=None,
                ).recalculate_auc(),
            )

    # Interpolation of ROC curve
    # relative to thresholds
    roc_interpolated = roc_coll.get(f"rep.{0}.split.{0}").interpolate(
        to=40, reference="thresholds"
    )
    print(roc_interpolated.fpr)
    print(roc_interpolated.tpr)
    print(roc_interpolated.thresholds)
    print(roc_interpolated.auc)

    # fmt: off
    np.testing.assert_array_almost_equal(roc_interpolated.fpr, np.asarray([0.0544, 0.0676, 0.0808, 0.0941, 0.1073, 0.1221, 0.1376, 0.1531, 0.1686, 0.1844, 0.2007, 0.2171, 0.2335, 0.2499, 0.2558, 0.2617, 0.2676, 0.2734, 0.2799, 0.2866, 0.2933, 0.3000, 0.3046, 0.3049, 0.3053, 0.3056, 0.3059, 0.3188, 0.3316, 0.3444, 0.3573, 0.3875, 0.4263, 0.4652, 0.5041, 0.5545, 0.6281, 0.7017, 0.7752, 0.8488]), decimal=3)
    np.testing.assert_array_almost_equal(roc_interpolated.tpr, np.asarray([0.0394, 0.0411, 0.0428, 0.0445, 0.0461, 0.0653, 0.0931, 0.1209, 0.1487, 0.1698, 0.1774, 0.1850, 0.1926, 0.2002, 0.2026, 0.2050, 0.2074, 0.2098, 0.2188, 0.2312, 0.2435, 0.2559, 0.2990, 0.4037, 0.5084, 0.6131, 0.7178, 0.7384, 0.7590, 0.7796, 0.8002, 0.8161, 0.8296, 0.8432, 0.8567, 0.8697, 0.8817, 0.8937, 0.9056, 0.9176]), decimal=3)
    np.testing.assert_array_almost_equal(roc_interpolated.thresholds, np.asarray([1.0000, 0.9744, 0.9487, 0.9231, 0.8974, 0.8718, 0.8462, 0.8205, 0.7949, 0.7692, 0.7436, 0.7179, 0.6923, 0.6667, 0.6410, 0.6154, 0.5897, 0.5641, 0.5385, 0.5128, 0.4872, 0.4615, 0.4359, 0.4103, 0.3846, 0.3590, 0.3333, 0.3077, 0.2821, 0.2564, 0.2308, 0.2051, 0.1795, 0.1538, 0.1282, 0.1026, 0.0769, 0.0513, 0.0256, 0.0000]), decimal=3)
    np.testing.assert_almost_equal(roc_interpolated.auc, 0.50309, decimal=3)
    # fmt: on

    # Difference curve
    # rocs_diffs = roc_coll.get(f"rep.{0}.split.{0}") - roc_coll.get(f"rep.{0}.split.{1}")
    # assert len(rocs_diffs.fpr_diff) == 1001
    # assert len(rocs_diffs.tpr_diff) == 1001
    # assert len(rocs_diffs.thresholds) == 1001

    # print("ROC Differences:")
    # print(rocs_diffs.fpr_diff[:10])
    # print(rocs_diffs.tpr_diff[:10])
    # print(rocs_diffs.thresholds[:10])
    # print(rocs_diffs.auc_diff)

    # # fmt: off
    # np.testing.assert_array_almost_equal(rocs_diffs.fpr_diff[:10], [
    #     -0.05508343, -0.05485601, -0.05462858, -0.05440116, -0.05417373,
    #     -0.05394631, -0.05371888, -0.05349146, -0.05326403, -0.05303661],
    #     decimal=3
    # )
    # np.testing.assert_array_almost_equal(rocs_diffs.tpr_diff[:10], [
    #     -0.25957634, -0.26009876, -0.26062117, -0.26114359, -0.261666, -0.26218842,
    #     -0.26271083, -0.26323325, -0.26375566, -0.26427808],
    #     decimal=3
    # )
    # np.testing.assert_array_almost_equal(rocs_diffs.thresholds[:10], [
    #     1., 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992, 0.991
    #     ],
    #     decimal=3
    # )
    # np.testing.assert_almost_equal(rocs_diffs.auc_diff,  0.0557986, decimal=3)
    # fmt: on


def test_roc_curve_interpolated_metrics():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    preds = np.array([0.3, 0.4, 0.7, 0.25, 0.8, 0.4, 0.7, 0.4])

    roc = ROCCurve.from_data(targets=labels, predicted_probabilities=preds)
    print(roc)

    sens_thresh_interp = roc.get_threshold_at_sensitivity(
        above_sensitivity=0.99999, interpolate=True
    )
    spec_thresh_interp = roc.get_threshold_at_specificity(
        above_specificity=0.99999, interpolate=True
    )

    print("Interpolation")
    print(sens_thresh_interp)
    print(spec_thresh_interp)

    # Closest is 1.0 (fails if we say 1.0?)
    sens_thresh_closest = roc.get_threshold_at_sensitivity(
        above_sensitivity=0.99999, interpolate=False
    )
    spec_thresh_closest = roc.get_threshold_at_specificity(
        above_specificity=0.99999, interpolate=False
    )

    print("No Interpolation")
    print(sens_thresh_closest)
    print(spec_thresh_closest)

    assert np.round(sens_thresh_interp["Specificity"], decimals=3) == np.round(sens_thresh_closest["Specificity"], decimals=3)
    assert np.round(sens_thresh_interp["Sensitivity"], decimals=3) == np.round(sens_thresh_closest["Sensitivity"], decimals=3)

    assert np.round(spec_thresh_interp["Specificity"], decimals=3) == np.round(spec_thresh_closest["Specificity"], decimals=3)
    assert np.round(spec_thresh_interp["Sensitivity"], decimals=3) == np.round(spec_thresh_closest["Sensitivity"], decimals=3)


def test_roc_curve_interpolation():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    preds = np.array([0.3, 0.4, 0.7, 0.25, 0.8, 0.4, 0.7, 0.4])

    roc = ROCCurve.from_data(
        targets=labels, predicted_probabilities=preds
    ).recalculate_auc()

    np.testing.assert_almost_equal(roc.auc, 0.78125, decimal=3)

    all_roc_dfs = []
    all_rankings = []
    references = ["fpr", "tpr", "thresholds"]
    for interp_n in [5, 7, 11, 15, 19, 27, 37, 51]:
        roc_dfs = []
        for reference in references:
            roc_interpolated = roc.interpolate(to=interp_n, reference=reference)
            interp_df = pd.DataFrame(
                {
                    "Reference": reference,
                    "Grid Size": interp_n,
                    "FPR": roc_interpolated.fpr,
                    "TPR": roc_interpolated.tpr,
                    "Thresholds": roc_interpolated.thresholds,
                    "AUC": roc_interpolated.auc,
                    "Label": f"{reference} (area={roc_interpolated.auc})",
                }
            )
            roc_dfs.append(interp_df)
            all_roc_dfs.append(interp_df)

        # Check lengths of data frames are correct
        np.testing.assert_array_almost_equal(
            [df.shape[0] for df in roc_dfs],
            [interp_n for _ in references],
        )

        # Check how close the interpolated ROC AUCs are to the original AUC
        # Order: "fpr", "tpr", "thresholds" (so fpr is closest)
        abs_auc_diffs = [np.abs(roc.auc - df["AUC"].tolist()[0]) for df in roc_dfs]
        all_rankings.append(np.argsort(abs_auc_diffs))

    # print(all_rankings)
    np.testing.assert_array_almost_equal(
        np.mean(all_rankings, axis=0), [0.5, 1.5, 1.0], decimal=5
    )

    # Enable to check visually how well
    # different interpolation references
    # reproduce the original curve and AUC
    if False:
        import matplotlib.pyplot as plt

        roc_df = pd.concat(all_roc_dfs)
        roc_df = roc_df.loc[roc_df["Grid Size"].isin([5, 11, 19, 37])]

        facet_col = "Grid Size"
        facet_groups = roc_df[facet_col].unique()
        num_facet_groups = len(facet_groups)

        fig, axes = plt.subplots(
            num_facet_groups, 1, figsize=(10, 3.5 * num_facet_groups)
        )
        if num_facet_groups == 1:
            axes = [axes]

        for ax, (facet_group, data) in zip(axes, roc_df.groupby(facet_col)):
            line_groups = data.groupby("Label")

            for name, group in line_groups:
                ax.plot(
                    group["FPR"], group["TPR"], marker="o", linestyle="-", label=name
                )
            ax.plot(roc.fpr, roc.tpr, label=f"Original (area = {roc.auc})")
            ax.plot([0, 1], [0, 1], "k--", label="No Skill")

            # Step 5: Customize the plot
            ax.set_title(f"{facet_group}")
            ax.set_xlabel("")
            ax.set_ylabel("TPR")
            ax.legend(title="Reference")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

        plt.show()

    # assert False
