import random
import numpy as np

from generalize.evaluate.roc_curves import ROCCurves, ROCCurve


def test_roc_curves_set_get():

    seed = 15
    np.random.seed(seed)
    random.seed(seed)

    roc_coll = ROCCurves()

    for rep in range(3):
        for split in range(2):
            for thresh in ["a", "b"]:
                roc_coll.add(
                    path=f"rep.{rep}.split.{split}.threshold.{thresh}",
                    roc_curve=ROCCurve(
                        fpr=np.random.uniform(size=(10)),
                        tpr=np.random.uniform(size=(10)),
                        thresholds=np.linspace(num=10, start=0, stop=1),
                        auc=random.random(),
                    ),
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
                        "AUC": 0.9652, "FPR (10)": [0.8488, 0.1789, 0.0544, "..."], "TPR (10)": [0.9176, 0.2641, 0.7178, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.0117, "FPR (10)": [0.9985, 0.3728, 0.7605, "..."], "TPR (10)": [0.299, 0.5377, 0.6656, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }
                    }
                    }, "1": {
                    "threshold": {
                        "a": {
                        "AUC": 0.736, "FPR (10)": [0.0797, 0.0568, 0.0783, "..."], "TPR (10)": [0.9746, 0.3298, 0.1954, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.158, "FPR (10)": [0.3713, 0.0977, 0.7275, "..."], "TPR (10)": [0.8879, 0.2204, 0.1068, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }
                    }
                    }
                }
                }, "1": {
                "split": {
                    "0": {
                    "threshold": {
                        "a": {
                        "AUC": 0.9863, "FPR (10)": [0.1276, 0.3037, 0.2953, "..."], "TPR (10)": [0.0016, 0.0717, 0.4205, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.0169, "FPR (10)": [0.015, 0.8997, 0.6154, "..."], "TPR (10)": [0.009, 0.7748, 0.1214, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }
                    }
                    }, "1": {
                    "threshold": {
                        "a": {
                        "AUC": 0.8795, "FPR (10)": [0.4013, 0.59, 0.9939, "..."], "TPR (10)": [0.37, 0.2282, 0.8599, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.6814, "FPR (10)": [0.1154, 0.1595, 0.7454, "..."], "TPR (10)": [0.7892, 0.8388, 0.7613, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }
                    }
                    }
                }
                }, "2": {
                "split": {
                    "0": {
                    "threshold": {
                        "a": {
                        "AUC": 0.8573, "FPR (10)": [0.7974, 0.9691, 0.9685, "..."], "TPR (10)": [0.2174, 0.1016, 0.249, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.9998, "FPR (10)": [0.0309, 0.5909, 0.7483, "..."], "TPR (10)": [0.1679, 0.4104, 0.4842, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }
                    }
                    }, "1": {
                    "threshold": {
                        "a": {
                        "AUC": 0.2397, "FPR (10)": [0.7692, 0.5408, 0.2683, "..."], "TPR (10)": [0.4881, 0.4945, 0.2094, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
                        }, "b": {
                        "AUC": 0.3381, "FPR (10)": [0.2645, 0.417, 0.0261, "..."], "TPR (10)": [0.1444, 0.941, 0.0159, "..."], "Thresholds (10)": [0.0, 0.1111, 0.2222, "..."]
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
            roc_coll.add(
                path=f"rep.{rep}.split.{split}",
                roc_curve=ROCCurve(
                    fpr=np.random.uniform(size=(10)),
                    tpr=np.random.uniform(size=(10)),
                    thresholds=np.linspace(num=10, start=0, stop=1),
                    auc=random.random(),
                ),
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
