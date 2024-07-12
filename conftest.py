import pytest
from functools import partial
import pathlib
import random
from typing import Callable, Optional
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax
from utipy import random_alphanumeric, mk_dir


@pytest.fixture
def create_splits_fn() -> Callable:
    def create_split(
        num_samples: int, num_splits: int, id_names: Optional[str] = None
    ) -> np.ndarray:
        per_split = int(np.ceil(num_samples / num_splits))
        split_id_groups = [[i] * per_split for i in range(num_splits)]
        num_excess = (per_split * num_splits) - num_samples
        rm_from = np.random.choice(range(num_splits), size=num_excess)
        split_id_groups = np.concatenate(
            [
                ids[:-1] if idx in rm_from else ids
                for idx, ids in enumerate(split_id_groups)
            ],
            axis=0,
        )
        split_ids = split_id_groups.flatten().astype(int)
        np.random.shuffle(split_ids)
        if id_names is not None:
            assert len(id_names) == num_splits
            split_ids = np.asarray([id_names[i] for i in split_ids], dtype=object)
        return split_ids

    return create_split


@pytest.fixture
def create_y_fn() -> Callable:
    def create_y(
        num_samples: int, digitize: bool = True, num_classes: int = 2
    ) -> np.ndarray:
        # Multiclass probabilities
        if not digitize and num_classes > 2:
            # Draw random distribution
            y = np.random.uniform(size=(num_samples, num_classes))
            y = softmax(y, axis=-1)
            return y
        # Draw random distribution
        y = np.random.uniform(size=(num_samples))
        # Bin y
        if digitize:
            y = np.digitize(
                y, bins=np.linspace(0, 1, num_classes + 1)[1:-1], right=False
            )
        return y

    return create_y


##            ##
###   Data   ###
##            ##


#              #
## Regression ##
#              #


@pytest.fixture
def xy_regression():
    seed = 15
    np.random.seed(seed)

    num_samples = 50
    num_features = 5

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bias x by the targets
    x += np.expand_dims(y, 1) * 0.5

    return {"x": x, "y": y, "num_samples": num_samples, "num_features": num_features}


def test_xy_regression(xy_regression):

    x = xy_regression["x"]
    y = xy_regression["y"]
    num_samples = xy_regression["num_samples"]
    num_features = xy_regression["num_features"]

    print(y)

    # Ensure y didn't change
    np.testing.assert_array_almost_equal(
        y[:18],
        np.asarray(
            [
                0.92547864,
                0.26051745,
                0.08258404,
                0.05352811,
                0.80512636,
                0.31172871,
                0.65006592,
                0.17141982,
                0.80011088,
                0.40250834,
                0.06040717,
                0.60243982,
                0.17427871,
                0.42956401,
                0.418464,
                0.92826276,
                0.92836599,
                0.31155852,
            ]
        ),
    )

    print(x)

    # Test a small sample of x values
    # to ensure it didn't change
    np.testing.assert_almost_equal(
        x[4:6, 2:4], np.array([[1.16307, 0.87604], [0.62096, 0.29742]]), decimal=5
    )

    assert num_samples == 50
    assert num_features == 5
    assert x.shape == (50, 5)
    assert y.shape == (50,)


@pytest.fixture
def xy_regression_multiple_feature_sets():
    seed = 15
    np.random.seed(seed)

    num_samples = 50
    num_features = 5
    num_feature_sets = 3

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_feature_sets, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bias x by the targets
    x += np.reshape(y, (num_samples, 1, 1)) * 0.5

    return {
        "x": x,
        "y": y,
        "num_samples": num_samples,
        "num_features": num_features,
        "num_feature_sets": num_feature_sets,
    }


def test_xy_regression_multiple_feature_sets(xy_regression_multiple_feature_sets):

    x = xy_regression_multiple_feature_sets["x"]
    y = xy_regression_multiple_feature_sets["y"]
    num_samples = xy_regression_multiple_feature_sets["num_samples"]
    num_features = xy_regression_multiple_feature_sets["num_features"]
    num_feature_sets = xy_regression_multiple_feature_sets["num_feature_sets"]

    print(y)

    # Ensure y didn't change
    np.testing.assert_array_almost_equal(
        y[:18],
        np.asarray(
            [
                0.531599,
                0.514484,
                0.16243,
                0.501825,
                0.236588,
                0.240513,
                0.874645,
                0.396031,
                0.126165,
                0.923501,
                0.440676,
                0.552849,
                0.94114,
                0.301705,
                0.157883,
                0.16037,
                0.237279,
                0.976972,
            ]
        ),
    )

    print(x)

    # Test a small sample of x values
    # to ensure it didn't change
    np.testing.assert_almost_equal(
        x[4:6, 0, 2:4], np.array([[0.84582, 0.57774], [0.66781, 1.04106]]), decimal=5
    )

    # Test feature sets are different
    # (these shouldn't be 0s)
    np.testing.assert_almost_equal(
        x[4:6, 0, 2:4] - x[4:6, 1, 2:4],
        np.array([[0.12405, -0.3644], [0.25226, 0.35169]]),
        decimal=5,
    )

    assert num_samples == 50
    assert num_features == 5
    assert num_feature_sets == 3
    assert x.shape == (50, 3, 5)
    assert y.shape == (50,)


#                         #
## Binary classification ##
#                         #


@pytest.fixture
def xy_binary_classification():
    seed = 15
    np.random.seed(seed)

    num_samples = 50
    num_features = 5

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bin y
    y = np.digitize(y, bins=[0.5], right=False)

    # Bias x by the targets
    x += np.expand_dims(y, 1) * 0.5

    return {"x": x, "y": y, "num_samples": num_samples, "num_features": num_features}


def test_xy_binary_classification(xy_binary_classification):

    x = xy_binary_classification["x"]
    y = xy_binary_classification["y"]
    num_samples = xy_binary_classification["num_samples"]
    num_features = xy_binary_classification["num_features"]

    print(y)

    # Ensure y didn't change
    np.testing.assert_array_equal(
        y,
        np.asarray(
            [
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                1,
            ]
        ),
    )

    print(x)

    # Test a small sample of x values
    # to ensure it didn't change
    np.testing.assert_almost_equal(
        x[4:6, 2:4], np.array([[1.26051, 0.97347], [0.46509, 0.141555260]]), decimal=5
    )

    assert num_samples == 50
    assert num_features == 5
    assert x.shape == (50, 5)
    assert y.shape == (50,)


@pytest.fixture
def xy_binary_classification_multiple_feature_sets():
    seed = 15
    random.seed(1)
    np.random.seed(seed)

    num_samples = 50
    num_features = 5
    num_feature_sets = 3

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_feature_sets, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bin y
    y = np.digitize(y, bins=[0.5], right=False)

    # Bias x by the targets
    x += np.reshape(y, (num_samples, 1, 1)) * 0.5

    return {
        "x": x,
        "y": y,
        "num_samples": num_samples,
        "num_features": num_features,
        "num_feature_sets": num_feature_sets,
    }


def test_xy_binary_classification_multiple_feature_sets(
    xy_binary_classification_multiple_feature_sets,
):

    x = xy_binary_classification_multiple_feature_sets["x"]
    y = xy_binary_classification_multiple_feature_sets["y"]
    num_samples = xy_binary_classification_multiple_feature_sets["num_samples"]
    num_feature_sets = xy_binary_classification_multiple_feature_sets[
        "num_feature_sets"
    ]
    num_features = xy_binary_classification_multiple_feature_sets["num_features"]

    print(y)

    # Ensure y didn't change
    np.testing.assert_array_equal(
        y,
        np.asarray(
            [
                1,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
            ]
        ),
    )

    print(x)

    # Test a small sample of x values
    # to ensure it didn't change
    np.testing.assert_almost_equal(
        x[4:6, 0, 2:4], np.array([[0.72752, 0.45944], [0.54755, 0.9208]]), decimal=5
    )

    # Test feature sets are different
    # (these shouldn't be 0s)
    np.testing.assert_almost_equal(
        x[4:6, 0, 2:4] - x[4:6, 1, 2:4],
        np.array([[0.12405, -0.3644], [0.25226, 0.35169]]),
        decimal=5,
    )

    assert num_samples == 50
    assert num_features == 5
    assert num_feature_sets == 3
    assert x.shape == (50, 3, 5)
    assert y.shape == (50,)


@pytest.fixture
def xy_binary_classification_xl():
    """
    Extra large version with 150 samples
    """
    seed = 15
    np.random.seed(seed)

    num_samples = 150
    num_features = 5

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bin y
    y = np.digitize(y, bins=[0.5], right=False)

    # Bias x by the targets
    x += np.expand_dims(y, 1) * 0.5

    return {"x": x, "y": y, "num_samples": num_samples, "num_features": num_features}


#                             #
## Multiclass classification ##
#                             #


@pytest.fixture
def xy_mc_classification():
    seed = 15
    np.random.seed(seed)

    num_samples = 50
    num_features = 5

    # Random distributions
    x = np.random.uniform(size=(num_samples, num_features))
    y = np.random.uniform(size=(num_samples))

    # Bin y
    y = np.digitize(y, bins=[0.33, 0.66], right=False)

    # Bias x by the targets
    x += np.expand_dims(y, 1) * 0.5

    return {"x": x, "y": y, "num_samples": num_samples, "num_features": num_features}


def test_xy_mc_classification(xy_mc_classification):

    x = xy_mc_classification["x"]
    y = xy_mc_classification["y"]
    num_samples = xy_mc_classification["num_samples"]
    num_features = xy_mc_classification["num_features"]

    print(y)

    # Ensure y didn't change
    np.testing.assert_array_almost_equal(
        y[:18], np.asarray([2, 0, 0, 0, 2, 0, 1, 0, 2, 1, 0, 1, 0, 1, 1, 2, 2, 0])
    )

    print(x)

    # Test a small sample of x values
    # to ensure it didn't change
    np.testing.assert_almost_equal(
        x[4:6, 2:4], np.array([[1.76051, 1.47347], [0.46509, 0.14156]]), decimal=5
    )

    assert num_samples == 50
    assert num_features == 5
    assert x.shape == (50, 5)
    assert y.shape == (50,)


##            ##
###  Models  ###
##            ##


@pytest.fixture
def LogisticRegressionClassifierPartial():
    # Requires setting random_state only
    return {
        "model": partial(
            LogisticRegression,
            **{
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": "balanced",
            }
        ),
        "grid_binary": {"model__C": np.geomspace(10 ** (-3), 10**5, num=10)},
        "grid_mc": {"model__C": np.geomspace(10 ** (-4), 10**7, num=13)},
    }


@pytest.fixture
def UnbalancedLogisticRegressionClassifierPartial():
    # Version without class-balancing of loss
    # Requires setting random_state only
    return {
        "model": partial(
            LogisticRegression,
            **{
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": None,
            }
        ),
        "grid_binary": {"model__C": np.geomspace(10 ** (-3), 10**5, num=10)},
        "grid_mc": {"model__C": np.geomspace(10 ** (-4), 10**7, num=13)},
    }


@pytest.fixture
def SavingLogisticRegressionClassifierPartial():

    class SavingLogisticRegression(LogisticRegression):
        def __init__(
            self,
            penalty,
            tol,
            C,
            class_weight,
            save_path,
            random_state=None,
            solver="saga",
            max_iter=100,
            multi_class="auto",
            verbose=0,
            n_jobs=None,
            l1_ratio=None,
        ) -> None:
            super().__init__(
                penalty=penalty,
                tol=tol,
                C=C,
                class_weight=class_weight,
                random_state=random_state,
                solver=solver,
                max_iter=max_iter,
                multi_class=multi_class,
                verbose=verbose,
                n_jobs=n_jobs,
                l1_ratio=l1_ratio,
            )
            self.save_path = save_path

        def fit(self, X, y, sample_weight=None):
            self.out_dir_ = pathlib.Path(self.save_path) / random_alphanumeric(size=10)
            mk_dir(self.out_dir_ / "fit", raise_on_exists=False, messenger=None)
            np.save(self.out_dir_ / "fit" / "X.npy", X, allow_pickle=True)
            np.save(self.out_dir_ / "fit" / "y.npy", y, allow_pickle=True)
            np.save(
                self.out_dir_ / "fit" / "sample_weight.npy",
                sample_weight,
                allow_pickle=True,
            )
            return super().fit(X=X, y=y, sample_weight=sample_weight)

        def predict(self, X):
            mk_dir(self.out_dir_ / "predict", raise_on_exists=False, messenger=None)
            np.save(self.out_dir_ / "predict" / "X.npy", X, allow_pickle=True)
            return super().predict(X=X)

        def predict_proba(self, X):
            mk_dir(
                self.out_dir_ / "predict_proba", raise_on_exists=False, messenger=None
            )
            np.save(self.out_dir_ / "predict_proba" / "X.npy", X, allow_pickle=True)
            return super().predict_proba(X=X)

        def predict_log_proba(self, X):
            mk_dir(
                self.out_dir_ / "predict_log_proba",
                raise_on_exists=False,
                messenger=None,
            )
            np.save(self.out_dir_ / "predict_log_proba" / "X.npy", X, allow_pickle=True)
            return super().predict_log_proba(X=X)

    # Requires setting random_state only
    return {
        "model": partial(
            SavingLogisticRegression,
            **{
                "C": 0.01,
                "penalty": "l1",
                "solver": "saga",
                "tol": 0.0001,
                "max_iter": 1000,
                "class_weight": "balanced",
            }
        ),
        "grid_binary": {"model__C": np.geomspace(10 ** (-3), 10**5, num=1)},
        "grid_mc": {"model__C": np.geomspace(10 ** (-4), 10**7, num=1)},
    }


@pytest.fixture
def SVMClassifierPartial():
    # Requires setting random_state only
    return {
        "model": partial(
            SVC,
            **{
                "max_iter": 1000,
                "tol": 0.0001,
                "class_weight": "balanced",
                # "probability": True # Want to test it works with discrete class predictions
            }
        ),
        "grid": {
            "model__kernel": ["linear", "rbf"],
            "model__C": np.geomspace(10 ** (-4), 10**7, num=12),
        },
    }


@pytest.fixture
def RandomForestClassifierPartial():
    # Requires setting random_state only
    return {
        "model": partial(
            RandomForestClassifier, **{"n_estimators": 50, "class_weight": "balanced"}
        ),
        "grid": {"model__n_estimators": [50, 100]},
    }


@pytest.fixture
def LassoLinearRegressionPartial():
    # Requires setting random_state only
    return {
        "model": partial(Lasso, **{"tol": 0.0001, "max_iter": 5000}),
        "grid": {"model__alpha": np.geomspace(10 ** (-4), 1.5, num=13)},
    }
