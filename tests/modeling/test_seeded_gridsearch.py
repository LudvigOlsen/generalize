from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from generalize.model.cross_validate.grid_search import (
    NestableGridSearchCV,
    make_simplest_model_refit_strategy,
)
from generalize.model.pipeline.pipeline_designer import PipelineDesigner


# Assuming your custom refit function is defined as `lowest_c_refit_strategy`
def test_custom_refit_strategy():
    # Create a simple dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Define the model and parameter space
    pipe_steps = (
        PipelineDesigner()
        .add_step(
            name="standardize",
            transformer="scale_features",
            add_dim_transformer_wrapper=True,
        )
        .build()
    )
    pipe_steps += [("model", LogisticRegression(solver="liblinear"))]
    pipeline = Pipeline(pipe_steps)
    param_grid = {"model__C": [0.1, 1, 10, 100]}

    # Setup GridSearchCV with the custom refit strategy
    grid_search = NestableGridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        seed=None,
        refit=make_simplest_model_refit_strategy(
            main_var="model__C",
            score_name="balanced_accuracy",
            verbose=True,
        ),
        scoring="balanced_accuracy",
        error_score="raise",
    )

    # Fit the GridSearchCV instance
    grid_search.fit(X, y)

    # After fitting, you can access best_estimator_ if everything went well
    print(grid_search.best_estimator_)

    # assert False


def test_standard_grid_search():
    # Create a simple dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Define the model and parameter space
    model = LogisticRegression(solver="liblinear")
    param_grid = {"C": [0.1, 1, 10, 100]}

    # Setup standard GridSearchCV with balanced_accuracy scoring
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        refit=make_simplest_model_refit_strategy(
            main_var="C",
            score_name="balanced_accuracy",
            verbose=True,
        ),
        scoring="balanced_accuracy",
    )

    # Fit the GridSearchCV instance
    grid_search.fit(X, y)

    # After fitting, attempt to access best_estimator_
    print(grid_search.best_estimator_)

    # assert False


# Assuming your custom refit function is defined as `lowest_c_refit_strategy`
def test_inner_results(tmp_path):
    # Create a simple dataset
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Define the model and parameter space
    pipe_steps = (
        PipelineDesigner()
        .add_step(
            name="standardize",
            transformer="scale_features",
            add_dim_transformer_wrapper=True,
        )
        .build()
    )
    pipe_steps += [("model", LogisticRegression(solver="liblinear"))]
    pipeline = Pipeline(pipe_steps)
    param_grid = {"model__C": [0.1, 1, 10, 100]}

    # Setup GridSearchCV with the custom refit strategy
    grid_search = NestableGridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        seed=None,
        refit=True,
        save_cv_results_path=tmp_path / "inner_cv_results.csv",
    )

    # Fit the GridSearchCV instance
    grid_search.fit(X, y)

    tmp_files = set([str(p) for p in tmp_path.glob("*")])
    exp_tmp_files = set(
        [
            str(p)
            for p in [
                tmp_path / "inner_cv_results.csv",
                tmp_path / "inner_cv_results.header.csv",
                tmp_path / "inner_cv_results.best_coefficients.csv",
            ]
        ]
    )

    assert tmp_files == exp_tmp_files

    # assert False
