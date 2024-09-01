import warnings
from typing import List, Optional, Tuple, Callable, Dict, Any
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from nattrs import nested_getattr
from utipy import Messenger

from generalize.model.pipeline.pipeline_designer import PipelineDesigner
from generalize.model.utils.weighting import calculate_sample_weight


class NestablePipeline(Pipeline):
    def __init__(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        is_seedable: bool,
        train_test_steps: List[str] = [],
        weight_loss_by_groups: bool = False,
        weight_loss_by_class: bool = False,
        weight_per_split: bool = False,
        weight_splits_equally: bool = False,
        split_weights: Optional[Dict[str, float]] = None,
        memory=None,
        verbose: bool = False,
        messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
    ) -> None:
        """
        Pipeline class with additional functionality.

        Sets the `is_seedable` attribute (skorch models) since
        assigning the attribute to a pipeline object does not seem
        to work with the cloning in the cross-validation.

        train_test_steps
            Names of transformers with the `training_mode`
            parameter for enabling different functionality
            during training and testing.
            The mode can be changed with `.enable_testing_mode()` and `.enable_training_mode()`.
            NOTE: Testing mode is automatically enabled after fitting.
        """
        self.is_seedable = is_seedable
        self.train_test_steps = train_test_steps
        self.weight_loss_by_groups = weight_loss_by_groups
        self.weight_loss_by_class = weight_loss_by_class
        self.weight_per_split = weight_per_split
        self.weight_splits_equally = weight_splits_equally
        self.split_weights = split_weights
        self.messenger = messenger
        super().__init__(steps=steps, memory=memory, verbose=verbose)

        if steps[-1][0] != "model":
            if weight_loss_by_groups:
                raise ValueError(
                    "NestablePipeline: `weight_loss_by_groups` can only be enabled "
                    "when the last step is named `'model'`."
                )
            if weight_loss_by_class:
                raise ValueError(
                    "NestablePipeline: `weight_loss_by_class` can only be enabled "
                    "when the last step is named `'model'`."
                )

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps`.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.

        Returns
        -------
        self : object
            Pipeline class instance.
        """
        # NOTE: Reason for overwriting `self.set_params()`
        # is that `dim_wrapper` needs to have its own
        # `.set_params()` called in order to
        # set parameters in sub estimators
        if "steps" in kwargs:
            setattr(self, "steps", kwargs.pop("steps"))

        for step_name, value in kwargs.items():
            step_parts = step_name.split("__")
            estimator = self.named_steps[step_parts[0]]
            for part_idx, part in enumerate(step_parts[1:-1]):
                full_part = "__".join(step_parts[: (part_idx + 2)])
                if hasattr(estimator, "named_steps"):
                    try:
                        estimator = estimator.named_steps[part]
                    except KeyError:
                        self.messenger(
                            f"Did not find `{full_part}`. `{part}` not in {estimator.named_steps.keys()}",
                            add_msg_fn=warnings.warn,
                        )
                        raise
                elif hasattr(estimator, "named_transformers"):
                    try:
                        estimator = estimator.named_transformers[part]
                    except KeyError:
                        self.messenger(
                            f"Did not find `{full_part}`. `{part}` not in {estimator.named_transformers.keys()}",
                            add_msg_fn=warnings.warn,
                        )
                        raise
                elif hasattr(estimator, "transformer_list"):  # Sklearn<1.2
                    try:
                        estimator = [
                            est
                            for est_name, est in estimator.transformer_list
                            if est_name == part
                        ][0]
                    except IndexError:
                        self.messenger(
                            f"Did not find `{full_part}`. `{part}` not in {dict(estimator.transformer_list).keys()}",
                            add_msg_fn=warnings.warn,
                        )
                        raise
                else:
                    self.messenger(f"{part}: {estimator}")
                    raise NotImplementedError(
                        f"estimator `{part}` has neither `named_steps` "
                        f"nor `named_transformers`: \n{dir(estimator)}"
                    )

            estimator.set_params(**{step_parts[-1]: value})
        return self

    def enable_testing_mode(self) -> None:
        """
        Enable testing mode in all `train_test_steps`.
        """
        self.set_params(
            **{
                step_name + "__training_mode": False
                for step_name in self.train_test_steps
            }
        )

    def enable_training_mode(self) -> None:
        """
        Enable training mode in all `train_test_steps`.
        """
        self.set_params(
            **{
                step_name + "__training_mode": True
                for step_name in self.train_test_steps
            }
        )

    def fit(self, X, y=None, groups=None, **fit_params):
        # When refitting, enable training mode
        # Only works when the transformers have been initialized
        try:
            self.enable_training_mode()
        except NotFittedError:
            pass

        # Calculate sample weights based on groups and classes
        sample_weights, groups = calculate_sample_weight(
            y=y,
            groups=groups,
            weight_loss_by_groups=self.weight_loss_by_groups,
            weight_loss_by_class=self.weight_loss_by_class,
            weight_per_split=self.weight_per_split,
            weight_splits_equally=self.weight_splits_equally,
            split_weights=self.split_weights,
        )

        # Either multiply with passed sample weights
        # Or use group-based sample weights
        if "model" in self.named_steps:
            if "model__sample_weight" in fit_params:
                fit_params["model__sample_weight"] = (
                    np.asarray(fit_params["model__sample_weight"]) * sample_weights
                )

                # Normalize to sum-to-length
                fit_params["model__sample_weight"] = (
                    fit_params["model__sample_weight"]
                    / fit_params["model__sample_weight"].sum()
                ) * len(fit_params["model__sample_weight"])
            else:
                fit_params["model__sample_weight"] = sample_weights

        with warnings.catch_warnings(record=True) as self.warnings:
            super().fit(X=X, y=y, **fit_params)
        for warn in self.warnings:
            warnings.warn_explicit(
                message=warn.message,
                category=warn.category,
                filename=warn.filename,
                lineno=warn.lineno,
                source=warn.source,
            )
            self.messenger(
                f"{warn.category} {warn.message} in "
                f"{warn.filename} at line {warn.lineno}"
            )

        self.enable_testing_mode()

        return self

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        # Ensures fit() is called (instead of `._fit()`)
        self.fit(X=X, y=y, groups=groups, **fit_params)
        return self.transform(X)

    def fit_predict(self, X, y=None, groups=None, **fit_params):
        # Ensures fit() is called (instead of `._fit()`)
        self.fit(X=X, y=y, groups=groups, **fit_params)
        return self.predict(X)

    def __str__(self) -> str:
        """
        Custom string representation that does not use truncation.
        """
        return pipeline_to_str(self)


class AttributeToDataFrameExtractor:
    def __init__(
        self,
        name: str,
        attr_path: str,
        to_df_fn: Optional[Callable] = None,
        pad_to: Optional[int] = None,
    ) -> None:
        """
        Class to extract an attribute from a pipeline and
        convert it to a pandas data frame.

        When the conversion to a data frame cannot be done with `pandas.DataFrame()`
        please provide the `to_df_fn` function which takes in the
        attribute and should return a pandas data frame.

        name
            Name of the attribute as you want it in filepaths.
            Should be suitable for filenames.
        pad_to
            Whether to pad the resulting data frame column-wise with NaNs
            to get `pad_to` columns. This is for when different pipeline
            instances can get a different number of columns for this attribute.
        """
        self.name = name.replace(" ", "_")
        self.attr_path = attr_path
        self.to_df_fn = to_df_fn
        self.pad_to = pad_to

    def __call__(self, pipe: Pipeline) -> pd.DataFrame:
        attr = self._extract_attribute(pipe)
        attr_df = self._convert_to_df(attr)
        if self.pad_to is not None:
            attr_df = AttributeToDataFrameExtractor.pad(
                attr_df,
                total_columns=self.pad_to,
            )
        return attr_df

    def _extract_attribute(self, pipe: Pipeline) -> Any:
        estimator_name = self.attr_path.split(".")[0]
        attr_path = ".".join(self.attr_path.split(".")[1:])
        try:
            attr = nested_getattr(
                pipe.named_steps[estimator_name],
                attr=attr_path,
                default="not found",
            )
        except Exception as e:  # noqa: E722
            raise ValueError(f"Failed to extract {self.name} from the pipeline: {e}")
        if isinstance(attr, str) and attr == "not found":
            raise ValueError(
                f"Did not find {self.name} in pipeline. "
                "Please check `attr_path` is correct."
            )
        return attr

    def _convert_to_df(self, attr: Any) -> pd.DataFrame:
        try:
            if self.to_df_fn is not None:
                df = self.to_df_fn(attr)
            else:
                df = pd.DataFrame(attr)
        except Exception as e:  # noqa: E722
            raise ValueError(
                f"Failed to convert {self.name} to a `pandas.FataFrame`: {e}"
            )
        return df

    @staticmethod
    def pad(df: pd.DataFrame, total_columns: int = 2000) -> pd.DataFrame:
        # Get the current maximum column number in the new data
        max_col_num = max(map(int, df.columns))

        # Determine the starting column number for padding
        start_col_num = max_col_num + 1

        # Pad the DataFrame to ensure it has total_columns columns
        if df.shape[1] < total_columns:
            # Add NaN columns to the new_data DataFrame
            padding_cols = total_columns - df.shape[1]
            df = df.reindex(
                columns=[
                    *df.columns,
                    *range(start_col_num, start_col_num + padding_cols),
                ],
                fill_value=np.nan,
            )
        elif df.shape[1] > total_columns:
            raise ValueError(f"New data has more than {total_columns} columns")

        return df


def create_pipeline(
    model: BaseEstimator,
    add_channel_dim: bool = False,
    flatten_feature_sets: bool = False,
    transformers: Optional[List[Tuple[str, BaseEstimator]]] = None,
    train_test_transformers: List[str] = [],
    # num_confounders: int = 0,
    weight_loss_by_groups: bool = False,
    weight_loss_by_class: bool = False,
    weight_per_split: bool = False,
    split_weights: Optional[Dict[str, float]] = None,
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> Pipeline:
    """
    Create `sklearn` pipeline for use during model training (e.g. within cross-validation).

    Parameters
    ----------
    model
        A model estimator to add as the last step in the pipeline.
        Should be instantiated.
    add_channel_dim
        Whether to add a transformer for adding a singleton dimension
        at the second index of the dataset. Useful when the model contains
        convolutional layers.
    flatten_feature_sets
        Whether to flatten feature sets
        just before the model. Should only be enabled for 3+d arrays.
    transformers
        List of tuples with (<name>, <transformer>).
        These are added as steps to the first part of the pipeline.
        Should handle 2- and/or 3-dimensional arrays.
            See `DimTransformerWrapper` for this.
        Can be built with `PipelineDesigner`.
    train_test_transformers
        Names of transformers with the `training_mode`
        parameter for enabling different functionality
        during training and testing.
        The resulting pipeline allows changing this mode
        with `.enable_testing_mode()` and `.enable_training_mode()`.
        NOTE: Testing mode is automatically enabled after fitting.
    weight_loss_by_groups
        Whether to weight samples by the group size in training loss.
        Each sample in a group gets the weight `1 / group_size`.
        Passed to model's `.fit(sample_weight=)` method.
        NOTE: When training with `NestableGridSearchCV`, this
        weighting is taken care of by that class!

    Returns
    -------
    `sklearn.pipeline.Pipeline`.
    """
    # Init list of steps (tuples with (name, transformer))
    steps = []

    # Add given transformers
    if transformers is not None:
        assert isinstance(transformers, list)
        steps += transformers

    designer = PipelineDesigner()

    # Flatten 3D structure to 2D
    # TODO: How to handle confounders here? I guess they need to be supplied as 3d as well
    # when they're concatenated?
    if flatten_feature_sets:
        designer.add_step(
            name="flatten_feature_sets",
            transformer="flatten_feature_sets",
            add_dim_transformer_wrapper=False,
        )

    # Add a transformer for adding a channel dimension (singleton)
    if add_channel_dim:
        designer.add_step(
            name="add_channel_dim",
            transformer="add_channel_dim",
            add_dim_transformer_wrapper=False,
        )

    if not steps:
        designer.add_step(
            name="identity",
            transformer="identity",
            add_dim_transformer_wrapper=False,
        )

    steps += designer.build()

    # Add the model to the pipeline when given
    if model is not None:
        steps.append(("model", model))

    # Create pipeline
    # Add flag for whether the model is seedable
    pipe = NestablePipeline(
        steps=steps,
        is_seedable=hasattr(model, "is_seedable") and model.is_seedable,
        train_test_steps=train_test_transformers,
        weight_loss_by_groups=weight_loss_by_groups,
        weight_loss_by_class=weight_loss_by_class,
        weight_per_split=weight_per_split,
        split_weights=split_weights,
        messenger=messenger,
    )

    return pipe


def pipeline_to_str(pipeline: Pipeline, with_extra_params: bool = False) -> str:
    """
    Create string that represents the full pipeline with
    all parameter settings without truncation.
    Non-tested ChatGPT code (works for my context).

    Parameters
    ----------
    pipeline
        Scikit-learn `Pipeline`.
    with_extra_params
        Whether to print additional parameters.

    Returns
    -------
    str
        The string representation of the pipeline.
    """
    # Convert pipeline to a serializable format
    serializable_pipeline = pipeline_to_serializable(
        pipeline, with_extra_params=with_extra_params
    )

    # Convert to JSON string with pretty printing
    pipeline_json_str = json.dumps(serializable_pipeline, indent=4).replace("\\n", "\n")

    # Print the entire pipeline without truncation
    return pipeline_json_str


# Function to convert the pipeline to a serializable format
def pipeline_to_serializable(pipeline, with_extra_params: bool = True):
    def get_params(step, with_extra_params: bool = True):
        params = step.get_params() if hasattr(step, "get_params") else {}
        serializable_params = {}
        for key, value in params.items():
            if not with_extra_params and key != "transformer_list":
                continue
            if isinstance(value, (int, float, str, bool, type(None))):
                serializable_params[key] = value
            elif isinstance(value, (list, tuple)):
                serializable_params[key] = [
                    (
                        str(v)
                        if not isinstance(v, (int, float, str, bool, type(None)))
                        else v
                    )
                    for v in value
                ]
            else:
                serializable_params[key] = str(value)
        return serializable_params

    return {
        "steps": [
            {
                "name": name,
                "class": step.__class__.__name__,
                "params": get_params(step, with_extra_params),
            }
            for name, step in pipeline.steps
        ]
    }
