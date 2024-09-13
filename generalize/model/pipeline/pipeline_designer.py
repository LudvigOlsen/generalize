from typing import Union, List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


from generalize.model.transformers import (
    DimTransformerWrapper,
    FlattenFeatureSetsTransformer,
    RowScaler,
    IndexFeatureSelector,
    IndexFeatureRemover,
    MeanDuringTest,
)


@dataclass
class PipelineElement:
    name: str
    transformer: BaseEstimator

    def rm_split_name(self):
        """
        Remove the string part before the first "__".
        """
        self.name = "__".join(self.name.split("__")[1:])
        return self

    def to_tuple(self):
        return (self.name, self.transformer)


@dataclass
class PipelineSplit:
    name: str
    split_indices: Dict[str, List[int]]


@dataclass
class PipelineCollect:
    pass


def _add_channel_dimension(x):
    return np.expand_dims(x, 1)


def _rm_channel_dimension(x):
    return np.squeeze(x, 1)


def _identity(x):
    return x


class PipelineDesigner:
    # name -> (transformer, kwargs)
    PRECONFIGURED_TRANSFORMERS: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
        "identity": (
            FunctionTransformer,
            {"func": _identity, "inverse_func": _identity},
        ),
        "scale_features": (StandardScaler, {}),
        "scale_rows": (RowScaler, {}),
        "add_channel_dim": (
            FunctionTransformer,
            {"func": _add_channel_dimension, "inverse_func": _rm_channel_dimension},
        ),
        "flatten_feature_sets": (FlattenFeatureSetsTransformer, {}),
        "mean_during_test": (MeanDuringTest, {}),
        "feature_selector": (IndexFeatureSelector, {}),
        "feature_remover": (IndexFeatureRemover, {}),
    }

    def __init__(self) -> None:
        self.steps = []

    @property
    def n_transformers(self):
        """
        Get the number of transformers in the pipeline.
        This excludes any splitting and collecting steps.
        """

        return sum([isinstance(step, PipelineElement) for step in self.steps])

    def add_step(
        self,
        name: str,
        transformer: Union[str, BaseEstimator],
        add_dim_transformer_wrapper: bool,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(transformer, str):
            if transformer not in PipelineDesigner.PRECONFIGURED_TRANSFORMERS.keys():
                raise ValueError(
                    f"The provided transformer name was not recognized: {transformer}."
                )
            transformer, preconfig_kwargs = (
                PipelineDesigner.get_preconfigured_transformer(name=transformer)
            )
            if kwargs is not None:
                preconfig_kwargs.update(kwargs)
            kwargs = preconfig_kwargs

        # Initialize transformer
        transformer = PipelineDesigner._initialize_transformer(
            transformer=transformer,
            kwargs=kwargs,
            add_dim_transformer_wrapper=add_dim_transformer_wrapper,
        )

        self.steps.append(
            PipelineElement(
                name=name,
                transformer=transformer,
            )
        )

        return self

    def split(self, name, split_indices: Dict[str, List[int]]):
        """
        Splits features into separate pipelines.
        """
        self.steps.append(PipelineSplit(name=name, split_indices=split_indices))

        return self

    def collect(self):
        """
        Collects the output of split pipelines.
        Note the features are concatenated in the insertion
        order of the `split_indices` dict from `split()` and thus may
        differ from before the splitting.
        """
        self.steps.append(PipelineCollect())

        return self

    def build(self) -> List[Tuple[str, BaseEstimator]]:
        """
        Add steps with wrappers handling confounders etc.
        """
        pipeline_steps = []
        while self.steps:
            step = self.steps.pop(0)

            if isinstance(step, PipelineSplit):
                split_indices = step.split_indices
                split_name = step.name
                split_steps = {key: [] for key in split_indices.keys()}
                try:
                    # Gather all steps for the split
                    split_step = self.steps.pop(0)
                    while not isinstance(split_step, PipelineCollect):
                        split_steps[split_step.name.split("__")[0]].append(
                            split_step.rm_split_name()
                        )
                        split_step = self.steps.pop(0)
                except IndexError:
                    raise ValueError(
                        "A `split` step was not followed by a `collect` step."
                    )

                # Append the split pipelines
                # The final outputs are concatenated
                pipeline_steps += PipelineDesigner._make_split_pipelines(
                    split_name=split_name,
                    split_steps=split_steps,
                    split_indices=split_indices,
                )

            elif isinstance(step, PipelineCollect):
                raise ValueError("A `collect` step came before its `split` step.")

            elif isinstance(step, PipelineElement):
                pipeline_steps.append(step.to_tuple())

        return pipeline_steps

    @staticmethod
    def _make_split_pipelines(
        split_name: str,
        split_steps: Dict[str, List[PipelineElement]],
        split_indices: Dict[str, List[int]],
    ):
        return [
            (
                split_name,
                FeatureUnion(
                    n_jobs=1,
                    transformer_list=[
                        (
                            name,
                            PipelineDesigner._create_split_pipeline(
                                indices=idxs, elements=split_steps[name]
                            ),
                        )
                        for name, idxs in split_indices.items()
                        if len(idxs) > 0
                    ],
                ),
            )
        ]

    @staticmethod
    def _create_split_pipeline(indices: List[int], elements: List[PipelineElement]):
        # Create list of steps (tuples with (name, transformer))
        steps = [
            (
                "select_features",
                DimTransformerWrapper(
                    IndexFeatureSelector, kwargs={"feature_indices": indices}
                ),
            )
        ]

        steps += [e.to_tuple() for e in elements]

        # Handle empty pipelines
        if not len(elements):
            steps += [
                (
                    "identity",
                    FunctionTransformer(func=_identity, inverse_func=_identity),
                )
            ]

        return Pipeline(steps=steps)

    @staticmethod
    def get_preconfigured_transformer(
        name: str,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Get uninitialized preconfigured transformer and kwargs."""
        return PipelineDesigner.PRECONFIGURED_TRANSFORMERS[name]

    @staticmethod
    def _initialize_transformer(
        transformer: BaseEstimator,
        add_dim_transformer_wrapper: bool,
        kwargs: Optional[Dict[str, Any]],
    ):
        if add_dim_transformer_wrapper:
            return DimTransformerWrapper(transformer, kwargs=kwargs)
        return transformer(**kwargs) if kwargs is not None else transformer()
