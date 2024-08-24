from .model.cross_validate.nested_cross_validation import nested_cross_validate
from .model.cross_validate.cross_validate import cross_validate
from .model.univariate.evaluate_univariate_models import (
    evaluate_univariate_models,
    explain_feature_evaluation_df,
)
from .model.full_model import train_full_model
from .model.pipeline.pipeline_designer import PipelineDesigner
from .evaluate.evaluate import Evaluator
from .evaluate.roc_curves import ROCCurve, ROCCurves
from .dataset.load_dataset import load_dataset
from .dataset.order_dataset import order_by_label
from .dataset.parse_labels_to_use import parse_labels_to_use
from .dataset.subset_dataset import (
    select_samples,
    remove_features,
    remove_nan_features,
    select_feature_sets,
    select_indices,
)


def get_version():
    import importlib.metadata

    return importlib.metadata.version("generalize")


__version__ = get_version()
