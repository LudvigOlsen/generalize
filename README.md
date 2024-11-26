# Generalize <a href='https://github.com/LudvigOlsen/generalize'><img src='https://raw.githubusercontent.com/LudvigOlsen/generalize/master/generalize_242x280_250dpi.png' align="right" height="140" /></a>

**Author:** [Ludvig R. Olsen](https://www.ludvigolsen.dk/) ( <r-pkgs@ludvigolsen.dk> )

The ultimate goal of training machine learning models is to generalize to new, unseen data. This package contains tools for measuring model performance across multiple datasets via cross-dataset-validation (aka. leave-one-dataset-out).

Under development!

 - Not generalized enough for general usage (ironic, I know)
 - Relies on an old version of scikit-learn, needs updating
 - Linear regression is not currently implemented
 - Help strings are likely not up-to-date

### Main functions and classes

| Function                       | Description                                                                        |
|:-------------------------------|:-----------------------------------------------------------------------------------|
| `nested_cross_validate()`      | Run (repeated) nested cross-validation.                                            |
| `train_full_model()`           | Train model on all data and save to disk.                                          |
| `evaluate_univariate_models()` | Evaluate prediction potential of every predictor separately.                       |
| `PipelineDesigner`             | Design a scikit-learn pipeline for use in cross-validation.                        |
| `ROCCurve`, `ROCCurves`        | ROC curve containers with various utility methods.                                 |
| `select_samples()`             | Utility for selecting samples based on (collapsed) labels.                         |