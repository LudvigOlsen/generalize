# Changelog

## 0.3.0 

**Breaking changes**:

 - Replaces the use of "truncate" / "truncation" with "clip" / "clipping". This changes the names of the `TruncateExtremes*` functions to `ClipExtremes*`. The functionality stays the same, but this is more precise terminology. By clipping, we mean "replacing value beyond a threshold with the value at that threshold".

## 0.2.3

 - Fixes calculation of interpolated specificity when calling `ROCCurve.get_threshold_at_sensitivity()`.
 - Bumps joblib dependency to `1.4.2`.
 - Improves `parse_labels_to_use`.

## 0.2.2

 - Adds project URLs to package to list them on PyPI.
 - Updates pyproject.toml for newer poetry version.
