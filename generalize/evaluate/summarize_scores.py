
import random
from typing import List, Optional
import numpy as np
import pandas as pd
from utipy import move_column_inplace


def summarize_data_frame(
    df: pd.DataFrame,
    by: Optional[List[str]] = None,
    threshold_versions: Optional[List[str]] = None,
    drop_cols: List[str] = []
) -> pd.DataFrame:
    """
    To summarize by thresholds, `df` and `by` must contain the column/name `"Threshold Version"`.
    It should be the last element in by.
    """

    # Reset the data frame
    df = df.copy().reset_index(drop=True)

    tmp_by_col = None
    if by is None:
        tmp_by_col = f"__.tmp_col.{random.randint(0, 10000)}.__"
        df[tmp_by_col] = 1
        by = [tmp_by_col]

    missing_bys = list(set(by).difference(set(df.columns)))
    if missing_bys:
        raise ValueError(
            "The following names in `by` were not columns in `df`: "
            f"{', '.join(missing_bys)}.")

    threshold_versions_enum = None
    if "Threshold Version" in by:
        if by[-1] != "Threshold Version":
            raise ValueError(
                "When summarizing by `'Threshold Version'`, "
                "it must be the last string in `by`."
            )

        if threshold_versions is None:
            raise ValueError(
                "When summarizingg by `'Threshold Version'`, "
                "`threshold_versions` must be specified."
            )

        # Enumerate threshold version to get requested ordering
        threshold_versions_enum = {
            val: idx
            for idx, val in enumerate(
                threshold_versions
            )
        }

    if drop_cols:
        missing_drops = list(set(drop_cols).difference(set(df.columns)))
        if missing_drops:
            raise ValueError(
                "The following names in `drop_cols` were not columns in `df`: "
                f"{', '.join(missing_drops)}.")
        df = df.drop(columns=drop_cols)

    # Non-summarizable object-type columns
    obj_cols = list(df.columns[df.dtypes.eq(object)])
    excessive_obj_cols = list(set(obj_cols).difference(set(by)))
    if excessive_obj_cols:
        raise ValueError(
            "The following columns in `df` were not numerically summarizable "
            "and should be added to `drop_cols` or `by`: "
            f"{', '.join(excessive_obj_cols)}.")

    # Summarize columns
    summary = summarize_cols_by(df=df, by=by, count_nans= True)

    # Fix threshold version sorting
    if threshold_versions_enum is not None:
        summary = summary \
            .sort_values(
                by="Threshold Version",
                key=lambda xs: [threshold_versions_enum[x] for x in xs],
                kind="stable"  # Maintain previous sortings
            ).reset_index(drop=True)

    for by_col in by:
        move_column_inplace(
            summary,
            by_col,
            pos=len(summary.columns) - 1
        )

    if tmp_by_col is not None:
        del summary[tmp_by_col]

    return summary


def summarize_cols_by(df: pd.DataFrame, by: List[str], count_nans: bool = True) -> pd.DataFrame:

    # Reset the data frame
    df = df.copy().reset_index(drop=True)

    if not by:
        raise ValueError("`by` was empty.")

    agg_fn_names = ["mean", "std", "min", "max"]
    if count_nans:
        agg_fn_names.append(lambda xs: np.sum(np.isnan(xs)))

    summary = df \
        .groupby(by) \
        .agg(agg_fn_names) \
        .stack() \
        .reset_index() \
        .rename(columns={f"level_{len(by)}": "Measure"})

    # Fix names in `Measure` column
    # Add numbering for sorting purposes (removed post-sort)
    measure_names = {
        "mean": "0_Average",
        "std": "1_SD",
        "min": "2_Min",
        "max": "3_Max",
        "<lambda_0>": "4_# NaNs"
    }
    summary["Measure"] = summary["Measure"].apply(lambda m: measure_names[m])

    # Sort by `by` columns and the `Measure`
    summary = summary \
        .sort_values(
            by + ["Measure"]
        ).reset_index(drop=True)

    # Remove numbering
    summary["Measure"] = summary["Measure"].apply(lambda m: m[2:])

    return summary
