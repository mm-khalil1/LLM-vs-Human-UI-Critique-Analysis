from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .embeddings import prepare_match_text

STANDARD_COMMENT_CORE_COLUMNS = [
    "comment_id",
    "screen_id",
    "app",
    "app_category",
    "screen_task_id",
    "task",
    "comment_source",
    "expected_standard",
    "observed_issue",
    "suggested_fix",
]
STANDARD_OPTIONAL_COLUMN_ORDER = [
    "missing_parts",
    "guideline_reference",
    "model_name",
    "bounding_box",
    "raw_text",
]


def coerce_missing_parts_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, (np.ndarray, pd.Series)):
        return list(value)
    if value is None:
        return []
    if pd.isna(value):
        return []
    return [value]


def build_standardized_human_comments_from_nested(
    task_level_df: pd.DataFrame,
    *,
    nested_comments_col: str = "comments",
    comment_id_prefix: str = "H",
) -> pd.DataFrame:
    comments_raw_df = task_level_df.explode(nested_comments_col, ignore_index=True).copy()
    comments_raw_df = comments_raw_df[comments_raw_df[nested_comments_col].notna()].copy()

    comments_raw_df["bounding_box"] = comments_raw_df[nested_comments_col].apply(
        lambda x: x.get("bounding_box") if isinstance(x, dict) else None
    )
    comment_fields = comments_raw_df[nested_comments_col].apply(
        lambda d: {k: v for k, v in d.items() if k != "bounding_box"} if isinstance(d, dict) else {}
    )
    comment_fields = pd.json_normalize(comment_fields)

    comments_raw_df = pd.concat(
        [comments_raw_df.drop(columns=[nested_comments_col]).reset_index(drop=True), comment_fields.reset_index(drop=True)],
        axis=1,
    )
    if "comment_id" not in comments_raw_df.columns:
        comments_raw_df.insert(0, "comment_id", [f"{comment_id_prefix}_{i+1:06d}" for i in range(len(comments_raw_df))])
    else:
        comments_raw_df["comment_id"] = comments_raw_df["comment_id"].astype(str)

    if "missing_parts" not in comments_raw_df.columns:
        comments_raw_df["missing_parts"] = [[] for _ in range(len(comments_raw_df))]
    comments_raw_df["missing_parts"] = comments_raw_df["missing_parts"].apply(coerce_missing_parts_list)

    standardized_df = standardize_comment_table(
        comments_raw_df,
        comment_id_col="comment_id",
        comment_source="human",
        optional_column_map={
            "missing_parts": "missing_parts",
            "raw_text": "raw_text",
            "bounding_box": "bounding_box",
            "comment_label": "comment_label",
            "comment_source": "comment_source",
        },
    )
    standardized_df["comment_source"] = (
        standardized_df["comment_source"].astype("string").str.strip().str.lower().fillna("human")
    )
    return standardized_df

def build_standardized_llm_comments_from_nested(
    task_level_df: pd.DataFrame,
    *,
    nested_comments_col: str = "critiques",
    comment_id_prefix: str = "L",
    model_name: Optional[str] = None,
    model_code: Optional[str] = None,
) -> pd.DataFrame:
    comments_raw_df = task_level_df.explode(nested_comments_col, ignore_index=True).copy()
    comments_raw_df = comments_raw_df[comments_raw_df[nested_comments_col].notna()].copy()

    comment_fields = comments_raw_df[nested_comments_col].apply(lambda x: x if isinstance(x, dict) else {})
    comment_fields = pd.json_normalize(comment_fields)

    comments_raw_df = pd.concat(
        [comments_raw_df.drop(columns=[nested_comments_col]).reset_index(drop=True), comment_fields.reset_index(drop=True)],
        axis=1,
    )

    id_prefix = f"{comment_id_prefix}_{model_code}" if model_code else comment_id_prefix
    comments_raw_df.insert(0, "comment_id", [f"{id_prefix}_{i+1:06d}" for i in range(len(comments_raw_df))])
    if model_name is not None:
        comments_raw_df["model_name"] = model_name

    if "app_category" not in comments_raw_df.columns:
        comments_raw_df["app_category"] = pd.NA

    for col in ["expected_standard", "observed_issue", "suggested_fix", "guideline_reference"]:
        if col not in comments_raw_df.columns:
            comments_raw_df[col] = pd.NA

    return standardize_comment_table(
        comments_raw_df,
        comment_id_col="comment_id",
        comment_source="llm",
        optional_column_map={
            "guideline_reference": "guideline_reference",
            "model_name": "model_name",
        },
    )

def standardize_comment_table(
    df: pd.DataFrame,
    *,
    comment_id_col: str,
    comment_source: str,
    screen_id_col: str = "screen_id",
    screen_task_id_col: str = "screen_task_id",
    task_col: str = "task",
    app_col: str = "app",
    app_category_col: str = "app_category",
    observed_issue_col: str = "observed_issue",
    expected_standard_col: str = "expected_standard",
    suggested_fix_col: str = "suggested_fix",
    optional_column_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    standardized_df = pd.DataFrame(
        {
            "comment_id": df[comment_id_col],
            "comment_source": comment_source,
            "screen_id": df[screen_id_col],
            "app": df[app_col] if app_col in df.columns else pd.NA,
            "screen_task_id": df[screen_task_id_col],
            "task": df[task_col],
            "app_category": df[app_category_col] if app_category_col in df.columns else pd.NA,
            "observed_issue": df[observed_issue_col],
            "expected_standard": df[expected_standard_col] if expected_standard_col in df.columns else pd.NA,
            "suggested_fix": df[suggested_fix_col] if suggested_fix_col in df.columns else pd.NA,
        }
    )

    for col in STANDARD_COMMENT_CORE_COLUMNS:
        if col not in standardized_df.columns:
            standardized_df[col] = pd.NA

    standardized_df["match_text"] = standardized_df["observed_issue"].apply(prepare_match_text)

    if optional_column_map:
        for new_col, old_col in optional_column_map.items():
            mapped = df[old_col] if old_col in df.columns else pd.Series(pd.NA, index=df.index)
            if new_col in standardized_df.columns:
                standardized_df[new_col] = mapped.where(mapped.notna(), standardized_df[new_col])
            else:
                standardized_df[new_col] = mapped

    core_plus = STANDARD_COMMENT_CORE_COLUMNS + ["match_text"]
    ordered_optional_cols = [
        col for col in STANDARD_OPTIONAL_COLUMN_ORDER if col in standardized_df.columns and col not in core_plus
    ]
    extra_cols = [c for c in standardized_df.columns if c not in core_plus + ordered_optional_cols]
    return standardized_df[core_plus + ordered_optional_cols + extra_cols].copy()

def build_standardized_pairs(
    left_comments_df: pd.DataFrame,
    right_comments_df: pd.DataFrame,
    *,
    pair_id_prefix: str = "PAIR",
    exclude_same_comment_id_pairs: bool = False,
) -> pd.DataFrame:
    shared_cols = ["screen_id", "app", "app_category", "screen_task_id", "task"]
    comment_cols = [
        "comment_id",
        "comment_source",
        "observed_issue",
        "expected_standard",
        "suggested_fix",
        "match_text",
    ]
    required_cols = shared_cols + comment_cols
    missing_left_cols = [col for col in required_cols if col not in left_comments_df.columns]
    missing_right_cols = [col for col in required_cols if col not in right_comments_df.columns]
    if missing_left_cols:
        raise ValueError(f"Left comments table is missing required columns: {missing_left_cols}")
    if missing_right_cols:
        raise ValueError(f"Right comments table is missing required columns: {missing_right_cols}")

    left_extra_cols = [col for col in left_comments_df.columns if col not in required_cols]
    right_extra_cols = [col for col in right_comments_df.columns if col not in required_cols]
    left_base_df = left_comments_df[shared_cols + comment_cols + left_extra_cols].copy()
    right_base_df = right_comments_df[shared_cols + comment_cols + right_extra_cols].copy()

    pairs_df = left_base_df.merge(
        right_base_df,
        on=shared_cols,
        how="inner",
        suffixes=("_left", "_right"),
    )

    if exclude_same_comment_id_pairs:
        pairs_df = pairs_df[pairs_df["comment_id_left"] != pairs_df["comment_id_right"]].copy()

    pairs_df.insert(0, "pair_id", [f"{pair_id_prefix}_{i+1:07d}" for i in range(len(pairs_df))])
    pairs_df = pairs_df.rename(
        columns={
            col: f"left_{col.removesuffix('_left')}"
            for col in pairs_df.columns
            if col.endswith("_left")
        }
        | {
            col: f"right_{col.removesuffix('_right')}"
            for col in pairs_df.columns
            if col.endswith("_right")
        }
    )
    shared_output_cols = ["pair_id", *shared_cols]
    left_cols = [
        "left_comment_id",
        "left_comment_source",
        "left_observed_issue",
        "left_expected_standard",
        "left_suggested_fix",
        "left_match_text",
    ]
    right_cols = [
        "right_comment_id",
        "right_comment_source",
        "right_observed_issue",
        "right_expected_standard",
        "right_suggested_fix",
        "right_match_text",
    ]
    ordered_cols = [col for col in shared_output_cols + left_cols + right_cols if col in pairs_df.columns]
    extra_cols = [col for col in pairs_df.columns if col not in ordered_cols]
    return pairs_df[ordered_cols + extra_cols].copy()

def select_similarity_band(
    pairs_df: pd.DataFrame,
    *,
    score_col: str = "cosine_similarity",
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    sort_cols: Optional[list[str]] = None,
    ascending: Optional[list[bool]] = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=pairs_df.index)
    if lower_bound is not None:
        mask &= pairs_df[score_col] >= lower_bound
    if upper_bound is not None:
        mask &= pairs_df[score_col] < upper_bound

    selected_df = pairs_df[mask].copy()
    if sort_cols:
        selected_df = selected_df.sort_values(
            sort_cols,
            ascending=ascending if ascending is not None else True,
            kind="stable",
        )
    return selected_df
