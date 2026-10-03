from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

GENERAL_COLUMNS = ["app_category", "app", "screen_id", "screen_task_id", "task"]
EXPERT_COMMENT_COLUMNS = [
    "expected_standard",
    "observed_issue",
    "suggested_fix",
    "bounding_box",
    "raw_text",
]
LLM_COMMENT_COLUMNS = [
    "expected_standard",
    "observed_issue",
    "suggested_fix",
    "guideline_reference",
]
COMMENT_COLUMN_ALIASES = {
    "guideline_reference": ("guideline_references", "guideline_reference"),
}
COMMENT_ID_ALIASES = ("comment_id", "id")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def get_comment_id(comment: dict[str, Any]) -> str | None:
    for key in COMMENT_ID_ALIASES:
        value = comment.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return None


def get_comment_value(comment: dict[str, Any], col: str) -> Any:
    for key in COMMENT_COLUMN_ALIASES.get(col, (col,)):
        if key in comment:
            return comment[key]
    return pd.NA


def parse_payload(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith(("[", "{")) else [{"observed_issue": value}]
    elif not isinstance(value, (list, dict)):
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        value = [value]
    return [item for item in value if isinstance(item, dict)]


def validate_unique_ids(df: pd.DataFrame, id_col: str) -> None:
    duplicate_ids = df.loc[df[id_col].duplicated(), id_col].head(10).tolist()
    if duplicate_ids:
        raise ValueError(f"Duplicate {id_col} values: {duplicate_ids}")


def expand_comments_long(
    task_df: pd.DataFrame,
    *,
    payload_col: str,
    id_col: str,
    id_prefix: str,
    comment_prefix: str,
    base_cols: list[str],
    comment_cols: list[str],
    comment_source: str | None = None,
    model_short: str | None = None,
    use_existing_comment_id: bool = True,
    require_existing_comment_id: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, task_row in task_df.iterrows():
        screen_task_id = task_row["screen_task_id"]
        for comment_number, comment in enumerate(parse_payload(task_row[payload_col]), start=1):
            row = {col: task_row[col] if col in task_df.columns else pd.NA for col in base_cols}
            comment_id = get_comment_id(comment) if use_existing_comment_id else None
            if comment_id is None and require_existing_comment_id:
                raise ValueError(f"Missing comment_id in {payload_col} for screen_task_id={screen_task_id}.")
            row[id_col] = comment_id or f"{id_prefix}_{model_short}_{screen_task_id}_{comment_number:03d}"
            if comment_source is not None:
                row["comment_source"] = comment.get("comment_source", comment_source)
            for col in comment_cols:
                row[f"{comment_prefix}_{col}"] = get_comment_value(comment, col)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    observed_issue_col = f"{comment_prefix}_observed_issue"
    out[observed_issue_col] = out[observed_issue_col].map(clean_text)
    return out[out[observed_issue_col].ne("")].reset_index(drop=True)


def expand_human_comments(human_df: pd.DataFrame) -> pd.DataFrame:
    comments_df = expand_comments_long(
        human_df,
        payload_col="comments",
        id_col="expert_comment_id",
        id_prefix="H",
        comment_prefix="expert",
        base_cols=GENERAL_COLUMNS,
        comment_cols=EXPERT_COMMENT_COLUMNS,
        comment_source="human",
        require_existing_comment_id=True,
    )
    validate_unique_ids(comments_df, "expert_comment_id")
    return comments_df


def expand_llm_comments(llm_df: pd.DataFrame, model_short: str) -> pd.DataFrame:
    comments_df = expand_comments_long(
        llm_df,
        payload_col="critiques",
        id_col="llm_comment_id",
        id_prefix="L",
        comment_prefix="llm",
        base_cols=GENERAL_COLUMNS,
        comment_cols=LLM_COMMENT_COLUMNS,
        model_short=model_short,
        use_existing_comment_id=False,
    )
    validate_unique_ids(comments_df, "llm_comment_id")
    return comments_df


def build_comment_pairs(
    expert_comments_df: pd.DataFrame,
    llm_comments_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    task_col = "screen_task_id"
    shared_task_ids = [
        task_id
        for task_id in expert_comments_df[task_col].drop_duplicates().tolist()
        if task_id in set(llm_comments_df[task_col])
    ]

    expert_cols = expert_comments_df.columns.tolist()
    llm_cols = [col for col in llm_comments_df.columns if col not in [*GENERAL_COLUMNS, "comment_source", task_col]]

    pairs_df = expert_comments_df[expert_comments_df[task_col].isin(shared_task_ids)][expert_cols].merge(
        llm_comments_df[llm_comments_df[task_col].isin(shared_task_ids)][[task_col, *llm_cols]],
        on=task_col,
        how="inner",
        sort=False,
    )
    pairs_df = pairs_df.sort_values(
        [task_col, "expert_comment_id", "llm_comment_id"],
        kind="stable",
    ).reset_index(drop=True)
    if "comment_source" in pairs_df.columns:
        pairs_df = pairs_df.rename(columns={"comment_source": "expert_comment_source"})
    pairs_df.insert(0, "pair_id", [f"P_{i:08d}" for i in range(1, len(pairs_df) + 1)])
    return pairs_df, sorted(shared_task_ids)
