"""Flatten nested comment lists into one row per comment and pair expert with LLM comments per task."""

from __future__ import annotations

import json

import pandas as pd

TASK_COLUMNS = ["app_category", "app", "screen_id", "screen_task_id", "task"]
EXPERT_FIELDS = ["comment_source", "expected_standard", "observed_issue", "suggested_fix", "bounding_box", "raw_text"]
LLM_FIELDS = ["expected_standard", "observed_issue", "suggested_fix", "guideline_reference"]


def explode_comments(task_df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per comment dict in `column`: the task's other columns followed by the comment's fields."""
    exploded = task_df.explode(column, ignore_index=True)
    exploded = exploded[exploded[column].map(lambda comment: isinstance(comment, dict))].reset_index(drop=True)
    return pd.concat([exploded.drop(columns=column), pd.DataFrame(exploded[column].tolist())], axis=1)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def expand_human_comments(human_df: pd.DataFrame) -> pd.DataFrame:
    comments = explode_comments(human_df, "comments")
    expert_df = comments[TASK_COLUMNS].assign(
        expert_comment_id=comments["comment_id"].astype(str),
        **{f"expert_{field}": comments[field] for field in EXPERT_FIELDS},
    )
    expert_df["expert_observed_issue"] = _clean_text(expert_df["expert_observed_issue"])
    return expert_df[expert_df["expert_observed_issue"].ne("")].reset_index(drop=True)


def expand_llm_comments(llm_df: pd.DataFrame, model_short: str) -> pd.DataFrame:
    """LLM critiques get ids L_<model>_<screen_task_id>_<nnn>, numbered in response order."""
    comments = explode_comments(llm_df.assign(critiques=llm_df["critiques"].map(json.loads)), "critiques")
    numbers = comments.groupby("screen_task_id", sort=False).cumcount() + 1
    missing = pd.Series(pd.NA, index=comments.index)
    # Models name this field either guideline_references or guideline_reference.
    comments["guideline_reference"] = comments.get("guideline_references", missing).combine_first(
        comments.get("guideline_reference", missing)
    )
    llm_df = pd.DataFrame({
        "screen_task_id": comments["screen_task_id"],
        "llm_comment_id": [f"L_{model_short}_{task}_{n:03d}" for task, n in zip(comments["screen_task_id"], numbers)],
        **{f"llm_{field}": comments.get(field, missing) for field in LLM_FIELDS},
    })
    llm_df["llm_observed_issue"] = _clean_text(llm_df["llm_observed_issue"])
    return llm_df[llm_df["llm_observed_issue"].ne("")].reset_index(drop=True)


def build_comment_pairs(expert_df: pd.DataFrame, llm_df: pd.DataFrame, model_short: str) -> pd.DataFrame:
    """Every expert x LLM comment pair of the same screen task.

    pair_id is <model>_<expert comment id>_<LLM comment number>, e.g. gpt5_10805_T02_004_001, unique across models.
    """
    pairs_df = (
        expert_df.merge(llm_df, on="screen_task_id")
        .sort_values(["screen_task_id", "expert_comment_id", "llm_comment_id"], kind="stable")
        .reset_index(drop=True)
    )
    pairs_df.insert(
        0, "pair_id", f"{model_short}_" + pairs_df["expert_comment_id"] + "_" + pairs_df["llm_comment_id"].str[-3:]
    )
    return pairs_df
