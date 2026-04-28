from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from IPython.display import display
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_REVIEW_MODEL = "gpt-5-mini"
DEFAULT_REVIEW_SCHEMA_NAME = "semantic_comment_relationship_judgment"
DEFAULT_REVIEW_SYSTEM_PROMPT = (
    "You judge the relationship between two UI critique comments written about the same task. "
    "The comments may be exact paraphrases, one may fully contain the other plus extra detail, "
    "they may partially overlap, or they may be different. "
    "Be conservative about merging and return valid JSON only."
)
DEFAULT_RELATIONSHIP_OPTIONS = [
    "duplicate",
    "a_contains_b",
    "b_contains_a",
    "partial_overlap",
    "different",
]
DEFAULT_ACTION_OPTIONS = [
    "keep_A",
    "keep_B",
    "keep_both",
    "manual_review",
]
STANDARD_COMMENT_CORE_COLUMNS = [
    "comment_id",
    "screen_id",
    "app_category",
    "screen_task_id",
    "task",
    "source_type",
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
DEFAULT_LEFT_COMMENT_COL = "left_observed_issue"
DEFAULT_RIGHT_COMMENT_COL = "right_observed_issue"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def prepare_match_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_embeddings_df(
    items_df: pd.DataFrame,
    *,
    id_col: str,
    text_col: str,
    model: SentenceTransformer,
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    embedding_input_df = items_df[[id_col, text_col]].drop_duplicates(subset=[id_col]).copy()
    embedding_input_df[text_col] = embedding_input_df[text_col].apply(prepare_match_text)

    embeddings = model.encode(
        embedding_input_df[text_col].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return pd.DataFrame(
        {
            id_col: embedding_input_df[id_col].tolist(),
            embedding_col: list(embeddings),
        }
    )


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
    comments_raw_df.insert(0, "comment_id", [f"{comment_id_prefix}_{i+1:06d}" for i in range(len(comments_raw_df))])

    if "missing_parts" not in comments_raw_df.columns:
        comments_raw_df["missing_parts"] = [[] for _ in range(len(comments_raw_df))]
    comments_raw_df["missing_parts"] = comments_raw_df["missing_parts"].apply(coerce_missing_parts_list)

    return standardize_comment_table(
        comments_raw_df,
        comment_id_col="comment_id",
        source_type="human",
        optional_column_map={
            "missing_parts": "missing_parts",
            "raw_text": "raw_text",
            "bounding_box": "bounding_box",
            "comment_label": "comment_label",
        },
    )


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
        source_type="llm",
        optional_column_map={
            "guideline_reference": "guideline_reference",
            "model_name": "model_name",
        },
    )


def standardize_comment_table(
    df: pd.DataFrame,
    *,
    comment_id_col: str,
    source_type: str,
    screen_id_col: str = "screen_id",
    screen_task_id_col: str = "screen_task_id",
    task_col: str = "task",
    app_category_col: str = "app_category",
    observed_issue_col: str = "observed_issue",
    expected_standard_col: str = "expected_standard",
    suggested_fix_col: str = "suggested_fix",
    optional_column_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    standardized_df = pd.DataFrame(
        {
            "comment_id": df[comment_id_col],
            "source_type": source_type,
            "screen_id": df[screen_id_col],
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
            standardized_df[new_col] = df[old_col] if old_col in df.columns else pd.NA

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
    group_col: str = "screen_task_id",
) -> pd.DataFrame:
    left_core_cols = [
        "comment_id",
        "source_type",
        "screen_id",
        "screen_task_id",
        "task",
        "observed_issue",
        "expected_standard",
        "suggested_fix",
        "match_text",
    ]
    right_core_cols = left_core_cols.copy()

    left_extra_cols = [c for c in left_comments_df.columns if c not in left_core_cols + ["app_category"]]
    right_extra_cols = [c for c in right_comments_df.columns if c not in right_core_cols + ["app_category"]]

    join_base_cols = [group_col]
    if group_col != "screen_id":
        join_base_cols.append("screen_id")
    if group_col != "screen_task_id":
        join_base_cols.append("screen_task_id")

    left_base_df = left_comments_df[
        join_base_cols + ["app_category"] + [c for c in left_core_cols if c not in join_base_cols] + left_extra_cols
    ].copy()
    right_base_df = right_comments_df[right_core_cols + right_extra_cols].copy()

    pairs_df = left_base_df.merge(
        right_base_df,
        on=group_col,
        how="inner",
        suffixes=("_left", "_right"),
    )

    if group_col == "screen_task_id":
        if "screen_id_left" in pairs_df.columns and "screen_id_right" in pairs_df.columns:
            mismatched_screen_ids = pairs_df["screen_id_left"] != pairs_df["screen_id_right"]
            if mismatched_screen_ids.any():
                raise ValueError("Found mismatched screen_id values inside a screen_task_id pair group.")
            pairs_df["screen_id"] = pairs_df["screen_id_left"]
            pairs_df = pairs_df.drop(columns=["screen_id_left", "screen_id_right"])
    elif group_col == "screen_id":
        pairs_df["screen_id"] = pairs_df[group_col]
    else:
        if "screen_id_left" in pairs_df.columns and "screen_id_right" in pairs_df.columns:
            mismatched_screen_ids = pairs_df["screen_id_left"] != pairs_df["screen_id_right"]
            if mismatched_screen_ids.any():
                raise ValueError("Found mismatched screen_id values inside a pair group.")
            pairs_df["screen_id"] = pairs_df["screen_id_left"]
            pairs_df = pairs_df.drop(columns=["screen_id_left", "screen_id_right"])

        if "screen_task_id_left" in pairs_df.columns and "screen_task_id_right" in pairs_df.columns:
            mismatched_task_ids = pairs_df["screen_task_id_left"] != pairs_df["screen_task_id_right"]
            if mismatched_task_ids.any():
                raise ValueError("Found mismatched screen_task_id values inside a pair group.")

    if exclude_same_comment_id_pairs:
        pairs_df = pairs_df[pairs_df["comment_id_left"] != pairs_df["comment_id_right"]].copy()

    pairs_df.insert(0, "pair_id", [f"{pair_id_prefix}_{i+1:07d}" for i in range(len(pairs_df))])
    pairs_df = pairs_df.rename(
        columns={
            "app_category": "app_category",
            "comment_id_left": "left_comment_id",
            "source_type_left": "left_source_type",
            "screen_task_id_left": "left_screen_task_id",
            "task_left": "left_task",
            "observed_issue_left": "left_observed_issue",
            "expected_standard_left": "left_expected_standard",
            "suggested_fix_left": "left_suggested_fix",
            "match_text_left": "left_match_text",
            "comment_id_right": "right_comment_id",
            "source_type_right": "right_source_type",
            "screen_task_id_right": "right_screen_task_id",
            "task_right": "right_task",
            "observed_issue_right": "right_observed_issue",
            "expected_standard_right": "right_expected_standard",
            "suggested_fix_right": "right_suggested_fix",
            "match_text_right": "right_match_text",
        }
    )
    return pairs_df


def score_standardized_pairs(
    pairs_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    *,
    embedding_model: SentenceTransformer,
    comment_id_col: str = "comment_id",
    match_text_col: str = "match_text",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    embeddings_df = build_embeddings_df(
        comments_df[[comment_id_col, match_text_col]].copy(),
        id_col=comment_id_col,
        text_col=match_text_col,
        model=embedding_model,
    )

    scored_df = pairs_df.merge(
        embeddings_df.rename(columns={comment_id_col: "left_comment_id", "embedding": "left_embedding"}),
        on="left_comment_id",
        how="left",
    ).merge(
        embeddings_df.rename(columns={comment_id_col: "right_comment_id", "embedding": "right_embedding"}),
        on="right_comment_id",
        how="left",
    )

    scored_df["cosine_similarity"] = scored_df.apply(
        lambda row: cosine_similarity(row["left_embedding"], row["right_embedding"]),
        axis=1,
    )
    return scored_df, embeddings_df


def build_standard_preview_df(
    pairs_df: pd.DataFrame,
    *,
    include_score: bool = True,
    limit: int = 10,
    sort_cols: Optional[list[str]] = None,
    ascending: Optional[list[bool]] = None,
    
) -> pd.DataFrame:
    preview_cols = [
        "screen_id",
        "screen_task_id",
        "left_observed_issue",
        "right_observed_issue",
    ]
    if include_score and "cosine_similarity" in pairs_df.columns:
        preview_cols.append("cosine_similarity")

    available_cols = [c for c in preview_cols if c in pairs_df.columns]
    preview_df = pairs_df[available_cols].copy()
    if sort_cols:
        preview_df = preview_df.sort_values(
            sort_cols,
            ascending=ascending if ascending is not None else True,
            kind="stable",
        )
    
    return preview_df.head(limit)


def cosine_similarity(vec_a: object, vec_b: object) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


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


def build_review_json_schema(
    *,
    schema_name: str = DEFAULT_REVIEW_SCHEMA_NAME,
    relationship_options: Optional[Iterable[str]] = None,
    action_options: Optional[Iterable[str]] = None,
) -> dict:
    relationship_values = list(relationship_options or DEFAULT_RELATIONSHIP_OPTIONS)
    action_values = list(action_options or DEFAULT_ACTION_OPTIONS)

    return {
        "name": schema_name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relationship": {
                    "type": "string",
                    "enum": relationship_values,
                },
                "recommended_action": {
                    "type": "string",
                    "enum": action_values,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["relationship", "recommended_action", "confidence"],
            "additionalProperties": False,
        },
    }


def build_semantic_review_user_prompt(
    row: pd.Series,
    *,
    screen_id_col: str,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    screen_task_id_col: Optional[str] = None,
    left_screen_task_id_col: Optional[str] = None,
    right_screen_task_id_col: Optional[str] = None,
    left_source_col: Optional[str] = None,
    right_source_col: Optional[str] = None,
) -> str:
    lines = []

    lines.append("Comment A")

    if left_source_col:
        lines.append(f"- source: {row[left_source_col]}")
    if left_screen_task_id_col and not screen_task_id_col:
        lines.append(f"- screen_task_id: {row[left_screen_task_id_col]}")
    lines.append(f"- observed_issue: {row[left_comment_col]}")

    lines.extend(["", "Comment B"])
    if right_source_col:
        lines.append(f"- source: {row[right_source_col]}")
    if right_screen_task_id_col and not screen_task_id_col:
        lines.append(f"- screen_task_id: {row[right_screen_task_id_col]}")
    lines.append(f"- observed_issue: {row[right_comment_col]}")

    lines.extend(
        [
            "",
            "Decide the relationship using one of:",
            "- duplicate: same issue, same meaning, different wording at most",
            "- a_contains_b: A includes B's meaning plus extra detail",
            "- b_contains_a: B includes A's meaning plus extra detail",
            "- partial_overlap: related and overlapping, but not safe to merge automatically",
            "- different: different issue",
            "",
            "Then choose a recommended action:",
            "- keep_A",
            "- keep_B",
            "- keep_both",
            "- manual_review",
            "",
            "Then provide a confidence number between 0 and 1",
            "",
            'Return JSON with exactly these keys:',
            '{"relationship": string, "recommended_action": string, "confidence": number}',
        ]
    )
    return "\n".join(lines)


def build_semantic_review_request(
    row: pd.Series,
    *,
    custom_id: str,
    screen_id_col: str,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    screen_task_id_col: Optional[str] = None,
    left_screen_task_id_col: Optional[str] = None,
    right_screen_task_id_col: Optional[str] = None,
    left_source_col: Optional[str] = None,
    right_source_col: Optional[str] = None,
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_REVIEW_SCHEMA_NAME,
) -> dict:
    user_prompt = build_semantic_review_user_prompt(
        row,
        screen_id_col=screen_id_col,
        left_comment_col=left_comment_col,
        right_comment_col=right_comment_col,
        screen_task_id_col=screen_task_id_col,
        left_screen_task_id_col=left_screen_task_id_col,
        right_screen_task_id_col=right_screen_task_id_col,
        left_source_col=left_source_col,
        right_source_col=right_source_col,
    )

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": build_review_json_schema(schema_name=schema_name),
            },
        },
    }


def build_semantic_review_requests(
    candidate_df: pd.DataFrame,
    *,
    custom_id_col: Optional[str] = None,
    custom_id_prefix: str = "semantic_pair",
    left_id_col: str,
    right_id_col: str,
    screen_id_col: str,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    screen_task_id_col: Optional[str] = None,
    left_screen_task_id_col: Optional[str] = None,
    right_screen_task_id_col: Optional[str] = None,
    left_source_col: Optional[str] = None,
    right_source_col: Optional[str] = None,
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_REVIEW_SCHEMA_NAME,
) -> list[dict]:
    requests = []
    for _, row in candidate_df.iterrows():
        if custom_id_col:
            custom_id = str(row[custom_id_col])
        else:
            custom_id = f"{custom_id_prefix}::{row[left_id_col]}::{row[right_id_col]}"

        requests.append(
            build_semantic_review_request(
                row,
                custom_id=custom_id,
                screen_id_col=screen_id_col,
                left_comment_col=left_comment_col,
                right_comment_col=right_comment_col,
                screen_task_id_col=screen_task_id_col,
                left_screen_task_id_col=left_screen_task_id_col,
                right_screen_task_id_col=right_screen_task_id_col,
                left_source_col=left_source_col,
                right_source_col=right_source_col,
                model=model,
                system_prompt=system_prompt,
                schema_name=schema_name,
            )
        )
    return requests


def chunk_list(items: list, chunk_size: int) -> list[list]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def prepare_review_chunks(
    candidate_df: pd.DataFrame,
    *,
    chunk_size: int,
    custom_id_col: Optional[str] = None,
    custom_id_prefix: str = "semantic_pair",
    left_id_col: str,
    right_id_col: str,
    screen_id_col: str,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    screen_task_id_col: Optional[str] = None,
    left_screen_task_id_col: Optional[str] = None,
    right_screen_task_id_col: Optional[str] = None,
    left_source_col: Optional[str] = None,
    right_source_col: Optional[str] = None,
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_REVIEW_SCHEMA_NAME,
) -> tuple[list[dict], list[list[dict]], list[pd.DataFrame]]:
    requests = build_semantic_review_requests(
        candidate_df,
        custom_id_col=custom_id_col,
        custom_id_prefix=custom_id_prefix,
        left_id_col=left_id_col,
        right_id_col=right_id_col,
        screen_id_col=screen_id_col,
        left_comment_col=left_comment_col,
        right_comment_col=right_comment_col,
        screen_task_id_col=screen_task_id_col,
        left_screen_task_id_col=left_screen_task_id_col,
        right_screen_task_id_col=right_screen_task_id_col,
        left_source_col=left_source_col,
        right_source_col=right_source_col,
        model=model,
        system_prompt=system_prompt,
        schema_name=schema_name,
    )
    request_chunks = chunk_list(requests, chunk_size)
    candidate_chunks = [
        pd.DataFrame(chunk)
        for chunk in chunk_list(candidate_df.to_dict(orient="records"), chunk_size)
    ]
    return requests, request_chunks, candidate_chunks


def write_jsonl(records: Iterable[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def extract_message_content_from_batch_record(record: dict) -> str:
    response_body = record.get("response", {}).get("body", {})
    choices = response_body.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "") or ""


def parse_review_message_content(message_content: str) -> dict:
    if not message_content:
        return {}
    return json.loads(message_content)


def parse_semantic_review_results_jsonl(
    results_path: Path | str,
    candidate_df: pd.DataFrame,
    *,
    left_id_col: str,
    right_id_col: str,
    include_candidate_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    def _resolve_candidate_value(pair_row: pd.Series, col: str):
        if col in pair_row.index:
            return pair_row.get(col)
        if col == "task":
            left_task = pair_row.get("left_task")
            right_task = pair_row.get("right_task")
            return left_task if pd.notna(left_task) else right_task
        return None

    results_path = Path(results_path)
    candidate_lookup = candidate_df.copy()

    result_rows = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            custom_id = record.get("custom_id")
            message_content = extract_message_content_from_batch_record(record)
            parsed = parse_review_message_content(message_content)

            parts = (custom_id or "").split("::")
            left_id = parts[1] if len(parts) > 2 else None
            right_id = parts[2] if len(parts) > 2 else None

            matched_rows = candidate_lookup[
                (candidate_lookup[left_id_col].astype(str) == str(left_id))
                & (candidate_lookup[right_id_col].astype(str) == str(right_id))
            ]
            pair_row = matched_rows.iloc[0] if len(matched_rows) else None

            row_data = {
                "custom_id": custom_id,
                "left_id": left_id,
                "right_id": right_id,
                "relationship": parsed.get("relationship"),
                "recommended_action": parsed.get("recommended_action"),
                "confidence": parsed.get("confidence"),
            }

            if pair_row is not None and include_candidate_columns:
                for col in include_candidate_columns:
                    row_data[col] = _resolve_candidate_value(pair_row, col)

            result_rows.append(row_data)

    return pd.DataFrame(result_rows)


def identify_review_needed(
    results_df: pd.DataFrame,
    *,
    action_col: str = "recommended_action",
    confidence_col: str = "confidence",
    manual_review_value: str = "manual_review",
    confidence_threshold: float = 0.90,
) -> pd.DataFrame:
    return results_df[
        (results_df[action_col] == manual_review_value)
        | (results_df[confidence_col].fillna(0) < confidence_threshold)
    ].copy().sort_values([confidence_col], ascending=[True])


def inspect_manual_review_rows(
    review_needed_df: pd.DataFrame,
    *,
    start: int = 0,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    task_col: str = "screen_task_id",
    task_name_col: str = "task",
    left_task_col: str = "left_screen_task_id",
    right_task_col: str = "right_screen_task_id",
    relationship_col: str = "relationship",
    action_col: str = "recommended_action",
    confidence_col: str = "confidence",
) -> pd.DataFrame:
    if review_needed_df is None or len(review_needed_df) == 0:
        if review_needed_df is not None:
            return pd.DataFrame(columns=list(review_needed_df.columns) + ["final_decision"])
        return pd.DataFrame()

    decisions = {}
    i = start
    n = len(review_needed_df)

    help_txt = """
Commands:
  a      -> keep_A
  b      -> keep_B
  both   -> keep_both
  m      -> manual_review
  n      -> next
  p      -> previous
  show   -> show current saved decisions
  q      -> quit and return decisions dataframe
  help   -> show this help
""".strip()

    while 0 <= i < n:
        row = review_needed_df.iloc[i]
        row_idx = review_needed_df.index[i]

        print("\n" + "=" * 110)
        print(f"{i+1}/{n} | row_index={row_idx} | confidence={row.get(confidence_col)}")
        print("-" * 110)
        if task_col in row.index:
            print("screen_task_id      :", row.get(task_col))
            if task_name_col in row.index:
                print("task                :", row.get(task_name_col))
        else:
            print("left_screen_task_id :", row.get(left_task_col))
            print("right_screen_task_id:", row.get(right_task_col))
        print("relationship       :", row.get(relationship_col))
        print("recommended_action :", row.get(action_col))
        print("-" * 110)
        print("LEFT TEXT:")
        print(row.get(left_comment_col))
        print("-" * 110)
        print("RIGHT TEXT:")
        print(row.get(right_comment_col))
        print("-" * 110)
        print("Current final decision:", decisions.get(row_idx))

        cmd = input("cmd (help for options): ").strip().lower()

        if cmd == "q":
            break
        if cmd == "help":
            print(help_txt)
            continue
        if cmd == "show":
            current = review_needed_df.copy()
            current["final_decision"] = current.index.map(decisions)
            display(current)
            continue
        if cmd == "p":
            i = max(0, i - 1)
            continue
        if cmd in {"n", ""}:
            i += 1
            continue
        if cmd == "a":
            decisions[row_idx] = "keep_A"
            i += 1
            continue
        if cmd == "b":
            decisions[row_idx] = "keep_B"
            i += 1
            continue
        if cmd == "both":
            decisions[row_idx] = "keep_both"
            i += 1
            continue
        if cmd == "m":
            decisions[row_idx] = "manual_review"
            i += 1
            continue

        print("Unknown command. Type 'help' to see available commands.")

    result = review_needed_df.copy()
    result["final_decision"] = result.index.map(decisions)
    return result


def finalize_review_decisions(
    results_df: pd.DataFrame,
    manual_review_decisions_df: Optional[pd.DataFrame] = None,
    *,
    action_col: str = "recommended_action",
    final_decision_col: str = "final_decision",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted_df = results_df.copy()
    accepted_df[final_decision_col] = accepted_df[action_col]

    if manual_review_decisions_df is not None and len(manual_review_decisions_df):
        override_map = manual_review_decisions_df[final_decision_col].dropna().to_dict()
        override_series = accepted_df.index.to_series().map(override_map)
        mask = override_series.notna()
        accepted_df.loc[mask, final_decision_col] = override_series[mask].values
        reviewed_rows_df = accepted_df.loc[mask].copy()
    else:
        reviewed_rows_df = pd.DataFrame()

    return accepted_df, reviewed_rows_df
