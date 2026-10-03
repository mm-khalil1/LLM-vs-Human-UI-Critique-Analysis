from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from IPython.display import display
from utils.batch_manager import (
    extract_batch_record_id,
    extract_message_content_from_batch_record,
)

DEFAULT_REVIEW_MODEL = "gpt-5-mini"
DEFAULT_MATCH_REVIEW_SCHEMA_NAME = "human_llm_comment_match_judgment"
DEFAULT_MATCH_REVIEW_SYSTEM_PROMPT = (
    "You are an expert evaluator of UI/UX critique alignment. "
    "Return valid JSON only. No markdown, comments, or extra text."
)
DEFAULT_DUPLICATE_REVIEW_SCHEMA_NAME = "semantic_comment_relationship_judgment"
DEFAULT_DUPLICATE_REVIEW_SYSTEM_PROMPT = (
    "You judge whether two UI critique comments about the same task describe the same atomic UI issue. "
    "Use the screenshot only to resolve UI context, not to invent issues that are not in the comments. "
    "Return valid JSON only."
)
DEFAULT_MATCH_RELATION_OPTIONS = [
    "equivalent",
    "A_broader",
    "B_broader",
    "partial_overlap",
    "related_but_different",
    "no_match",
    "contradiction",
]
DEFAULT_MATCH_TERNARY_OPTIONS = ["yes", "no", "partial"]
MATCH_REVIEW_DIAGNOSTIC_COLUMNS = [
    "same_target_element",
    "same_usability_problem",
    "comment_a_depends_on_unseen_interaction",
    "comment_b_depends_on_unseen_interaction",
    "logical_conflict",
]
DEFAULT_DUPLICATE_RELATION_OPTIONS = [
    "duplicate",
    "a_contains_b",
    "b_contains_a",
    "partial_overlap",
    "different",
]
DEFAULT_DUPLICATE_ACTION_OPTIONS = [
    "keep_A",
    "keep_B",
    "keep_both",
    "manual_review",
]
DEFAULT_LEFT_COMMENT_COL = "left_observed_issue"
DEFAULT_RIGHT_COMMENT_COL = "right_observed_issue"


def iter_jsonl_paths(path_or_paths: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_paths, (str, Path)):
        path = Path(path_or_paths)
        if path.is_dir():
            return sorted(path.glob("*.jsonl"), key=lambda item: item.name)
        return [path]
    return [Path(path) for path in path_or_paths]


def latest_jsonl_records(
    path_or_paths: Path | str | Iterable[Path | str],
) -> tuple[list[tuple[Path, int, dict]], list[tuple[Path, int, json.JSONDecodeError]]]:
    """Read batch response JSONL files in order, keeping the last record for each custom_id.

    List retry response files after the original batch files so retried records replace the
    failed ones; each record keeps the position of its first occurrence. Returns
    (records, unparsable) as (path, line_number, record_or_error) tuples.
    """
    latest = {}
    unparsable = []
    for path in iter_jsonl_paths(path_or_paths):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    unparsable.append((path, line_number, exc))
                    continue
                custom_id = extract_batch_record_id(record)
                latest[(path.name, line_number) if custom_id is None else custom_id] = (path, line_number, record)
    return list(latest.values()), unparsable


def _prompt_text(value: str | None) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _prompt_bbox(value: object) -> str:
    if value is None:
        return "unavailable"
    try:
        if pd.isna(value):
            return "unavailable"
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value if value.strip() else "unavailable"
    if isinstance(value, dict):
        values = [value.get(key) for key in ("x_min", "y_min", "x_max", "y_max")]
    else:
        try:
            values = list(value)
        except TypeError:
            return str(value)
    if len(values) != 4:
        return str(value)
    try:
        return "[" + ", ".join(f"{float(coord):.3f}" for coord in values) + "]"
    except (TypeError, ValueError):
        return str(value)


def build_match_review_json_schema(
    *,
    schema_name: str = DEFAULT_MATCH_REVIEW_SCHEMA_NAME,
    relation_options: Optional[Iterable[str]] = None,
) -> dict:
    relation_values = list(relation_options or DEFAULT_MATCH_RELATION_OPTIONS)

    return {
        "name": schema_name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "same_target_element": {
                    "type": "string",
                    "enum": DEFAULT_MATCH_TERNARY_OPTIONS,
                },
                "same_usability_problem": {
                    "type": "string",
                    "enum": DEFAULT_MATCH_TERNARY_OPTIONS,
                },
                "comment_a_depends_on_unseen_interaction": {
                    "type": "string",
                    "enum": DEFAULT_MATCH_TERNARY_OPTIONS,
                },
                "comment_b_depends_on_unseen_interaction": {
                    "type": "string",
                    "enum": DEFAULT_MATCH_TERNARY_OPTIONS,
                },
                "logical_conflict": {
                    "type": "string",
                    "enum": DEFAULT_MATCH_TERNARY_OPTIONS,
                },
                "relation": {
                    "type": "string",
                    "enum": relation_values,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "same_target_element",
                "same_usability_problem",
                "comment_a_depends_on_unseen_interaction",
                "comment_b_depends_on_unseen_interaction",
                "logical_conflict",
                "relation",
                "confidence",
            ],
            "additionalProperties": False,
        },
    }

def build_duplicate_review_json_schema(
    *,
    schema_name: str = DEFAULT_DUPLICATE_REVIEW_SCHEMA_NAME,
    relation_options: Optional[Iterable[str]] = None,
    action_options: Optional[Iterable[str]] = None,
) -> dict:
    relation_values = list(relation_options or DEFAULT_DUPLICATE_RELATION_OPTIONS)
    action_values = list(action_options or DEFAULT_DUPLICATE_ACTION_OPTIONS)

    return {
        "name": schema_name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relationship": {"type": "string", "enum": relation_values},
                "reasoning": {"type": "string"},
                "recommended_action": {"type": "string", "enum": action_values},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["relationship", "reasoning", "recommended_action", "confidence"],
            "additionalProperties": False,
        },
    }

def build_human_duplicate_review_prompt(
    *,
    task: str | None,
    comment_a: str | None,
    comment_b: str | None,
    bbox_a: object | None = None,
    bbox_b: object | None = None,
    bbox_iou: float | None = None,
) -> str:
    task_text = _prompt_text(task)
    comment_a_text = _prompt_text(comment_a)
    comment_b_text = _prompt_text(comment_b)
    bbox_a_text = _prompt_bbox(bbox_a)
    bbox_b_text = _prompt_bbox(bbox_b)
    bbox_iou_text = "unavailable" if bbox_iou is None or pd.isna(bbox_iou) else f"{float(bbox_iou):.3f}"

    return "\n".join(
        [
            "Decide whether these two comments describe the same atomic UI issue for the same task. Check the bounding boxes and IoU to see if they refer to the same visual target, but judge the underlying issue, not just wording or element overlap.",
            "The bounding boxes are in [x_min, y_min, x_max, y_max] format, normalized to the screen size. IoU is the intersection-over-union of the two bounding boxes.",
            "Use the attached screenshot as task context when wording is ambiguous.",
            "",
            f"Task: {task_text}",
            f"Comment A: {comment_a_text}",
            f"Comment A bounding box: {bbox_a_text}",
            f"Comment B: {comment_b_text}",
            f"Comment B bounding box: {bbox_b_text}",
            f"BBox IoU between annotated regions: {bbox_iou_text}",
            "",
            "Assign one of these relationship labels:",
            "- duplicate: same issue at the same scope; wording differs only.",
            "- a_contains_b: Comment A fully includes Comment B's issue and adds useful detail, without adding a separate issue.",
            "- b_contains_a: Comment B fully includes Comment A's issue and adds useful detail, without adding a separate issue.",
            "- partial_overlap: the comments share some issue content, but each has distinct issue content or neither fully contains the other.",
            "- different: the comments concern different issues.",
            "",
            "When one comment has multiple atomic issues and the other matches only one of them, use partial_overlap unless the unmatched content is only explanation, consequence, or specificity for the same issue.",
            "Provide a brief reasoning for your choice.",
            "Use one recommended_action: keep_A, keep_B, keep_both, manual_review.",
            "Choose keep_A or keep_B only for duplicate, a_contains_b, or b_contains_a. Keep the more complete, specific, and actionable comment.",
            "Use keep_both for partial_overlap or different. Use manual_review when the better action is ambiguous.",
            'Return JSON exactly as: {"reasoning": string, "relationship": string, "recommended_action": string, "confidence": number}',
        ]
    )

def build_human_llm_match_user_prompt(
    row: pd.Series,
    *,
    human_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    llm_comments: list[dict],
    task_col: Optional[str] = "task",
    human_bbox_col: Optional[str] = None,
) -> str:
    if len(llm_comments) != 1:
        raise ValueError("Pairwise match review prompts require exactly one Comment B.")
    comment_b = llm_comments[0]
    bbox_text = _prompt_bbox(row.get(human_bbox_col)) if human_bbox_col else "unavailable"
    lines = [
        "Determine the relationship between Comment A and Comment B for the same mobile screen and same task, using the attached screen image as grounding evidence.",
        "",
        "Judge the underlying UI/UX issue, not wording overlap. Two comments can match even if phrased "
        "differently. Do not treat two comments as a match or overlap just because they mention the same interface "
        "element; they must point to the same actual problem, or one must clearly contain the other’s problem content.",
        "",
        "Comment A is talking about the UI region identified by its bounding box. Use that box to ground the target element for Comment A, but judge the underlying usability problem described by both comments.",
    ]

    lines.extend(
        [
            "",
            "Allowed labels:",
            "- equivalent: both comments express essentially the same underlying issue at similar scope.",
            "- A_broader: Comment A clearly includes Comment B's issue, but adds more information, broader concerns, or an additional issue.",
            "- B_broader: Comment B clearly includes Comment A's issue, but adds more information, broader concerns, or an additional issue.",
            "- partial_overlap: both comments clearly discuss an issue, but each also includes distinct other aspects or issues, and neither clearly contains the other.",
            "- related_but_different: both comments concern a similar area, theme, or interface element, but not the same actual issue.",
            "- contradiction: the comments make mutually incompatible claims about the same UI/UX issue or element in the screen (e.g., one affirms an issue while the other denies it, or they provide opposing diagnoses, causes, or evaluations that cannot both be true).",
            "- no_match: the comments describe different issues.",
            "",
            "Do not assign A_broader or B_broader unless one comment clearly subsumes the other’s actual problem, not merely the same UI element or area.",
            "Do not assign partial_overlap when the comments only share a broad UX theme or element.",
            "",
            'For intermediate steps (same_target_element, same_usability_problem, comment_a_depends_on_unseen_interaction, comment_b_depends_on_unseen_interaction, and logical_conflict), use exactly one of: "yes", "no", "partial".',
            "Use depends_on_unseen_interaction=yes when deciding the comment requires observing an interaction, transition, dynamic state, or hidden content not visible in the screenshot. Use partial when only part of the comment depends on unseen interaction.",
        ]
    )
    if task_col and task_col in row.index:
        lines.extend(["", "Task:", _prompt_text(row[task_col])])

    lines.extend(
        [
            "",
            "Comment A bounding box in [x_min, y_min, x_max, y_max] format:",
            bbox_text,
            "",
            "Comment A:",
            _prompt_text(row[human_comment_col]),
            "",
            "Comment B:",
            _prompt_text(comment_b["observed_issue"]),
        ]
    )

    lines.extend(
        [
            "",
            "Return exactly the following JSON schema. You must evaluate the intermediate steps before generating the final relation.",
            '{',
            '  "same_target_element": "yes" | "no" | "partial",',
            '  "same_usability_problem": "yes" | "no" | "partial",',
            '  "comment_a_depends_on_unseen_interaction": "yes" | "no" | "partial",',
            '  "comment_b_depends_on_unseen_interaction": "yes" | "no" | "partial",',
            '  "logical_conflict": "yes" | "no" | "partial",',
            '  "relation": "<one allowed label>",',
            '  "confidence": a number between 0 and 1',
            '}',

        ]
    )
    return "\n".join(lines)

def _build_llm_comment_list_for_match_prompt(
    group_df: pd.DataFrame,
    *,
    llm_id_col: str,
    llm_comment_col: str,
) -> list[dict]:
    llm_df = group_df[[llm_id_col, llm_comment_col]].drop_duplicates(subset=[llm_id_col], keep="first")
    return [
        {
            "llm_comment_id": str(row[llm_id_col]),
            "observed_issue": _prompt_text(row[llm_comment_col]),
        }
        for _, row in llm_df.iterrows()
    ]

def build_multimodal_user_content(
    user_prompt: str,
    *,
    base64_screen: Optional[str] = None,
    image_format: str = "jpeg",
) -> str | list[dict]:
    if not base64_screen:
        return user_prompt
    return [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_format};base64,{base64_screen}"},
        },
    ]

def _build_gemini_parts(
    user_prompt: str,
    *,
    base64_screen: Optional[str] = None,
    image_format: str = "jpeg",
) -> list[dict]:
    parts = [{"text": user_prompt}]
    if base64_screen:
        parts.append(
            {
                "inlineData": {
                    "mimeType": f"image/{image_format}",
                    "data": base64_screen,
                },
            }
        )
    return parts

def _openai_response_schema(response_format: Optional[dict]) -> Optional[dict]:
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        return (response_format.get("json_schema") or {}).get("schema")
    return None

def build_chat_completion_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base64_screen: Optional[str] = None,
    image_format: str = "jpeg",
    response_format: Optional[dict] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    url: str = "/v1/chat/completions",
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_multimodal_user_content(
                    user_prompt,
                    base64_screen=base64_screen,
                    image_format=image_format,
                ),
            },
        ],
    }
    if temperature is not None:
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format
    if max_tokens is not None:
        body["max_completion_tokens"] = max_tokens

    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": url,
        "body": body,
    }

def build_gemini_batch_request(
    *,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    base64_screen: Optional[str] = None,
    image_format: str = "jpeg",
    response_format: Optional[dict] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict:
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens

    response_schema = _openai_response_schema(response_format)
    if response_schema is not None:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseJsonSchema"] = response_schema

    return {
        "key": str(custom_id),
        "request": {
            "model": f"models/{model}",
            "contents": [
                {
                    "role": "user",
                    "parts": _build_gemini_parts(
                        user_prompt,
                        base64_screen=base64_screen,
                        image_format=image_format,
                    ),
                }
            ],
            "systemInstruction": {
                "role": "user",
                "parts": [{"text": system_prompt}],
            },
            "generation_config": generation_config,
        },
    }

def build_human_duplicate_review_request(
    row: pd.Series,
    *,
    custom_id: str,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    left_bbox_col: Optional[str] = "left_bounding_box",
    right_bbox_col: Optional[str] = "right_bounding_box",
    bbox_iou_col: Optional[str] = "bbox_iou",
    task_col: Optional[str] = "task",
    base64_screen_col: Optional[str] = None,
    image_format: str = "jpeg",
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_DUPLICATE_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_DUPLICATE_REVIEW_SCHEMA_NAME,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    url: str = "/v1/chat/completions",
) -> dict:
    user_prompt = build_human_duplicate_review_prompt(
        task=row.get(task_col, "") if task_col else "",
        comment_a=row[left_comment_col],
        comment_b=row[right_comment_col],
        bbox_a=row.get(left_bbox_col) if left_bbox_col else None,
        bbox_b=row.get(right_bbox_col) if right_bbox_col else None,
        bbox_iou=row.get(bbox_iou_col) if bbox_iou_col else None,
    )
    base64_screen = None
    if base64_screen_col and base64_screen_col in row.index and pd.notna(row[base64_screen_col]):
        base64_screen = row[base64_screen_col]

    response_format = {
        "type": "json_schema",
        "json_schema": build_duplicate_review_json_schema(schema_name=schema_name),
    }
    provider = "gemini" if provider == "google" else provider
    if provider == "gemini":
        return build_gemini_batch_request(
            custom_id=custom_id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64_screen=base64_screen,
            image_format=image_format,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider != "openai":
        raise ValueError("provider must be either 'openai' or 'gemini'.")

    return build_chat_completion_request(
        custom_id=custom_id,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        base64_screen=base64_screen,
        image_format=image_format,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        url=url,
    )

def iter_human_duplicate_review_requests(
    candidate_df: pd.DataFrame,
    *,
    custom_id_col: Optional[str] = "custom_id",
    left_id_col: str = "left_comment_id",
    right_id_col: str = "right_comment_id",
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    left_bbox_col: Optional[str] = "left_bounding_box",
    right_bbox_col: Optional[str] = "right_bounding_box",
    bbox_iou_col: Optional[str] = "bbox_iou",
    task_col: Optional[str] = "task",
    screen_id_col: str = "screen_id",
    base64_screen_col: Optional[str] = None,
    base64_screen_lookup: Optional[dict] = None,
    image_format: str = "jpeg",
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_DUPLICATE_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_DUPLICATE_REVIEW_SCHEMA_NAME,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    url: str = "/v1/chat/completions",
):
    for _, row in candidate_df.iterrows():
        request_row = row
        if base64_screen_col and base64_screen_lookup is not None:
            request_row = row.copy()
            request_row[base64_screen_col] = base64_screen_lookup.get(row[screen_id_col], pd.NA)

        if custom_id_col:
            custom_id = str(request_row[custom_id_col])
        else:
            custom_id = f"semantic_pair::{request_row[left_id_col]}::{request_row[right_id_col]}"

        request = build_human_duplicate_review_request(
            request_row,
            custom_id=custom_id,
            left_comment_col=left_comment_col,
            right_comment_col=right_comment_col,
            left_bbox_col=left_bbox_col,
            right_bbox_col=right_bbox_col,
            bbox_iou_col=bbox_iou_col,
            task_col=task_col,
            base64_screen_col=base64_screen_col,
            image_format=image_format,
            model=model,
            system_prompt=system_prompt,
            schema_name=schema_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            url=url,
        )
        yield request

def build_human_llm_match_review_request(
    row: pd.Series,
    *,
    custom_id: str,
    human_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    llm_comments: list[dict],
    task_col: Optional[str] = "task",
    human_bbox_col: Optional[str] = None,
    base64_screen_col: Optional[str] = None,
    image_format: str = "jpeg",
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_MATCH_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_MATCH_REVIEW_SCHEMA_NAME,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    url: str = "/v1/chat/completions",
) -> dict:
    user_prompt = build_human_llm_match_user_prompt(
        row,
        human_comment_col=human_comment_col,
        llm_comments=llm_comments,
        task_col=task_col,
        human_bbox_col=human_bbox_col,
    )
    base64_screen = None
    if base64_screen_col and base64_screen_col in row.index and pd.notna(row[base64_screen_col]):
        base64_screen = row[base64_screen_col]

    response_format = {
        "type": "json_schema",
        "json_schema": build_match_review_json_schema(schema_name=schema_name),
    }
    provider = "gemini" if provider == "google" else provider
    if provider == "gemini":
        return build_gemini_batch_request(
            custom_id=custom_id,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base64_screen=base64_screen,
            image_format=image_format,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider != "openai":
        raise ValueError("provider must be either 'openai' or 'gemini'.")

    return build_chat_completion_request(
        custom_id=custom_id,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        base64_screen=base64_screen,
        image_format=image_format,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
        url=url,
    )

def iter_human_llm_match_review_requests(
    candidate_df: pd.DataFrame,
    *,
    custom_id_col: Optional[str] = None,
    custom_id_prefix: str = "",
    human_id_col: str,
    llm_id_col: str,
    screen_task_id_col: str = "screen_task_id",
    screen_id_col: str,
    human_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    llm_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    task_col: Optional[str] = "task",
    human_bbox_col: Optional[str] = None,
    base64_screen_col: Optional[str] = None,
    base64_screen_lookup: Optional[dict] = None,
    image_format: str = "jpeg",
    model: str = DEFAULT_REVIEW_MODEL,
    system_prompt: str = DEFAULT_MATCH_REVIEW_SYSTEM_PROMPT,
    schema_name: str = DEFAULT_MATCH_REVIEW_SCHEMA_NAME,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    provider: str = "openai",
    url: str = "/v1/chat/completions",
    pairwise: bool = False,
):
    if human_id_col not in candidate_df.columns:
        raise ValueError(f"Missing human id column: {human_id_col}")
    if llm_id_col not in candidate_df.columns:
        raise ValueError(f"Missing LLM id column: {llm_id_col}")
    if llm_comment_col not in candidate_df.columns:
        raise ValueError(f"Missing LLM comment column: {llm_comment_col}")
    if not custom_id_col and screen_task_id_col not in candidate_df.columns:
        raise ValueError(f"Missing screen task column: {screen_task_id_col}")

    if pairwise:
        row_iter = (
            (
                row,
                [
                    {
                        "llm_comment_id": str(row[llm_id_col]),
                        "observed_issue": _prompt_text(row[llm_comment_col]),
                    }
                ],
            )
            for _, row in candidate_df.iterrows()
        )
    else:
        row_iter = (
            (
                group_df.iloc[0],
                _build_llm_comment_list_for_match_prompt(
                    group_df,
                    llm_id_col=llm_id_col,
                    llm_comment_col=llm_comment_col,
                ),
            )
            for _, group_df in candidate_df.groupby(human_id_col, sort=False, dropna=False)
        )

    for row, llm_comments in row_iter:
        request_row = row
        if base64_screen_col and base64_screen_lookup is not None:
            request_row = row.copy()
            request_row[base64_screen_col] = base64_screen_lookup.get(row[screen_id_col], pd.NA)

        if custom_id_col:
            custom_id = str(request_row[custom_id_col])
        elif pairwise and "pair_id" in request_row.index:
            custom_id = "::".join(
                [
                    str(request_row["pair_id"]),
                    str(request_row[human_id_col]),
                    str(request_row[llm_id_col]),
                ]
            )
        else:
            custom_id_parts = [str(request_row[screen_task_id_col]), str(request_row[human_id_col])]
            if pairwise:
                custom_id_parts.append(str(request_row[llm_id_col]))
            if custom_id_prefix:
                custom_id_parts.insert(0, str(custom_id_prefix))
            custom_id = "::".join(custom_id_parts)

        yield build_human_llm_match_review_request(
            request_row,
            custom_id=custom_id,
            human_comment_col=human_comment_col,
            llm_comments=llm_comments,
            task_col=task_col,
            human_bbox_col=human_bbox_col,
            base64_screen_col=base64_screen_col,
            image_format=image_format,
            model=model,
            system_prompt=system_prompt,
            schema_name=schema_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            url=url,
        )

def write_human_llm_match_review_jsonl_parts(
    candidate_df: pd.DataFrame,
    output_dir: Path | str,
    *,
    output_stem: str = "requests-semantic_match",
    split_mode: str = "requests",
    max_requests_per_file: Optional[int] = None,
    max_screen_tasks_per_file: Optional[int] = None,
    screen_task_id_col: str = "screen_task_id",
    **request_kwargs,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if split_mode not in {"requests", "screen_tasks"}:
        raise ValueError("split_mode must be either 'requests' or 'screen_tasks'.")

    manifest_rows = []

    if split_mode == "requests":
        if max_requests_per_file is None or max_requests_per_file <= 0:
            raise ValueError("max_requests_per_file must be positive when split_mode='requests'.")

        part_index = 1
        current_path = output_dir / f"{output_stem}-part_{part_index:03d}.jsonl"
        current_file = current_path.open("w", encoding="utf-8")
        current_requests = 0

        def flush_current_part() -> None:
            nonlocal part_index, current_path, current_file, current_requests
            if current_file.closed:
                return
            current_file.close()
            if current_requests == 0:
                current_path.unlink(missing_ok=True)
                return
            manifest_rows.append(
                {
                    "part_index": part_index,
                    "path": current_path,
                    "requests": current_requests,
                    "screen_tasks": pd.NA,
                }
            )
            part_index += 1
            current_path = output_dir / f"{output_stem}-part_{part_index:03d}.jsonl"
            current_file = current_path.open("w", encoding="utf-8")
            current_requests = 0

        try:
            for request in iter_human_llm_match_review_requests(candidate_df, **request_kwargs):
                current_file.write(json.dumps(request, ensure_ascii=False) + "\n")
                current_requests += 1
                if current_requests >= max_requests_per_file:
                    flush_current_part()
        finally:
            flush_current_part()

    if split_mode == "screen_tasks":
        if max_screen_tasks_per_file is None or max_screen_tasks_per_file <= 0:
            raise ValueError("max_screen_tasks_per_file must be positive when split_mode='screen_tasks'.")
        if screen_task_id_col not in candidate_df.columns:
            raise ValueError(f"Missing screen task column: {screen_task_id_col}")

        screen_task_ids = candidate_df[screen_task_id_col].drop_duplicates().tolist()
        for part_index, start in enumerate(range(0, len(screen_task_ids), max_screen_tasks_per_file), start=1):
            part_screen_task_ids = set(screen_task_ids[start : start + max_screen_tasks_per_file])
            part_df = candidate_df[candidate_df[screen_task_id_col].isin(part_screen_task_ids)]
            current_path = output_dir / f"{output_stem}-part_{part_index:03d}.jsonl"
            request_count = 0
            with current_path.open("w", encoding="utf-8") as current_file:
                for request in iter_human_llm_match_review_requests(part_df, **request_kwargs):
                    current_file.write(json.dumps(request, ensure_ascii=False) + "\n")
                    request_count += 1
            manifest_rows.append(
                {
                    "part_index": part_index,
                    "path": current_path,
                    "requests": request_count,
                    "screen_tasks": len(part_screen_task_ids),
                }
            )

    return pd.DataFrame(manifest_rows)

def parse_review_message_content(message_content: str) -> dict:
    if not message_content:
        return {}
    return json.loads(message_content)

def parse_review_custom_id(custom_id: object) -> tuple[Optional[str], Optional[str]]:
    parts = str(custom_id or "").split("::")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    return None, None

def audit_llm_review_results_jsonl(
    results_path: Path | str | Iterable[Path | str],
    *,
    expected_custom_ids: Optional[Iterable[object]] = None,
    relation_options: Optional[Iterable[str]] = None,
    action_options: Optional[Iterable[str]] = None,
) -> dict:
    expected_custom_ids = [] if expected_custom_ids is None else expected_custom_ids
    expected_custom_ids = {str(value) for value in expected_custom_ids}
    seen_custom_ids = set()
    relation_options = set(relation_options or DEFAULT_DUPLICATE_RELATION_OPTIONS)
    action_options = set(action_options or DEFAULT_DUPLICATE_ACTION_OPTIONS)
    invalid_examples = []
    invalid_record_keys = set()
    counts = {
        "records": 0,
        "http_error_records": 0,
        "provider_error_records": 0,
        "missing_content": 0,
        "json_parse_errors": 0,
        "missing_required_fields": 0,
        "invalid_relationship": 0,
        "invalid_recommended_action": 0,
    }

    records, unparsable = latest_jsonl_records(results_path)
    counts["records"] = len(records) + len(unparsable)
    for result_path, line_number, exc in unparsable:
        counts["json_parse_errors"] += 1
        invalid_record_keys.add((result_path.name, line_number))
        if len(invalid_examples) < 5:
            invalid_examples.append({"file": result_path.name, "line": line_number, "error": str(exc)})

    for result_path, line_number, record in records:
        custom_id = extract_batch_record_id(record)
        record_key = str(custom_id) if custom_id is not None else (result_path.name, line_number)
        if custom_id is not None:
            seen_custom_ids.add(str(custom_id))
        status_code = record.get("response", {}).get("status_code")
        if status_code is not None and status_code != 200:
            counts["http_error_records"] += 1
            invalid_record_keys.add(record_key)

        provider_error = record.get("error")
        if provider_error is not None:
            counts["provider_error_records"] += 1
            invalid_record_keys.add(record_key)
            if len(invalid_examples) < 5:
                invalid_examples.append(
                    {
                        "file": result_path.name,
                        "custom_id": custom_id,
                        "error": "provider error",
                        "provider_error": provider_error,
                    }
                )
            continue

        message_content = extract_message_content_from_batch_record(record)
        if not message_content:
            counts["missing_content"] += 1
            invalid_record_keys.add(record_key)
            if len(invalid_examples) < 5:
                invalid_examples.append({"file": result_path.name, "custom_id": custom_id, "error": "missing content"})
            continue

        try:
            parsed = parse_review_message_content(message_content)
        except json.JSONDecodeError as exc:
            counts["json_parse_errors"] += 1
            invalid_record_keys.add(record_key)
            if len(invalid_examples) < 5:
                invalid_examples.append({"file": result_path.name, "custom_id": custom_id, "error": str(exc)})
            continue

        required_missing = [
            field for field in ("relationship", "recommended_action", "confidence")
            if parsed.get(field) is None
        ]
        if required_missing:
            counts["missing_required_fields"] += 1
            invalid_record_keys.add(record_key)
            if len(invalid_examples) < 5:
                invalid_examples.append({"file": result_path.name, "custom_id": custom_id, "missing": required_missing})
        if parsed.get("relationship") is not None and parsed.get("relationship") not in relation_options:
            counts["invalid_relationship"] += 1
            invalid_record_keys.add(record_key)
        if parsed.get("recommended_action") is not None and parsed.get("recommended_action") not in action_options:
            counts["invalid_recommended_action"] += 1
            invalid_record_keys.add(record_key)

    missing_expected = sorted(expected_custom_ids - seen_custom_ids)
    invalid_custom_ids = sorted(key for key in invalid_record_keys if isinstance(key, str))
    return {
        **counts,
        "expected_records": len(expected_custom_ids) if expected_custom_ids else None,
        "missing_expected_records": len(missing_expected),
        "invalid_response_count": len(invalid_record_keys),
        "invalid_custom_ids": invalid_custom_ids,
        "invalid_examples": invalid_examples,
        "missing_expected_custom_ids": missing_expected,
        "missing_expected_examples": missing_expected[:5],
    }

def parse_llm_review_results_jsonl(
    results_path: Path | str | Iterable[Path | str],
    candidate_df: pd.DataFrame,
    *,
    left_id_col: str,
    right_id_col: str,
    include_candidate_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    include_candidate_columns = include_candidate_columns or []
    candidate_columns = [
        col
        for col in dict.fromkeys(
            [
                left_id_col,
                right_id_col,
                "pair_id",
                "custom_id",
                *include_candidate_columns,
            ]
        )
        if col in candidate_df.columns
    ]
    candidate_lookup_df = candidate_df[candidate_columns].copy()
    for col in (left_id_col, right_id_col):
        candidate_lookup_df[col] = candidate_lookup_df[col].astype(str)

    if "custom_id" in candidate_lookup_df.columns:
        candidate_lookup_df["_review_custom_id"] = candidate_lookup_df["custom_id"].astype(str)
    elif "pair_id" in candidate_lookup_df.columns:
        candidate_lookup_df["_review_custom_id"] = (
            candidate_lookup_df["pair_id"].astype(str)
            + "::"
            + candidate_lookup_df[left_id_col]
            + "::"
            + candidate_lookup_df[right_id_col]
        )
    else:
        candidate_lookup_df["_review_custom_id"] = pd.NA

    candidate_records = candidate_lookup_df.to_dict(orient="records")
    custom_id_lookup = {
        str(row["_review_custom_id"]): row
        for row in candidate_records
        if pd.notna(row.get("_review_custom_id"))
    }
    pair_lookup = {
        (row[left_id_col], row[right_id_col]): row
        for row in candidate_records
    }

    records, unparsable = latest_jsonl_records(results_path)
    if unparsable:
        raise unparsable[0][2]

    result_rows = []
    for _, _, record in records:
        custom_id = extract_batch_record_id(record)
        message_content = extract_message_content_from_batch_record(record)
        parsed = parse_review_message_content(message_content)

        left_id, right_id = parse_review_custom_id(custom_id)
        pair_row = custom_id_lookup.get(str(custom_id))
        if pair_row is None:
            pair_row = pair_lookup.get((str(left_id), str(right_id)))

        row_data = {
            "custom_id": custom_id,
            "left_id": left_id,
            "right_id": right_id,
            "relationship": parsed.get("relationship"),
            "reasoning": parsed.get("reasoning"),
            "recommended_action": parsed.get("recommended_action"),
            "relation": parsed.get("relation"),
            "confidence": parsed.get("confidence"),
        }
        for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS:
            row_data[col] = parsed.get(col)

        if pair_row is not None and include_candidate_columns:
            for col in include_candidate_columns:
                row_data[col] = pair_row.get(col)

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


def audit_match_review_result_frame(
    judge_df: pd.DataFrame,
    expected_pair_ids: Iterable[object],
    *,
    relation_options: Optional[Iterable[str]] = None,
) -> dict:
    """Summarize pair-level coverage and validity for parsed human-LLM match judgments."""
    expected_pair_ids = {str(value) for value in expected_pair_ids}
    required_cols = ["pair_id", "expert_comment_id", "llm_comment_id", "relation"]
    missing_cols = [col for col in required_cols if col not in judge_df.columns]
    if missing_cols:
        return {
            "parsed_rows": len(judge_df),
            "expected_pairs": len(expected_pair_ids),
            "missing_pair_ids": len(expected_pair_ids),
            "extra_pair_ids": 0,
            "duplicate_pair_ids": 0,
            "invalid_relations": 0,
            "missing_diagnostic_values": 0,
            "invalid_diagnostic_values": 0,
            "missing_identity_rows": len(judge_df),
            "missing_parse_columns": missing_cols,
            "missing_examples": sorted(expected_pair_ids)[:5],
            "extra_examples": [],
            "duplicate_examples": [],
            "invalid_relation_examples": [],
        }

    parsed_pair_ids = set(judge_df["pair_id"].dropna().astype(str))
    duplicate_pair_ids = (
        judge_df.loc[judge_df["pair_id"].notna() & judge_df["pair_id"].duplicated(), "pair_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    invalid_relations = sorted(
        set(judge_df["relation"].dropna()) - set(relation_options or DEFAULT_MATCH_RELATION_OPTIONS)
    )
    diagnostic_options = set(DEFAULT_MATCH_TERNARY_OPTIONS)
    missing_diagnostic_values = 0
    invalid_diagnostic_values = 0
    for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS:
        if col not in judge_df.columns:
            missing_diagnostic_values += len(judge_df)
            continue
        missing_diagnostic_values += int(judge_df[col].isna().sum())
        invalid_diagnostic_values += int((~judge_df[col].dropna().isin(diagnostic_options)).sum())
    missing_pair_ids = sorted(expected_pair_ids - parsed_pair_ids)
    extra_pair_ids = sorted(parsed_pair_ids - expected_pair_ids)
    missing_identity_rows = judge_df[["pair_id", "expert_comment_id", "llm_comment_id"]].isna().any(axis=1).sum()

    return {
        "parsed_rows": len(judge_df),
        "expected_pairs": len(expected_pair_ids),
        "missing_pair_ids": len(missing_pair_ids),
        "extra_pair_ids": len(extra_pair_ids),
        "duplicate_pair_ids": len(duplicate_pair_ids),
        "invalid_relations": len(invalid_relations),
        "missing_diagnostic_values": int(missing_diagnostic_values),
        "invalid_diagnostic_values": int(invalid_diagnostic_values),
        "missing_identity_rows": int(missing_identity_rows),
        "missing_parse_columns": [],
        "missing_examples": missing_pair_ids[:5],
        "extra_examples": extra_pair_ids[:5],
        "duplicate_examples": duplicate_pair_ids[:5],
        "invalid_relation_examples": invalid_relations[:5],
    }


def match_review_audit_has_issues(audit_df: pd.DataFrame) -> pd.Series:
    issue_cols = [
        "missing_pair_ids",
        "extra_pair_ids",
        "duplicate_pair_ids",
        "invalid_relations",
        "missing_diagnostic_values",
        "invalid_diagnostic_values",
        "missing_identity_rows",
    ]
    missing_parse_col = audit_df.get("missing_parse_columns", pd.Series([[]] * len(audit_df), index=audit_df.index))
    return audit_df[issue_cols].sum(axis=1).gt(0) | missing_parse_col.map(len).gt(0)


def compare_match_review_judges(
    all_results_df: pd.DataFrame,
    *,
    left_judge: str,
    right_judge: str,
) -> pd.DataFrame:
    index_cols = ["pair_id", "custom_id", "expert_comment_id", "llm_comment_id"]
    context_cols = ["screen_task_id", "task", "expert_observed_issue", "llm_observed_issue"]
    compare_cols = index_cols + ["relation", "confidence", *MATCH_REVIEW_DIAGNOSTIC_COLUMNS]
    left_cols = compare_cols + [col for col in context_cols if col in all_results_df.columns]

    left_df = all_results_df[all_results_df["judge_model_short"].eq(left_judge)][
        left_cols
    ].rename(columns={
        "relation": f"relation__{left_judge}",
        "confidence": f"confidence__{left_judge}",
        **{col: f"{col}__{left_judge}" for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS},
    })
    right_df = all_results_df[all_results_df["judge_model_short"].eq(right_judge)][compare_cols].rename(
        columns={
            "relation": f"relation__{right_judge}",
            "confidence": f"confidence__{right_judge}",
            **{col: f"{col}__{right_judge}" for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS},
        }
    )

    comparison_df = left_df.merge(right_df, on=index_cols, how="outer", indicator=True)
    comparison_df["same_relation"] = (
        comparison_df[f"relation__{left_judge}"] == comparison_df[f"relation__{right_judge}"]
    )
    for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS:
        comparison_df[f"same_{col}"] = (
            comparison_df[f"{col}__{left_judge}"] == comparison_df[f"{col}__{right_judge}"]
        )
    return comparison_df


def finalize_match_review_judges(
    comparison_df: pd.DataFrame,
    *,
    left_judge: str,
    right_judge: str,
    manual_relation_overrides: Optional[dict[str, str]] = None,
    manual_confidence_overrides: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    manual_relation_overrides = manual_relation_overrides or {}
    manual_confidence_overrides = manual_confidence_overrides or {}

    for pair_id, relation in manual_relation_overrides.items():
        if relation not in DEFAULT_MATCH_RELATION_OPTIONS:
            raise ValueError(f"Invalid manual relation for {pair_id}: {relation}")
    for pair_id, confidence in manual_confidence_overrides.items():
        if pair_id not in manual_relation_overrides:
            raise ValueError(f"Manual confidence supplied without manual relation for {pair_id}.")
        if not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid manual confidence for {pair_id}: {confidence}")
    left_relation_col = f"relation__{left_judge}"
    right_relation_col = f"relation__{right_judge}"
    left_confidence_col = f"confidence__{left_judge}"
    right_confidence_col = f"confidence__{right_judge}"

    final_df = comparison_df.copy()
    agreed = final_df["same_relation"] & final_df["_merge"].eq("both")

    final_df["manual_relation"] = final_df["pair_id"].map(manual_relation_overrides)
    final_df["manual_confidence"] = final_df["pair_id"].map(manual_confidence_overrides)
    final_df["final_relation"] = np.select(
        [agreed, final_df["manual_relation"].notna()],
        [final_df[left_relation_col], final_df["manual_relation"]],
        default=pd.NA,
    )
    final_df["final_confidence"] = np.select(
        [agreed, final_df["manual_relation"].notna()],
        [
            final_df[[left_confidence_col, right_confidence_col]].mean(axis=1),
            final_df["manual_confidence"],
        ],
        default=np.nan,
    )
    final_df["decision_source"] = np.select(
        [agreed, final_df["manual_relation"].notna()],
        ["judge_agreement", "manual_override"],
        default="manual_review",
    )
    for col in MATCH_REVIEW_DIAGNOSTIC_COLUMNS:
        left_col = f"{col}__{left_judge}"
        right_col = f"{col}__{right_judge}"
        same_col = f"same_{col}"
        final_df[col] = np.where(
            final_df[same_col] & final_df["_merge"].eq("both"),
            final_df[left_col],
            pd.NA,
        )
    final_df["relation"] = final_df["final_relation"]
    final_df["confidence"] = final_df["final_confidence"]
    return final_df

def inspect_manual_review_rows(
    review_needed_df: pd.DataFrame,
    *,
    start: int = 0,
    left_comment_col: str = DEFAULT_LEFT_COMMENT_COL,
    right_comment_col: str = DEFAULT_RIGHT_COMMENT_COL,
    left_bbox_col: str = "left_bounding_box",
    right_bbox_col: str = "right_bounding_box",
    screen_id_col: str = "screen_id",
    base64_screen_col: str = "base64_screen",
    task_col: str = "screen_task_id",
    task_name_col: str = "task",
    relationship_col: str = "relationship",
    action_col: str = "recommended_action",
    confidence_col: str = "confidence",
    iou_col: str = "bbox_iou",
    display_screen_image: bool = True,
    image_dir: Optional[str | Path] = None,
    screen_width: int = 300,
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

        def suffixed_cols(base_col: str) -> list[str]:
            prefix = f"{base_col}__"
            return [col for col in row.index if str(col).startswith(prefix)]

        def print_base_or_suffixed(base_col: str, label: Optional[str] = None) -> bool:
            cols = suffixed_cols(base_col)
            if cols:
                for col in cols:
                    col_label = str(col)
                    if label is not None:
                        suffix = col_label[len(base_col):]
                        col_label = f"{label}{suffix}"
                    print(f"{col_label:<40}:", row.get(col))
                return True
            elif base_col in row.index:
                print(f"{(label or base_col):<40}:", row.get(base_col))
                return True
            return False

        def has_value(value: object) -> bool:
            if value is None:
                return False
            try:
                is_missing = pd.isna(value)
            except (TypeError, ValueError):
                return True
            if isinstance(is_missing, (bool, np.bool_)):
                return not bool(is_missing)
            return True

        def format_display_value(value: object, *, decimals: Optional[int] = None) -> object:
            if decimals is None or not has_value(value):
                return value
            try:
                return f"{float(value):.{decimals}f}"
            except (TypeError, ValueError):
                return value

        def display_current_screen() -> None:
            if not display_screen_image:
                return

            screen_id = row.get(screen_id_col) if screen_id_col in row.index else None
            base64_screen = row.get(base64_screen_col) if base64_screen_col in row.index else None
            bboxes = []
            labels = []
            for bbox_col, label in ((left_bbox_col, "A"), (right_bbox_col, "B")):
                bbox = row.get(bbox_col) if bbox_col in row.index else None
                if has_value(bbox):
                    bboxes.append(bbox)
                    labels.append(label)

            if not has_value(screen_id) and not has_value(base64_screen):
                return

            try:
                from utils.screen_display import show_screen

                show_screen(
                    screen_id=screen_id if has_value(screen_id) else None,
                    base64_screen=str(base64_screen) if has_value(base64_screen) else None,
                    image_dir=image_dir,
                    bboxes=bboxes,
                    labels=labels,
                    width=screen_width,
                )
            except Exception as exc:
                print(f"screen display unavailable: {exc}")

        print("\n" + "=" * 110)
        print(f"{i+1}/{n} | row_index={row_idx}")
        print("-" * 110)
        if task_col in row.index:
            print("screen_task_id      :", row.get(task_col))
            if task_name_col in row.index:
                print("task                :", row.get(task_name_col))
        print_base_or_suffixed(relationship_col)
        print_base_or_suffixed(action_col)
        print_base_or_suffixed(confidence_col, label="judge_confidence")
        printed_iou = False
        iou_cols = suffixed_cols(iou_col)
        if iou_cols:
            for col in iou_cols:
                print(f"{str(col):<40}:", format_display_value(row.get(col), decimals=3))
            printed_iou = True
        elif iou_col in row.index:
            print(f"{'iou':<40}:", format_display_value(row.get(iou_col), decimals=3))
            printed_iou = True
        if not printed_iou and iou_col != "iou" and "iou" in row.index:
            print(f"{'iou':<40}:", format_display_value(row.get("iou"), decimals=3))
        display_current_screen()
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

def audit_pairwise_merge_decisions(
    decisions_df: pd.DataFrame,
    *,
    final_decision_col: str = "final_decision",
    left_id_col: str = "left_id",
    right_id_col: str = "right_id",
    keep_left_value: str = "keep_A",
    keep_right_value: str = "keep_B",
) -> dict[str, object]:
    merge_df = decisions_df[decisions_df[final_decision_col].isin([keep_left_value, keep_right_value])].copy()
    if merge_df.empty:
        empty_components = pd.DataFrame(
            columns=[
                "component_id",
                "component_size",
                "comment_ids",
                "drop_ids",
                "survivor_ids",
                "num_survivors",
                "has_conflict",
            ]
        )
        return {
            "merge_decisions_df": merge_df,
            "drop_ids": set(),
            "conflicting_drop_ids": [],
            "conflict_rows_df": merge_df,
            "components_df": empty_components,
            "problem_components_df": empty_components.copy(),
        }

    merge_df["kept_id"] = np.where(
        merge_df[final_decision_col].eq(keep_left_value),
        merge_df[left_id_col],
        merge_df[right_id_col],
    )
    merge_df["dropped_id"] = np.where(
        merge_df[final_decision_col].eq(keep_left_value),
        merge_df[right_id_col],
        merge_df[left_id_col],
    )

    kept_ids = set(merge_df["kept_id"].dropna().astype(str))
    drop_ids = set(merge_df["dropped_id"].dropna().astype(str))
    conflicting_drop_ids = sorted(kept_ids & drop_ids)

    parent: dict[str, str] = {}

    def find_parent(comment_id: str) -> str:
        parent.setdefault(comment_id, comment_id)
        while parent[comment_id] != comment_id:
            parent[comment_id] = parent[parent[comment_id]]
            comment_id = parent[comment_id]
        return comment_id

    def union_ids(left_id: str, right_id: str) -> None:
        left_root = find_parent(left_id)
        right_root = find_parent(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in merge_df[[left_id_col, right_id_col]].dropna().astype(str).itertuples(index=False):
        union_ids(row[0], row[1])

    component_rows = []
    component_ids = pd.Series(list(parent.keys()), dtype="string")
    if len(component_ids):
        for component_id, ids in component_ids.groupby(component_ids.apply(find_parent), sort=False):
            ids_list = sorted(map(str, ids.tolist()))
            component_drop_ids = sorted(set(ids_list) & drop_ids)
            survivor_ids = sorted(set(ids_list) - set(component_drop_ids))
            component_rows.append(
                {
                    "component_id": component_id,
                    "component_size": len(ids_list),
                    "comment_ids": ids_list,
                    "drop_ids": component_drop_ids,
                    "survivor_ids": survivor_ids,
                    "num_survivors": len(survivor_ids),
                    "has_conflict": bool(set(ids_list) & set(conflicting_drop_ids)),
                }
            )

    components_df = pd.DataFrame(
        component_rows,
        columns=[
            "component_id",
            "component_size",
            "comment_ids",
            "drop_ids",
            "survivor_ids",
            "num_survivors",
            "has_conflict",
        ],
    ).sort_values(
        ["has_conflict", "num_survivors", "component_size"],
        ascending=[False, True, False],
        kind="stable",
    ).reset_index(drop=True)

    conflict_rows_df = merge_df[
        merge_df["kept_id"].astype(str).isin(conflicting_drop_ids)
        | merge_df["dropped_id"].astype(str).isin(conflicting_drop_ids)
    ].copy()
    problem_components_df = components_df[components_df["num_survivors"].ne(1)].copy()

    return {
        "merge_decisions_df": merge_df,
        "drop_ids": drop_ids,
        "conflicting_drop_ids": conflicting_drop_ids,
        "conflict_rows_df": conflict_rows_df,
        "components_df": components_df,
        "problem_components_df": problem_components_df,
    }

def build_problem_merge_review_rows(
    audit_result: dict[str, object],
    decisions_df: Optional[pd.DataFrame] = None,
    *,
    left_id_col: str = "left_id",
    right_id_col: str = "right_id",
) -> pd.DataFrame:
    problem_components_df = audit_result.get("problem_components_df")
    if not isinstance(problem_components_df, pd.DataFrame) or problem_components_df.empty:
        return pd.DataFrame()

    source_df = decisions_df
    if source_df is None:
        source_df = audit_result.get("merge_decisions_df")
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        return pd.DataFrame()

    review_rows = []
    for component in problem_components_df.itertuples(index=False):
        comment_ids = set(map(str, getattr(component, "comment_ids")))
        left_ids = source_df[left_id_col].astype(str)
        right_ids = source_df[right_id_col].astype(str)
        component_rows = source_df[left_ids.isin(comment_ids) & right_ids.isin(comment_ids)].copy()

        component_rows.insert(0, "component_id", getattr(component, "component_id"))
        component_rows.insert(1, "component_size", getattr(component, "component_size"))
        component_rows.insert(2, "component_survivor_ids", [getattr(component, "survivor_ids")] * len(component_rows))
        component_rows.insert(3, "component_drop_ids", [getattr(component, "drop_ids")] * len(component_rows))
        component_rows.insert(4, "component_has_conflict", getattr(component, "has_conflict"))
        review_rows.append(component_rows)

    if not review_rows:
        return pd.DataFrame()

    return pd.concat(review_rows, axis=0).sort_values(
        ["component_has_conflict", "component_size", "component_id"],
        ascending=[False, False, True],
        kind="stable",
    )
