"""LLM-judge reviews of comment pairs: prompts, batch requests and parsing of batch responses.

Two reviews share this machinery:
- DUPLICATE_REVIEW: do two expert comments on one task describe the same issue (and which one to keep)?
- MATCH_REVIEW: how does an expert comment relate to an LLM comment on the same task?

Requests and responses are keyed by each pair's `custom_id` column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import numpy as np
import pandas as pd

from utils.batch_manager import extract_batch_record_id, extract_message_content_from_batch_record

BBOX_KEYS = ("x_min", "y_min", "x_max", "y_max")
CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}

DUPLICATE_RELATIONS = ["duplicate", "a_contains_b", "b_contains_a", "partial_overlap", "different"]
DUPLICATE_ACTIONS = ["keep_A", "keep_B", "keep_both", "manual_review"]
MATCH_RELATIONS = [
    "equivalent",
    "A_broader",
    "B_broader",
    "partial_overlap",
    "related_but_different",
    "no_match",
    "contradiction",
]
DIAGNOSTIC_COLUMNS = [
    "same_target_element",
    "same_usability_problem",
    "comment_a_depends_on_unseen_interaction",
    "comment_b_depends_on_unseen_interaction",
    "logical_conflict",
]
# Diagnostics that follow from the relation itself; used to set the flags of manually overridden pairs.
RELATION_DIAGNOSTICS = {
    "same_usability_problem": {
        "equivalent": "yes",
        "A_broader": "yes",
        "B_broader": "yes",
        "partial_overlap": "partial",
        "related_but_different": "no",
        "no_match": "no",
    },
    "logical_conflict": {relation: "yes" if relation == "contradiction" else "no" for relation in MATCH_RELATIONS},
}


class Review(NamedTuple):
    system_prompt: str
    schema: dict
    prompt: Callable[[pd.Series], str]


def _json_schema(name: str, properties: dict) -> dict:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _bbox_text(bbox) -> str:
    if not isinstance(bbox, dict):
        return "unavailable"
    return "[" + ", ".join(f"{float(bbox[key]):.3f}" for key in BBOX_KEYS) + "]"


def duplicate_prompt(row: pd.Series) -> str:
    iou = row["bbox_iou"]
    return "\n".join([
        "Decide whether these two comments describe the same atomic UI issue for the same task. Check the bounding boxes and IoU to see if they refer to the same visual target, but judge the underlying issue, not just wording or element overlap.",
        "The bounding boxes are in [x_min, y_min, x_max, y_max] format, normalized to the screen size. IoU is the intersection-over-union of the two bounding boxes.",
        "Use the attached screenshot as task context when wording is ambiguous.",
        "",
        f"Task: {_text(row['task'])}",
        f"Comment A: {_text(row['left_observed_issue'])}",
        f"Comment A bounding box: {_bbox_text(row['left_bounding_box'])}",
        f"Comment B: {_text(row['right_observed_issue'])}",
        f"Comment B bounding box: {_bbox_text(row['right_bounding_box'])}",
        f"BBox IoU between annotated regions: {'unavailable' if pd.isna(iou) else f'{float(iou):.3f}'}",
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
    ])


def match_prompt(row: pd.Series) -> str:
    return "\n".join([
        "Determine the relationship between Comment A and Comment B for the same mobile screen and same task, using the attached screen image as grounding evidence.",
        "",
        "Judge the underlying UI/UX issue, not wording overlap. Two comments can match even if phrased "
        "differently. Do not treat two comments as a match or overlap just because they mention the same interface "
        "element; they must point to the same actual problem, or one must clearly contain the other’s problem content.",
        "",
        "Comment A is talking about the UI region identified by its bounding box. Use that box to ground the target element for Comment A, but judge the underlying usability problem described by both comments.",
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
        "",
        "Task:",
        _text(row["task"]),
        "",
        "Comment A bounding box in [x_min, y_min, x_max, y_max] format:",
        _bbox_text(row["expert_bounding_box"]),
        "",
        "Comment A:",
        _text(row["expert_observed_issue"]),
        "",
        "Comment B:",
        _text(row["llm_observed_issue"]),
        "",
        "Return exactly the following JSON schema. You must evaluate the intermediate steps before generating the final relation.",
        "{",
        '  "same_target_element": "yes" | "no" | "partial",',
        '  "same_usability_problem": "yes" | "no" | "partial",',
        '  "comment_a_depends_on_unseen_interaction": "yes" | "no" | "partial",',
        '  "comment_b_depends_on_unseen_interaction": "yes" | "no" | "partial",',
        '  "logical_conflict": "yes" | "no" | "partial",',
        '  "relation": "<one allowed label>",',
        '  "confidence": a number between 0 and 1',
        "}",
    ])


DUPLICATE_REVIEW = Review(
    system_prompt=(
        "You judge whether two UI critique comments about the same task describe the same atomic UI issue. "
        "Use the screenshot only to resolve UI context, not to invent issues that are not in the comments. "
        "Return valid JSON only."
    ),
    schema=_json_schema("semantic_comment_relationship_judgment", {
        "relationship": {"type": "string", "enum": DUPLICATE_RELATIONS},
        "reasoning": {"type": "string"},
        "recommended_action": {"type": "string", "enum": DUPLICATE_ACTIONS},
        "confidence": CONFIDENCE,
    }),
    prompt=duplicate_prompt,
)
MATCH_REVIEW = Review(
    system_prompt=(
        "You are an expert evaluator of UI/UX critique alignment. "
        "Return valid JSON only. No markdown, comments, or extra text."
    ),
    schema=_json_schema("human_llm_comment_match_judgment", {
        **{col: {"type": "string", "enum": ["yes", "no", "partial"]} for col in DIAGNOSTIC_COLUMNS},
        "relation": {"type": "string", "enum": MATCH_RELATIONS},
        "confidence": CONFIDENCE,
    }),
    prompt=match_prompt,
)


def build_request(
    custom_id: str,
    prompt: str,
    review: Review,
    *,
    provider: str,
    model: str,
    base64_screen: str | None,
    image_format: str = "jpeg",
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
) -> dict:
    """One OpenAI chat-completions or Gemini batch request with the screenshot attached."""
    if provider == "openai":
        content = prompt
        if base64_screen:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_screen}"}},
            ]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": review.system_prompt}, {"role": "user", "content": content}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        body["response_format"] = {"type": "json_schema", "json_schema": review.schema}
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}

    if provider == "gemini":
        parts = [{"text": prompt}]
        if base64_screen:
            parts.append({"inlineData": {"mimeType": f"image/{image_format}", "data": base64_screen}})
        config = {}
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["maxOutputTokens"] = max_tokens
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = review.schema["schema"]
        return {
            "key": custom_id,
            "request": {
                "model": f"models/{model}",
                "contents": [{"role": "user", "parts": parts}],
                "systemInstruction": {"role": "user", "parts": [{"text": review.system_prompt}]},
                "generation_config": config,
            },
        }

    raise ValueError(f"Unsupported provider: {provider}")


def iter_requests(pairs_df: pd.DataFrame, review: Review, *, screens: dict, **request_kwargs):
    """Yield one request per pair; `screens` maps screen_id to its base64 screenshot."""
    for _, row in pairs_df.iterrows():
        yield build_request(
            str(row["custom_id"]),
            review.prompt(row),
            review,
            base64_screen=screens.get(row["screen_id"]),
            **request_kwargs,
        )


def _parse_record(record: dict, schema: dict) -> tuple[dict, str | None]:
    """The judge's JSON answer and the reason it is unusable (None when it is valid)."""
    status = (record.get("response") or {}).get("status_code")
    if status not in (None, 200):
        return {}, f"http status {status}"
    if record.get("error") is not None:
        return {}, "provider error"
    content = extract_message_content_from_batch_record(record)
    if not content:
        return {}, "missing content"
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        return {}, "invalid json"
    if not isinstance(answer, dict):
        return {}, "invalid json"
    for field in schema["required"]:
        if answer.get(field) is None:
            return answer, f"missing {field}"
        if answer[field] not in schema["properties"][field].get("enum", [answer[field]]):
            return answer, f"invalid {field}"
    return answer, None


def read_results(paths: Iterable[Path | str], review: Review) -> pd.DataFrame:
    """One row per custom_id: the judge's answer fields plus `error` (None when the answer is valid).

    Later files win for repeated custom_ids, so list retry responses after the batches they retry.
    """
    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[extract_batch_record_id(record)] = record

    rows = []
    for custom_id, record in records.items():
        answer, error = _parse_record(record, review.schema["schema"])
        rows.append({"custom_id": custom_id, **answer, "error": error})
    return pd.DataFrame(rows, columns=["custom_id", *review.schema["schema"]["required"], "error"])


def failed_ids(results_df: pd.DataFrame, expected_ids: Iterable) -> set[str]:
    """Expected custom_ids without a valid answer (missing or invalid)."""
    valid_ids = set(results_df.loc[results_df["error"].isna(), "custom_id"])
    return set(map(str, expected_ids)) - valid_ids


def compare_judges(results: dict[str, pd.DataFrame], fields: list[str]) -> pd.DataFrame:
    """The judges' answers side by side per custom_id, as `<field>__<judge>` columns."""
    return pd.concat(
        [df.set_index("custom_id")[fields].add_suffix(f"__{judge}") for judge, df in results.items()],
        axis=1,
    ).rename_axis("custom_id").reset_index()


def finalize_match_judges(
    comparison_df: pd.DataFrame,
    left_judge: str,
    right_judge: str,
    manual_relations: dict[str, str],
) -> pd.DataFrame:
    """Final relation per pair: the judges' shared answer, else a manual relation (keyed by pair_id).

    Agreed pairs get the judges' mean confidence and manual ones confidence 1. `decision_source` is
    judge_agreement, manual_override or manual_review (still unresolved).
    """
    if invalid := {pair_id: r for pair_id, r in manual_relations.items() if r not in MATCH_RELATIONS}:
        raise ValueError(f"Invalid manual relations: {invalid}")

    df = comparison_df.copy()
    agreed = df[f"relation__{left_judge}"].eq(df[f"relation__{right_judge}"])
    manual = df["pair_id"].map(manual_relations)
    df["relation"] = df[f"relation__{left_judge}"].where(agreed, manual)
    df["confidence"] = (
        df[[f"confidence__{left_judge}", f"confidence__{right_judge}"]].mean(axis=1)
        .where(agreed, np.where(manual.notna(), 1.0, np.nan))
    )
    df["decision_source"] = np.select(
        [agreed, manual.notna()], ["judge_agreement", "manual_override"], default="manual_review"
    )
    return df


def finalize_diagnostics(final_df: pd.DataFrame, left_judge: str, right_judge: str) -> pd.DataFrame:
    """Final value per diagnostic flag, consistent with the final relation.

    A flag both judges agree on is kept. Otherwise it comes from the judge whose relation became final
    (manual override or confidence fallback), and is "disagree" when both or neither judges' relation did.
    Manual overrides take the flags that follow from the relation itself (RELATION_DIAGNOSTICS).
    """
    df = final_df.copy()
    left_kept = df["relation"].eq(df[f"relation__{left_judge}"])
    right_kept = df["relation"].eq(df[f"relation__{right_judge}"])
    manual = df["decision_source"].eq("manual_override")
    for col in DIAGNOSTIC_COLUMNS:
        left, right = df[f"{col}__{left_judge}"], df[f"{col}__{right_judge}"]
        value = pd.Series(
            np.select([left.eq(right), left_kept & ~right_kept, right_kept & ~left_kept], [left, left, right], "disagree"),
            index=df.index,
        )
        derived = df["relation"].map(RELATION_DIAGNOSTICS.get(col, {}))
        df[col] = value.mask(manual & derived.notna(), derived)
    return df
