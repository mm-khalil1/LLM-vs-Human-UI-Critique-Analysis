import json
import time
import pandas as pd
from typing import Callable, Optional

EVALUATION_FIVE_ASPECTS = ['aesthetics_rating', 'learnability', 'efficiency', 'usability_rating', 'design_quality_rating']
EVALUATION_MAIN_ASPECTS = ['aesthetics_rating', 'usability_rating', 'design_quality_rating']

GUIDELINES = "Nielsen Norman 10 Usability Heuristics, Apple Human Interface Guidelines, and CrowdCrit Visual Design Critiques"
RATING_SYSTEM_MESSAGE = """You are a senior user interface (UI) and user experience (UX) expert with deep knowledge of visual design principles, usability heuristics, and mobile app evaluation. You provide accurate ratings of screen designs based on provided tasks and design guidelines. Your output should follow the exact JSON format requested."""
CRITIQUES_SYSTEM_MESSAGE = """You are a senior UI/UX evaluator specializing in mobile app usability reviews.
You identify concrete, task-relevant design issues in screens and express them as structured critiques.
Each critique should be specific, atomic (one issue only), grounded in established usability or visual design guidelines, and written in clear, evaluative language similar to expert design reviews.
Do not praise the design or provide general summaries.
Return structured critiques only in the specified JSON format."""

def _extract_json_from_response(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    cleaned = text.replace("```json", "").replace("```", "").strip()

    # extract the JSON object even if extra text exists
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    json_str = cleaned[start:end+1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def append_response_jsonl(raw_response, *, screen_id, screen_task_id, trial, aspect, out_path):
    record = {
        "screen_id": screen_id,
        "screen_task_id": screen_task_id,
        "trial": trial,
        "aspect": aspect or "critiques",
        "raw_response": raw_response,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _row_mask(df, screen_task_id, trial):
    mask = df['screen_task_id'] == screen_task_id
    if 'trial' in df.columns and trial is not None:
        mask &= df['trial'] == trial
    return mask

def update_rating_in_df(df, screen_task_id, trial, aspect, response):
    rating_dict = _extract_json_from_response(response)
    if rating_dict is None:
        print(f"Response was: {response}")
        raise ValueError(f"Failed to parse JSON for {screen_task_id}, trial {trial}, aspect {aspect}")

    mask = _row_mask(df, screen_task_id, trial)
    if not mask.any():
        raise ValueError(f"No matching row for {screen_task_id}, trial={trial}")

    keys = ['learnability', 'efficiency', 'usability_rating'] if aspect == 'usability_rating' else [aspect]
    for k in keys:
        if k in rating_dict:
            df.loc[mask, k] = rating_dict[k]

def update_critiques_in_df(df, screen_task_id, trial, aspect, response):
    obj = _extract_json_from_response(response)
    if not isinstance(obj, dict) or not isinstance(obj.get("critiques"), list):
        print("Response was:", response)
        raise ValueError(f"Expected {{'critiques': [...]}} for {screen_task_id}, trial {trial}")

    df.loc[_row_mask(df, screen_task_id, trial), "critiques"] = json.dumps(obj["critiques"], ensure_ascii=False)

def run_llm_inference(
    responses_df: pd.DataFrame,
    screenshots_df: pd.DataFrame,
    few_shot_samples_df: Optional[pd.DataFrame],
    evaluation_aspects: Optional[list[str]],   # ratings: list[str], critiques: None
    build_prompt_fn: Callable[..., str],
    query_fn: Callable[..., str],
    query_args: dict,
    save_jsonl_fn: Callable[..., None],
    update_results_df_fn: Callable[..., None],
    output_jsonl,
    guidelines: str,
    prompting_type: str,
    stage: str,                                # "ratings" or "critiques"
    requests_per_minute: int = 1000,
    print_output: bool = False,
) -> None:

    delay = 60.0 / max(1, int(requests_per_minute))

    if screenshots_df.index.name != "screen_id":
        screenshots_df = screenshots_df.set_index("screen_id", drop=False)

    # --- determine which rows are incomplete and which "aspects" to run ---
    if stage == "ratings":
        if not evaluation_aspects:
            raise ValueError("evaluation_aspects must be provided for ratings stage.")
        completion_cols = aspects_to_run = evaluation_aspects
    elif stage == "critiques":
        completion_cols = ["critiques"]   # single column for whole JSON critique
        aspects_to_run = [""]             # single pseudo-aspect
    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    incomplete_rows = responses_df[responses_df[completion_cols].isnull().any(axis=1)]

    for row in incomplete_rows.itertuples(index=True):
        base64_image = screenshots_df.at[row.screen_id, "base64_screen"]
        print(f"[{row.Index}] Processing screen_task_id={row.screen_task_id} ({stage})")

        for aspect in aspects_to_run:
            # For ratings: skip if this aspect already filled
            if stage == "ratings" and pd.notnull(getattr(row, aspect)):
                continue

            prompt = build_prompt_fn(
                row.task,
                guidelines,
                evaluation_aspect=aspect,      # critiques can ignore this
                prompting_type=prompting_type,
                samples_df=few_shot_samples_df,
            )

            try:
                response_text = query_fn(
                    client=query_args["client"],
                    query_args=query_args,
                    prompt=prompt,
                    base64_target=base64_image,
                    samples_df=few_shot_samples_df,
                )
            except Exception as e:
                print(f"⚠️ API error ({row.screen_task_id}/{stage}/{aspect}): {e}")
                return

            if response_text is None:
                print("⚠️ response is None")
                return

            if print_output:
                print(f"Response for {row.screen_task_id} ({aspect}):\n{response_text}\n")

            ids = dict(screen_task_id=row.screen_task_id, trial=None, aspect=aspect)
            save_jsonl_fn(response_text, screen_id=row.screen_id, out_path=output_jsonl, **ids)
            update_results_df_fn(responses_df, response=response_text, **ids)

            time.sleep(delay)
