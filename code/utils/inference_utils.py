import json
import time
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any, Optional

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
    aspect = aspect or "critiques"

    record = {
        "screen_id": screen_id,
        "screen_task_id": screen_task_id,
        "trial": trial,
        "aspect": aspect,
        "raw_response": raw_response,
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def recover_responses_from_jsonl(responses_df, jsonl_path, update_results_df_fn):
    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        print(f"No JSONL recovery file found: {jsonl_path}")
        return {"recovered": 0, "errors": 0}

    recovered = 0
    errors = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                update_results_df_fn(
                    responses_df,
                    screen_task_id=record["screen_task_id"],
                    trial=record.get("trial"),
                    aspect=record.get("aspect"),
                    response=record["raw_response"],
                )

                recovered += 1

            except Exception as e:
                errors += 1
                print(f"Could not recover line {line_num}: {e}")

    print(f"Recovered {recovered} responses from JSONL. Errors: {errors}")
    return {"recovered": recovered, "errors": errors}

def update_rating_in_df(df, screen_task_id, trial, aspect, response):
    rating_dict = _extract_json_from_response(response)
    if rating_dict is None:
        print(f"Response was: {response}")
        raise ValueError(f"Failed to parse JSON for {screen_task_id}, trial {trial}, aspect {aspect}")

    has_trial = 'trial' in df.columns and trial is not None
    mask = (df['screen_task_id'] == screen_task_id)
    if has_trial:
        mask &= (df['trial'] == trial)

    if not mask.any():
        raise ValueError(f"No matching row for {screen_task_id}, trial={trial}")

    if aspect == 'usability_rating':
        for k in ['learnability', 'efficiency', 'usability_rating']:
            if k in rating_dict:
                df.loc[mask, k] = rating_dict[k]
    else:
        if aspect in rating_dict:
            df.loc[mask, aspect] = rating_dict[aspect]

def update_critiques_in_df(df, screen_task_id, trial, aspect, response):
    obj = _extract_json_from_response(response)

    if not isinstance(obj, dict) or "critiques" not in obj or not isinstance(obj["critiques"], list):
        print("Response was:", response)
        raise ValueError(f"Expected {{'critiques': [...]}} for {screen_task_id}, trial {trial}")

    has_trial = 'trial' in df.columns and trial is not None
    mask = (df['screen_task_id'] == screen_task_id)
    if has_trial:
        mask &= (df['trial'] == trial)

    df.loc[mask, "critiques"] = json.dumps(obj["critiques"], ensure_ascii=False)
    # df.loc[mask, "critiques"] = obj["critiques"]

def _calculate_delay(requests_per_minute: int) -> float:
    """Return delay (in seconds) between API calls for the given rate limit."""
    return 60.0 / max(1, int(requests_per_minute))

def run_llm_inference(
    responses_df: pd.DataFrame,
    screenshots_df: pd.DataFrame,
    few_shot_samples_df: Optional[pd.DataFrame],
    evaluation_aspects: Optional[list[str]],   # ratings: list[str], critiques: None
    build_prompt_fn: Callable[..., str],
    query_fn: Callable[..., str],
    query_args: Dict[str, Any],
    save_jsonl_fn: Callable[..., None],
    update_results_df_fn: Callable[..., None],
    output_jsonl: Path,
    guidelines: str,
    prompting_type: str,
    stage: str,                                # "ratings" or "critiques"
    requests_per_minute: int = 1000,
    print_output: bool = False,
) -> None:

    delay = _calculate_delay(requests_per_minute)

    if screenshots_df.index.name != "screen_id":
        if "screen_id" in screenshots_df.columns:
            screenshots_df = screenshots_df.set_index("screen_id", drop=False)
        else:
            raise KeyError("Column 'screen_id' not found in screenshots_df.")

    # --- determine which rows are incomplete and which "aspects" to run ---
    if stage == "ratings":
        if not evaluation_aspects:
            raise ValueError("evaluation_aspects must be provided for ratings stage.")
        completion_cols = evaluation_aspects
        aspects_to_run = evaluation_aspects
    elif stage == "critiques":
        completion_cols = ["critiques"]   # single column for whole JSON critique
        aspects_to_run = [""]                # single pseudo-aspect
    else:
        raise ValueError(f"Unknown stage: {stage!r}")

    incomplete_rows = responses_df[responses_df[completion_cols].isnull().any(axis=1)]

    # --- main loop (shared) ---
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
            
            save_jsonl_fn(
                response_text,
                screen_id=row.screen_id,
                screen_task_id=row.screen_task_id,
                trial=None,
                aspect=aspect,
                out_path=output_jsonl,
            )

            update_results_df_fn(
                responses_df,
                screen_task_id=row.screen_task_id,
                trial=None,
                aspect=aspect,
                response=response_text,
            )

            time.sleep(delay)
