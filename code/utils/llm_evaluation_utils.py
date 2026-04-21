import re
import json
import time
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, Any, Optional

EVALUATION_FIVE_ASPECTS = ['aesthetics_rating', 'learnability', 'efficiency', 'usability_rating', 'design_quality_rating']
EVALUATION_MAIN_ASPECTS = ['aesthetics_rating', 'usability_rating', 'design_quality_rating']

GUIDELINES = "Nielsen Norman 10 Usability Heuristics, CrowdCrit Visual Design Critiques, and Apple Human Interface Guidelines"
RATING_SYSTEM_MESSAGE = """You are a senior user interface (UI) and user experience (UX) expert with deep knowledge of visual design principles, usability heuristics, and mobile app evaluation. You provide accurate ratings of screen designs based on provided tasks and design guidelines. Your output should follow the exact JSON format requested."""
# CRITIQUES_SYSTEM_MESSAGE = """You are a design evaluation expert. Your task is to analyze the following mobile UI screen and provide constructive design feedback based on established usability heuristics and design guidelines."""
CRITIQUES_SYSTEM_MESSAGE = """You are a senior UI/UX evaluator specializing in mobile app usability reviews.
You identify concrete, task-relevant design issues in screens and express them as structured critiques.
Each critique should be specific, atomic (one issue only), grounded in established usability or visual design guidelines, and written in clear, evaluative language similar to expert design reviews.
Do not praise the design or provide general summaries.
Return structured critiques only in the specified JSON format."""



# def create_prompt(task, guidelines, evaluate=None):
#     if evaluate == 'aesthetics_rating':
#         return (
#             "You are a UI expert.\n"
#             f"Screen Task: {task}\n"
#             f"Guidelines to evaluate: {guidelines}\n\n"
#             "Evaluate the screen based on Aesthetics: the overall look of the UI. "
#             "Consider factors such as layout, color scheme, and visual complexity in your evaluation.\n"
#             "Rate the aesthetics on a scale from 1 to 10, where 1 = very poor and 10 = excellent.\n"
#             "Return ONLY this exact JSON format with no additional text:\n"
#             # "Return this exact JSON format followed by explanation:\n"
#             '{"aesthetics_rating": <number>}\n'            
#         )
#     elif evaluate == 'usability_rating':
#         return (
#             "You are a UI expert.\n"
#             f"Screen Task: {task}\n"
#             f"Guidelines to evaluate: {guidelines}\n\n"
#             "Evaluate the screen based on the following:\n"
#             "Learnability: How easy was it to figure out how to complete the task? How well do you understand the purpose of the UI and what each component in the UI is for?\n"
#             "Efficiency: How quickly and easily could a user complete the task based on what's shown?\n"
#             "Usability Rating: Based on your ratings for Learnability and Efficiency, give an overall usability rating.\n\n"
#             "Use the following scales:\n"
#             "- Learnability: 1 = Not at all, 5 = Very\n"
#             "- Efficiency: 1 = Not at all, 5 = Very\n"
#             "- Usability Rating: 1 = worst usability, 10 = best usability\n\n"
#             "Return ONLY this exact JSON format with no additional text:\n"
#             '{"learnability": <number>, "efficiency": <number>, "usability_rating": <number>}\n'
#         )
#     elif evaluate == 'design_quality_rating':
#         return (
#             "You are a UI expert.\n"
#             f"Screen Task: {task}\n"
#             f"Guidelines to evaluate: {guidelines}\n\n"
#             "Evaluate the screen based on Overall Design Quality\n"
#             "Give your design_quality_rating on a scale from 1 (very poor) to 10 (excellent).\n"
#             "Return ONLY this exact JSON format with no additional text:\n"
#             '{"design_quality_rating": <number>}\n'
#         )
#     elif evaluate == 'feedback':
#         return (
#             "You are a UX expert conducting a heuristic evaluation of the following UI screen.\n\n"
#             f"Screen Task: {task}\n"
#             f"Guidelines to evaluate: {guidelines}\n\n"
#             "For each relevant guideline to the task, identify any usability issue(s), explain how the current design falls short, and suggest improvements. "
#             "Use a structured format per guideline if possible."
#         )
#     else:
#         raise ValueError(f"Unknown evaluation type: {evaluate}. Supported types are: 'aesthetics_rating', 'usability_rating', 'design_quality_rating', 'feedback'.")

# def create_few_shot_prompt(task, guidelines, samples_df, evaluate=None):
#     """
#     Build a few-shot evaluation prompt.

#     samples_df: DataFrame with three rows (lowest, mid, highest)
#                 columns = ['screen_task_id', 'screen_id', 'task',
#                             'aesthetics_rating', 'learnability', 'efficiency',
#                             'usability_rating', 'design_quality_rating', 'base64_screen']
#     """
#     sample_labels = {
#         1: "low rating",
#         2: "mid rating",
#         3: "high rating"
#     }
#     metric_instruction_1 = ""
#     metric_instruction_json_format = ""

#     if evaluate == "aesthetics_rating":
#         metric_instruction_1 = (
#             f"Aesthetics: the overall look of the UI. Use the following guidelines: {guidelines}.\n"
#             "Consider factors such as layout, color scheme, and visual complexity in your evaluation.\n"
#             "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n\n"
#         )
#         metric_instruction_json_format = '{"aesthetics_rating": <number>}'

#     elif evaluate == "usability_rating":
#         metric_instruction_1 = (
#             "the following dimensions:\n"
#             "- Learnability: How easy was it to figure out how to complete the task? How well do you understand the purpose of the UI and what each component in the UI is for?\n"
#             "- Efficiency: How quickly and easily could a user complete the task based on what's shown?\n"
#             "- Usability Rating: Based on your ratings for Learnability and Efficiency, give an overall usability rating.\n"
#             f"Use the following guidelines: {guidelines}.\n"
#             "Scales:\n"
#             "- Learnability: 1 = Not at all, 5 = Very\n"
#             "- Efficiency: 1 = Not at all, 5 = Very\n"
#             "- Usability Rating: 1 = worst usability, 10 = best usability\n\n"
#         )
#         metric_instruction_json_format = (
#             '{"learnability": <number>, "efficiency": <number>, "usability_rating": <number>}'
#         )

#     elif evaluate == "design_quality_rating":
#         metric_instruction_1 = (
#             f"Overall Design Quality. Use the following guidelines: {guidelines}.\n"
#             "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n\n"
#         )
#         metric_instruction_json_format = '{"design_quality_rating": <number>}'

#     elif evaluate == "feedback":
#         metric_instruction_1 = (
#             "You are a UX expert conducting a heuristic evaluation of the following UI screen.\n\n"
#             f"Screen Task: {task}\n"
#             f"Guidelines to evaluate: {guidelines}\n\n"
#             "For each relevant guideline, identify usability issue(s), explain how the design falls short, "
#             "and suggest improvements. Use a structured format per guideline if possible."
#         )

#     else:
#         raise ValueError(
#             f"Unknown evaluation type: {evaluate}. Supported types are: "
#             "'aesthetics_rating', 'usability_rating', 'design_quality_rating', 'feedback'."
#         )

#     # Few-shot examples text
#     few_shot_samples_text = ""
#     if not samples_df.empty and evaluate != "feedback":
#         for i, (_, sample) in enumerate(samples_df.iterrows(), start=1):
#             if evaluate == "aesthetics_rating":
#                 rating_str = f"Expert Aesthetic Rating: {sample['aesthetics_rating']}"
#             elif evaluate == "usability_rating":
#                 rating_str = (
#                     f"Expert Learnability: {sample['learnability']}, "
#                     f"Efficiency: {sample['efficiency']}, "
#                     f"Usability: {sample['usability_rating']}"
#                 )
#             elif evaluate == "design_quality_rating":
#                 rating_str = f"Expert Design Quality: {sample['design_quality_rating']}"
#             else:
#                 rating_str = ""

#             few_shot_samples_text += (
#                 f"Example {i}: {sample_labels.get(i)}\n"
#                 f"Screen Task: {sample['task']}\n"
#                 f"{rating_str}\n"
#                 f"(Associated image: see image labeled Example {i})\n\n"
#             )

#     # Final assembly
#     if evaluate == "feedback":
#         final_prompt = metric_instruction_1
#     else:
#         final_prompt = (
#             f"You are an expert UI evaluator.\n"
#             f"Evaluate a mobile app screen based on: {metric_instruction_1}"
#             "Here are three example screens, with their tasks and expert ratings (lowest → mid → highest):\n"
#             f"{few_shot_samples_text}\n"
#             "Now evaluate the fourth screen."
#             f"Screen Task: {task}\n"
#             "Return ONLY this exact JSON format with no additional text:\n"
#             f"{metric_instruction_json_format}"
#         )
# -------------------------------------------------------------------------------------------------------------------------
#     return final_prompt
# def evaluation_aspect_block(aspect):
#     if aspect == "aesthetics_rating":
#         return {
#             "title": "Aesthetics",
#             "rubric": (
#                 "Evaluate the screen's aesthetics (overall look of the UI). "
#                 "Consider factors such as layout, color scheme, and visual complexity in your evaluation.\n"
#                 "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
#             ),
#             "json": '{"aesthetics_rating": <number>}',
#         }
#     elif aspect == "usability_rating":
#         return {
#             "title": "Usability",
#             "rubric": (
#                 "Evaluate the screen's usability through the following dimensions:\n"
#                 "- Learnability: How easy is it to figure out how to complete the task? How well does a user understand the purpose of the UI and what each component in the UI is for?\n"
#                 "- Efficiency: How quickly and easily could a user complete the task based on what's shown?\n"
#                 "- Usability Rating: Based on your ratings for Learnability and Efficiency, provide an overall usability rating.\n\n"
#                 "Use the following scales:\n"
#                 "- Learnability: 1 = not at all, 5 = very\n"
#                 "- Efficiency: 1 = not at all, 5 = very\n"
#                 "- Usability Rating: 1 = worst usability, 10 = best usability\n"
#             ),
#             "json": '{"learnability": <number>, "efficiency": <number>, "usability_rating": <number>}',
#         }
#     elif aspect == "design_quality_rating":
#         return {
#             "title": "Design Quality",
#             "rubric": (
#                 "Evaluate the screen's overall design quality.\n"
#                 "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
#             ),
#             "json": '{"design_quality_rating": <number>}',
#         }
#     raise ValueError(f"Unsupported aspect: {aspect}")

# def build_rating_prompt(task, guidelines, evaluation_aspect, prompting_type,samples=None):
#     """
#     Build a zero-shot or few-shot prompt to rate a mobile UI screen.

#     Image assumptions:
#       - Zero-shot: a single image is attached (the target).
#       - Few-shot: 3 example images are attached first (Image 1–3), then the target (Image 4).

#     Parameters
#     ----------
#     task : str
#         The target screen task description.
#     guidelines : str | Iterable[str]
#         Guidelines to follow (e.g., 'Nielsen Norman 10 Usability Heuristics, CrowdCrit, Apple HIG')
#         or an iterable that will be joined with ', '.
#     evaluation_aspect : str
#         One of: 'aesthetics_rating', 'usability_rating', 'design_quality_rating'.
#     prompting_type : str
#         'zero_shot' or 'few_shot'.
#     samples : DataFrame | list[dict] | None
#         For few_shot, exactly 3 examples.
#         Each example needs keys: 'task' and the target aspect(s).
#         Example keys:
#           - aesthetics: 'aesthetics_rating'
#           - usability:  'learnability', 'efficiency', 'usability_rating'
#           - design:     'design_quality_rating'
#     """
#     # Validate aspect up front
#     allowed = {"aesthetics_rating", "usability_rating", "design_quality_rating"}
#     if evaluation_aspect not in allowed:
#         raise ValueError(f"Unsupported aspect: {evaluation_aspect}. Must be one of {sorted(allowed)}.")

#     # --- Common header
#     header = (
#         "You are a UI expert.\n"
#         "Follow the specified guidelines to evaluate a mobile app screen for the given screen task.\n"
#         f"Guidelines to follow: {guidelines}\n"
#     )

#     # Add task to header (place task later for few-shot)
#     if prompting_type == "zero_shot":
#         header += f"Screen Task: {task}\n"

#     # --- Aspect-specific block
#     if evaluation_aspect == "aesthetics_rating":
#         block = {
#             "title": "Aesthetics",
#             "rubric": (
#                 "Evaluate the screen's aesthetics (overall look of the UI). "
#                 "Consider factors such as layout, color scheme, and visual complexity in your evaluation.\n"
#                 "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
#             ),
#             "json": '{"aesthetics_rating": <number>}',
#         }
#     elif evaluation_aspect == "usability_rating":
#         block = {
#             "title": "Usability",
#             "rubric": (
#                 "Evaluate the screen's usability through the following dimensions:\n"
#                 "- Learnability: How easy is it to figure out how to complete the task? How well does a user understand the purpose of the UI and what each component in the UI is for?\n"
#                 "- Efficiency: How quickly and easily could a user complete the task based on what's shown?\n"
#                 "- Usability Rating: Based on your ratings for Learnability and Efficiency, provide an overall usability rating.\n\n"
#                 "Use the following scales:\n"
#                 "- Learnability: 1 = not at all, 5 = very\n"
#                 "- Efficiency: 1 = not at all, 5 = very\n"
#                 "- Usability Rating: 1 = worst usability, 10 = best usability\n"
#             ),
#             "json": '{"learnability": <number>, "efficiency": <number>, "usability_rating": <number>}',
#         }
#     elif evaluation_aspect == "design_quality_rating":
#         block = {
#             "title": "Design Quality",
#             "rubric": (
#                 "Evaluate the screen's overall design quality.\n"
#                 "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
#             ),
#             "json": '{"design_quality_rating": <number>}',
#         }
#     else:
#         raise ValueError(f"Unsupported aspect: {evaluation_aspect}")

#     # Zero-shot prompt
#     if prompting_type == "zero_shot":
#         parts = [
#             header,
#             block["rubric"],
#         ]
#         parts.append("Return ONLY the specified JSON object with no additional text, code fences, or commentary.\n" + block["json"])
#         return "\n".join(parts)

#     # Few-shot prompt
#     elif prompting_type == "few_shot":
#         if samples is None:
#             raise ValueError("Few-shot requires `samples` with exactly 3 examples.")
#         # Normalize to list of dicts
#         if isinstance(samples, pd.DataFrame):
#             examples = samples.to_dict(orient="records")
#         else:
#             examples = list(samples)

#         if len(examples) != 3:
#             raise ValueError(f"Few-shot prompting requires exactly 3 examples (got {len(examples)}).")

#         labels = ["low rating", "mid rating", "high rating"]

#         lines = [
#             header,
#             # Put the aspect paragraph BEFORE examples
#             block["rubric"],
#             "First, you are given three example images (Image 1–3) with their tasks and expert ratings, followed by a target image (Image 4) to evaluate, in that order.",
#         ]

#         for idx, ex in enumerate(examples, start=1):
#             label = labels[idx - 1]
#             ex_task = ex.get("task", "N/A")

#             if evaluation_aspect == "aesthetics_rating":
#                 ex_rating = ex.get("aesthetics_rating", "N/A")
#                 rating_line = f"Expert Aesthetics Rating: {ex_rating}"
#             elif evaluation_aspect == "usability_rating":
#                 lr = ex.get("learnability", "N/A")
#                 ef = ex.get("efficiency", "N/A")
#                 ur = ex.get("usability_rating", "N/A")
#                 rating_line = f"Expert Learnability: {lr}, Efficiency: {ef}, Usability Rating: {ur}"
#             elif evaluation_aspect == "design_quality_rating":
#                 ex_rating = ex.get("design_quality_rating", "N/A")
#                 rating_line = f"Expert Design Quality Rating: {ex_rating}"

#             lines.append(
#                 f"Image {idx} — {label}\n"
#                 f"Screen Task: {ex_task}\n"
#                 f"{rating_line}"
#             )

#         # Target comes after examples
#         lines.append("\nNow evaluate ONLY the target image (Image 4) for the following screen task:")
#         lines.append(f"Screen Task: {task}")
#         lines.append("Return ONLY the specified JSON object with no additional text, code fences, or commentary.\n" + block["json"])
#         return "\n".join(lines)

#     raise ValueError("prompting_type must be 'zero_shot' or 'few_shot'.")

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

def save_response_text(raw_response, *, screen_id, screen_task_id, trial, aspect, out_path):
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
    base64_screens_df: pd.DataFrame,
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

    if base64_screens_df.index.name != "screen_id":
        if "screen_id" in base64_screens_df.columns:
            base64_screens_df = base64_screens_df.set_index("screen_id", drop=False)
        else:
            raise KeyError("Column 'screen_id' not found in base64_screens_df.")

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
        base64_image = base64_screens_df.at[row.screen_id, "base64_screen"]
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
