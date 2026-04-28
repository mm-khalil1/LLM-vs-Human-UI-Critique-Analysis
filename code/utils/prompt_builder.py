import pandas as pd
from typing import List, Dict, Union

# -------------------- Constants --------------------
ALLOWED_ASPECTS = {"aesthetics_rating", "usability_rating", "design_quality_rating"}

SCREEN_TASK_PREFIX = "Screen Task: "
RETURN_PREFIX = "Return ONLY the specified JSON object with no additional text, code fences, or commentary.\n"

FEWSHOT_INTRO = (
    "First, you are given three example images (Image 1-3) with their tasks and expert ratings, "
    "followed by a target image (Image 4) to evaluate, in that order."
)

TARGET_INTRO_LINE = "\nNow evaluate ONLY the target image (Image 4) for the following screen task:"

LABELS = ["low rating", "mid rating", "high rating"]

# Aspect blocks
AESTHETICS_BLOCK = {
    "title": "Aesthetics",
    "rubric": (
        "Evaluate the screen's aesthetics (overall look of the UI). "
        "Consider factors such as layout, color scheme, and visual complexity in your evaluation.\n"
        "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
    ),
    "json": '{"aesthetics_rating": <number>}',
}
USABILITY_BLOCK = {
    "title": "Usability",
    "rubric": (
        "Evaluate the screen's usability through the following dimensions:\n"
        "- Learnability: How easy is it to figure out how to complete the task? How well does a user understand the purpose of the UI and what each component in the UI is for?\n"
        "- Efficiency: How quickly and easily could a user complete the task based on what's shown?\n"
        "- Usability Rating: Based on your ratings for Learnability and Efficiency, provide an overall usability rating.\n\n"
        "Use the following scales:\n"
        "- Learnability: 1 = not at all, 5 = very\n"
        "- Efficiency: 1 = not at all, 5 = very\n"
        "- Usability Rating: 1 = worst usability, 10 = best usability\n"
    ),
    "json": '{"learnability": <number>, "efficiency": <number>, "usability_rating": <number>}',
}
DESIGN_QUALITY_BLOCK = {
    "title": "Design Quality",
    "rubric": (
        "Evaluate the screen's overall design quality.\n"
        "Rate from 1 to 10 (1 = very poor, 10 = excellent).\n"
    ),
    "json": '{"design_quality_rating": <number>}',
}


# -------------------- Helpers --------------------
def _validate_aspect(evaluation_aspect: str) -> None:
    if evaluation_aspect not in ALLOWED_ASPECTS:
        raise ValueError(f"Unsupported aspect: {evaluation_aspect}. Must be one of {sorted(ALLOWED_ASPECTS)}.")

def _get_aspect_block(evaluation_aspect: str) -> Dict[str, str]:
    if evaluation_aspect == "aesthetics_rating":
        return AESTHETICS_BLOCK
    if evaluation_aspect == "usability_rating":
        return USABILITY_BLOCK
    if evaluation_aspect == "design_quality_rating":
        return DESIGN_QUALITY_BLOCK
    # Should be unreachable due to validation
    raise ValueError(f"Unsupported aspect: {evaluation_aspect}")

def _build_header(guidelines: str, task: str, prompting_type: str) -> str:
    header = (
        "You are a UI expert.\n"
        "Follow the specified guidelines to evaluate a mobile app screen for the given screen task.\n"
        f"Guidelines to follow: {guidelines}\n"
    )
    if prompting_type == "zero_shot":
        header += f"{SCREEN_TASK_PREFIX}{task}\n"
    return header

def _return_json_line(block_json: str) -> str:
    return RETURN_PREFIX + block_json

def _normalize_examples(samples: Union[pd.DataFrame, List[Dict], None]) -> List[Dict]:
    if samples is None:
        return []
    if isinstance(samples, pd.DataFrame):
        return samples.to_dict(orient="records")
    return list(samples)

def _example_rating_line(ex: Dict, evaluation_aspect: str) -> str:
    if evaluation_aspect == "aesthetics_rating":
        ex_rating = ex.get("aesthetics_rating", "N/A")
        return f"Expert Aesthetics Rating: {ex_rating}"
    if evaluation_aspect == "usability_rating":
        lr = ex.get("learnability", "N/A")
        ef = ex.get("efficiency", "N/A")
        ur = ex.get("usability_rating", "N/A")
        return f"Expert Learnability: {lr}, Efficiency: {ef}, Usability Rating: {ur}"
    if evaluation_aspect == "design_quality_rating":
        ex_rating = ex.get("design_quality_rating", "N/A")
        return f"Expert Design Quality Rating: {ex_rating}"
    # Should be unreachable due to validation
    return "N/A"

def _example_block_line(idx: int, label: str, task_text: str, rating_line: str) -> str:
    return (
        f"Image {idx} — {label}\n"
        f"{SCREEN_TASK_PREFIX}{task_text}\n"
        f"{rating_line}"
    )

# -------------------- Main --------------------
def build_rating_prompt(task, guidelines, evaluation_aspect, prompting_type, samples_df=None):
    # Validate aspect up front
    _validate_aspect(evaluation_aspect)

    # Common header
    header = _build_header(guidelines=guidelines, task=task, prompting_type=prompting_type)

    # Aspect-specific block
    block = _get_aspect_block(evaluation_aspect)

    # Zero-shot prompt
    if prompting_type == "zero_shot":
        parts = [
            header,
            block["rubric"],
        ]
        parts.append(_return_json_line(block["json"]))
        return "\n".join(parts)

    # Few-shot prompt
    elif prompting_type == "few_shot":
        examples = _normalize_examples(samples_df)
        if len(examples) != 3:
            raise ValueError(f"Few-shot prompting requires exactly 3 examples (got {len(examples)}).")

        lines = [
            header,
            block["rubric"],
            FEWSHOT_INTRO,
        ]

        for idx, ex in enumerate(examples, start=1):
            label = LABELS[idx - 1]
            ex_task = ex.get("task", "N/A")
            rating_line = _example_rating_line(ex, evaluation_aspect)
            lines.append(_example_block_line(idx, label, ex_task, rating_line))

        lines.append(TARGET_INTRO_LINE)
        lines.append(f"{SCREEN_TASK_PREFIX}{task}")
        lines.append(_return_json_line(block["json"]))
        return "\n".join(lines)

    raise ValueError("prompting_type must be 'zero_shot' or 'few_shot'.")

def build_critique_prompt(task, guidelines, prompting_type='zero_shot', evaluation_aspect=None, samples_df=None):
    return f"""You are evaluating a mobile app screen in the context of the following task:
{task}

Identify all distinct, task-relevant usability or visual design issues that may hinder successful task completion.
Each issue must be concrete, non-overlapping, and grounded in the provided guidelines.

Guidelines to reference:
{guidelines}

For every issue, provide:
- expected_standard: What good design should look like.
- observed_issue: What is wrong in the current screen.
- suggested_fix: A clear, actionable improvement.
- guideline_reference: The exact guideline(s) the issue violates.

Rules:
- Focus on one clear problem per critique.
- No praise, summaries, or overall ratings.
- Do not repeat the same issue in different wording.
- Do not invent issues when none exist.

Return exactly this JSON structure with no extra text:

"""+"""{
  "critiques": [
    {
      "expected_standard": "...",
      "observed_issue": "...",
      "suggested_fix": "...",
      "guideline_reference": "..."
    }
  ]
}"""
