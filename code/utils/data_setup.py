from pathlib import Path
import pandas as pd
from typing import Union
from .llm_evaluation_utils import EVALUATION_FIVE_ASPECTS

VALID_TASK_SUBSETS = {"all_tasks", "1000_tasks", "50_tasks"}


def normalize_task_subset(task_subset: str) -> str:
    """Return the canonical task subset name used across notebooks and results files."""
    if task_subset not in VALID_TASK_SUBSETS:
        allowed = ", ".join(sorted(VALID_TASK_SUBSETS))
        raise ValueError(f"Unknown task_subset: {task_subset!r}. Expected one of: {allowed}")
    return task_subset

def initialize_responses_df(uicrit_file: Union[str, Path], num_trials: int, stage: str,) -> pd.DataFrame:
    """Build an empty response frame (one row per screen_task × trial)."""
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")

    # Load only what you need
    uicrit_df = pd.read_parquet(uicrit_file, columns=["screen_task_id", "screen_id", "task"])

    # Cross join with trials (vectorized; avoids iterrows)
    if num_trials > 1:
        trials = pd.DataFrame({"trial": range(1, num_trials + 1)})
        base = uicrit_df.merge(trials, how="cross")
    else:
        base = uicrit_df.copy()

    if stage == "ratings":
        # Initialize rating columns to NA (nullable)
        for c in EVALUATION_FIVE_ASPECTS:
            base[c] = None

        ordered_cols = (
            ["screen_task_id", "screen_id", "task"]
            + (["trial"] if num_trials > 1 else [])
            + EVALUATION_FIVE_ASPECTS
        )
        return base[ordered_cols]

    elif stage == "critiques":
        # Single column to store the full JSON critique response (as text)
        base["critiques"] = None

        ordered_cols = (
            ["screen_task_id", "screen_id", "task"]
            + (["trial"] if num_trials > 1 else [])
            + ["critiques"]
        )
        return base[ordered_cols]

    else:
        raise ValueError(f"Unknown stage: {stage!r}")

def get_dataset_file_path(stage: str) -> Path:
    DATA_DIR = Path("../../../data")

    if stage not in {"ratings", "critiques"}:
        raise ValueError(f"Unknown stage: {stage!r}")
    
    uicrit_data_file = DATA_DIR / "dataset/cleaned_dataset/uicrit_notna_deduped.parquet"
    if stage == "ratings":
        few_shot_samples_file = DATA_DIR / "few_shot_samples/few_shot_samples_df.parquet"
    elif stage == "critiques":
        few_shot_samples_file = None  # not used in critiques stage

    base64_screens_file = DATA_DIR / "dataset/cleaned_dataset/base64_screens.parquet"

    return uicrit_data_file, base64_screens_file, few_shot_samples_file

def load_existing_responses(
    results_file_path: Union[str, Path],
    uicrit_file: Union[str, Path],
    num_trials: int,
    stage: str,
) -> pd.DataFrame:
    path = Path(results_file_path)

    if path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    
    print(f"No existing results for '{path.name}'. Initializing new DataFrame.")
    return initialize_responses_df(uicrit_file, num_trials, stage)

def prepare_project_data(model_short: str, num_trials: int, shots: str, stage: str, task_subset: str = "all_tasks"):
    """
    Centralized data loader for ratings or critiques notebooks.

    Uses canonical subset names only: `all_tasks`, `1000_tasks`, `50_tasks`.
    """
    task_subset = normalize_task_subset(task_subset)

    # File paths
    uicrit_data_file, _, few_shot_samples_file = get_dataset_file_path(stage)

    # Outputs
    base_name = f"{stage}-{shots}-{task_subset}-{model_short}-responses"
    all_results_dir = Path(f"../../results/{stage}")
    all_results_dir.mkdir(parents=True, exist_ok=True)
    text_responses_jsonl_file = all_results_dir / "jsonl" / f"{base_name}.jsonl"
    model_results_file        = all_results_dir / "parquet" / f"{base_name}.parquet"
    text_responses_jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    model_results_file.parent.mkdir(parents=True, exist_ok=True)

    responses_df = load_existing_responses(
        results_file_path=model_results_file,
        uicrit_file=uicrit_data_file,
        num_trials=num_trials,
        stage=stage,
    )

    # ---- Limit by subset: all_tasks / 1000_tasks / 50_tasks ----

    if task_subset in {"1000_tasks", "50_tasks"}:
        # keep one task per screen
        first_tasks = responses_df.drop_duplicates(subset="screen_id", keep="first")
        # ids of tasks to keep
        keep_count = int(task_subset.split("_")[0])
        keep_ids = set(first_tasks["screen_task_id"].head(keep_count))
        # filter
        responses_df = responses_df[responses_df["screen_task_id"].isin(keep_ids)]
        responses_df = responses_df.reset_index(drop=True)

    paths = {
        "results_jsonl": text_responses_jsonl_file,
        "results_parquet": model_results_file,
        }

    if shots == "few_shot":
        few_shot_samples_df = pd.read_parquet(few_shot_samples_file)
        # Remove rows whose screen_task_id is in the few-shot samples
        few_shot_ids = set(few_shot_samples_df["screen_id"])
        responses_df = responses_df.loc[~responses_df["screen_id"].isin(few_shot_ids)].reset_index(drop=True)
    else:
        few_shot_samples_df = None  # empty

    return responses_df, paths

def merge_subset_into_full_by_key(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    *,
    key: str = "screen_task_id",
    value_col: str = "critiques",
    only_fill_missing: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Merge values from a subset results frame into a full results frame by key.

    This is useful when you ran a smaller subset first (for example 1000 tasks),
    then ran another batch on the remaining tasks, and now want one unified frame.
    """
    if key not in full_df.columns or key not in subset_df.columns:
        raise KeyError(f"Both DataFrames must contain '{key}'.")
    if value_col not in full_df.columns or value_col not in subset_df.columns:
        raise KeyError(f"Both DataFrames must contain '{value_col}'.")

    if full_df[key].duplicated().any():
        dupes = full_df.loc[full_df[key].duplicated(), key].head(10).tolist()
        raise ValueError(f"Duplicate keys found in full_df for '{key}'. Examples: {dupes}")

    if subset_df[key].duplicated().any():
        dupes = subset_df.loc[subset_df[key].duplicated(), key].head(10).tolist()
        raise ValueError(f"Duplicate keys found in subset_df for '{key}'. Examples: {dupes}")

    merged_df = full_df.copy(deep=True)

    full_missing_before = merged_df[value_col].isna() | (merged_df[value_col] == "")
    subset_has_value = subset_df[value_col].notna() & (subset_df[value_col] != "")
    subset_map = subset_df.loc[subset_has_value, [key, value_col]].set_index(key)[value_col]

    mapped_values = merged_df[key].map(subset_map)
    matches_in_full = merged_df[key].isin(subset_map.index)

    if only_fill_missing:
        fill_mask = full_missing_before & mapped_values.notna() & (mapped_values != "")
    else:
        fill_mask = mapped_values.notna() & (mapped_values != "")

    merged_df.loc[fill_mask, value_col] = mapped_values.loc[fill_mask]

    full_missing_after = merged_df[value_col].isna() | (merged_df[value_col] == "")

    diagnostics = {
        "full_rows": len(merged_df),
        "subset_rows": len(subset_df),
        "subset_non_null_values": int(subset_has_value.sum()),
        "subset_keys_found_in_full": int(matches_in_full.sum()),
        "full_missing_before": int(full_missing_before.sum()),
        "filled_from_subset": int(fill_mask.sum()),
        "full_missing_after": int(full_missing_after.sum()),
    }

    return merged_df, diagnostics
