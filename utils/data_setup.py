from pathlib import Path
import pandas as pd
from typing import Union
from .inference_utils import EVALUATION_FIVE_ASPECTS
from .paths import PREPROCESSED_DIR, RESULTS_DIR, SCHEDULES_DIR

VALID_TASK_SUBSETS = {"all_tasks", "1000_tasks", "50_tasks"}
STAGE_VALUE_COLUMNS = {"ratings": EVALUATION_FIVE_ASPECTS, "critiques": ["critiques"]}


def _check_stage(stage: str) -> None:
    if stage not in STAGE_VALUE_COLUMNS:
        raise ValueError(f"Unknown stage: {stage!r}")


def initialize_responses_df(uicrit_file: Union[str, Path], num_trials: int, stage: str) -> pd.DataFrame:
    """Build an empty response frame (one row per screen_task × trial)."""
    _check_stage(stage)
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")

    base = pd.read_parquet(uicrit_file, columns=["screen_task_id", "screen_id", "task"])
    if num_trials > 1:
        base = base.merge(pd.DataFrame({"trial": range(1, num_trials + 1)}), how="cross")

    for c in STAGE_VALUE_COLUMNS[stage]:
        base[c] = None
    return base


def get_dataset_file_path(stage: str) -> tuple[Path, Path, Path | None]:
    """Return (uicrit_file, base64_screens_file, few_shot_samples_file); the last is None for critiques."""
    _check_stage(stage)
    few_shot = SCHEDULES_DIR / "few_shot_samples_df.parquet" if stage == "ratings" else None
    return PREPROCESSED_DIR / "uicrit_notna.parquet", PREPROCESSED_DIR / "screenshots.parquet", few_shot


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
    if task_subset not in VALID_TASK_SUBSETS:
        allowed = ", ".join(sorted(VALID_TASK_SUBSETS))
        raise ValueError(f"Unknown task_subset: {task_subset!r}. Expected one of: {allowed}")

    uicrit_data_file, _, few_shot_samples_file = get_dataset_file_path(stage)

    base_name = f"{stage}-{shots}-{task_subset}-{model_short}-responses"
    paths = {
        "results_jsonl": RESULTS_DIR / stage / "responses" / f"{base_name}.jsonl",
        "results_parquet": RESULTS_DIR / stage / "responses" / f"{base_name}.parquet",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    responses_df = load_existing_responses(paths["results_parquet"], uicrit_data_file, num_trials, stage)

    if task_subset != "all_tasks":
        # Keep the first task of each screen, then the first N of those.
        keep_count = int(task_subset.split("_")[0])
        first_tasks = responses_df.drop_duplicates(subset="screen_id", keep="first")
        keep_ids = set(first_tasks["screen_task_id"].head(keep_count))
        responses_df = responses_df[responses_df["screen_task_id"].isin(keep_ids)].reset_index(drop=True)

    if shots == "few_shot":
        # Exclude screens used as few-shot examples.
        few_shot_ids = set(pd.read_parquet(few_shot_samples_file)["screen_id"])
        responses_df = responses_df.loc[~responses_df["screen_id"].isin(few_shot_ids)].reset_index(drop=True)

    return responses_df, paths
