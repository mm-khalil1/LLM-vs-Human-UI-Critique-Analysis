"""Project directory layout. Every notebook and helper resolves paths from here."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # original UICrit + RICO downloads; never written to
SCHEDULES_DIR = DATA_DIR / "schedules"  # schedules and few-shot samples used by the inference notebooks

# One folder per notebook in notebooks/1_data, holding its outputs and the manual inputs it reads.
PROCESSED_DIR = DATA_DIR / "processed"
PREPROCESSED_DIR = PROCESSED_DIR / "1_preprocessed"      # cleaned UICrit, screenshots
PARSED_DIR = PROCESSED_DIR / "2_parsed"                  # structured comments + manual parsing corrections
HUMAN_REFERENCE_DIR = PROCESSED_DIR / "3_human_reference"  # deduplicated expert comments
SCREENS_DIR = PREPROCESSED_DIR / "used_dataset_screens"

RESULTS_DIR = PROJECT_ROOT / "results"  # <stage>/{inference_batches,responses,matching,analysis}; archive/ for superseded runs
