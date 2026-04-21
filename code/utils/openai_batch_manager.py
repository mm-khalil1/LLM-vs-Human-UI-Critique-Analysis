import re
import json
import time
import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime
from openai import OpenAI
from openai.types import Batch


class OpenAIBatchManager:
    """Manage OpenAI batch operations for ratings or critiques runs."""

    def __init__(self, client: OpenAI, batch_main_dir: Path, selected_tasks: str):
        self.client = client
        self.root = Path(batch_main_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        self.prompts_file = self.root / f"prompts-{selected_tasks}.jsonl"

        self.prompts_dir = self.root / "Batch_Prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self.responses_dir = self.root / "Batch_Responses"
        self.responses_dir.mkdir(parents=True, exist_ok=True)

        self.batch_id: Optional[str] = None
        self.input_file_id: Optional[str] = None

    # -------------------- Preparing Prompt Files --------------------
    def _construct_request_jsonl(
        self,
        *,
        custom_id: str,
        model: str,
        system_message: str,
        user_message: str,
        base64_screen: str,
        temperature: float,
        max_tokens: int,
        samples_df=None,
        image_format: str = "jpeg",
    ) -> dict:
        """Return one JSONL-ready dict for /v1/chat/completions."""
        content = [{"type": "text", "text": user_message}]

        if samples_df is not None:
            for _, s in samples_df.iterrows():
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{image_format};base64,{s['base64_screen']}"},
                    }
                )

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{base64_screen}"},
            }
        )

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": content},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
            },
        }

    def generate_prompts_jsonl(
        self,
        schedule_file,
        responses_df,
        screens_df,
        *,
        model_version: str,
        system_message: str = "You are a UI expert",
        guidelines: str | list[str] = "Nielsen Norman 10 Usability Heuristics",
        temperature: float = 1.0,
        max_tokens: int = 4096,
        prompting_type: str = "zero_shot",
        samples_df=None,
        image_format: str = "jpeg",
        build_prompt_fn=None,
    ):
        """
        Build a single JSONL file of requests for ratings or critiques inference.

        Validates the schedule join before writing so missing `custom_id` rows are
        reported explicitly instead of silently producing unusable prompt entries.
        """
        schedule_df = pd.read_parquet(schedule_file)

        df = (
            responses_df[["screen_task_id", "screen_id", "task"]]
            .merge(schedule_df, on="screen_task_id", how="left")
            .merge(screens_df, on="screen_id", how="left")
        )
        if prompting_type == "zero_shot":
            samples_df = None
        elif prompting_type == "few_shot" and samples_df is not None:
            df = df[~df["screen_id"].isin(samples_df["screen_id"])]

        total_rows = len(df)
        missing_custom_id_mask = df["custom_id"].isna() if "custom_id" in df.columns else pd.Series(True, index=df.index)
        missing_aspect_mask = df["aspect"].isna() if "aspect" in df.columns else pd.Series(True, index=df.index)
        missing_screen_mask = df["base64_screen"].isna() if "base64_screen" in df.columns else pd.Series(True, index=df.index)

        print(
            "Prompt preparation summary | "
            f"rows={total_rows} | "
            f"missing_custom_id={int(missing_custom_id_mask.sum())} | "
            f"missing_aspect={int(missing_aspect_mask.sum())} | "
            f"missing_base64_screen={int(missing_screen_mask.sum())}"
        )

        valid_mask = ~missing_custom_id_mask & ~missing_screen_mask
        dropped_rows = int((~valid_mask).sum())
        if dropped_rows:
            preview_cols = [c for c in ["screen_task_id", "screen_id", "custom_id", "aspect"] if c in df.columns]
            print(f"Dropping {dropped_rows} row(s) before writing prompts due to missing required fields.")
            if preview_cols:
                print(df.loc[~valid_mask, preview_cols].head(10).to_string(index=False))
        df = df.loc[valid_mask].reset_index(drop=True)

        with self.prompts_file.open("w", encoding="utf-8") as fh:
            for row in df.itertuples(index=False):
                if "aspect" not in row._fields:
                    print("'aspect' column missing in schedule file.")
                    aspect = None
                else:
                    aspect = row.aspect

                prompt = build_prompt_fn(
                    task=row.task,
                    guidelines=guidelines,
                    evaluation_aspect=aspect,
                    prompting_type=prompting_type,
                    samples_df=samples_df,
                )

                obj = self._construct_request_jsonl(
                    custom_id=row.custom_id,
                    model=model_version,
                    system_message=system_message,
                    user_message=prompt,
                    base64_screen=row.base64_screen,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    samples_df=samples_df,
                    image_format=image_format,
                )

                fh.write(json.dumps(obj) + "\n")
        print(f"Wrote {len(df)} prompt(s) JSONL to: {self.prompts_file}")
        return self.prompts_file

    def split_prompts_jsonl(self, max_lines_per_file: int = 30):
        """Split a large JSONL file into smaller chunks with <= `max_lines_per_file` lines each."""
        if not self.prompts_file.exists():
            print(f"Prompts file '{self.prompts_file}' does not exist.")
            return

        with open(self.prompts_file, "r", encoding="utf-8") as infile:
            file_index = 1
            current_path = self.prompts_dir / f"prompts-batch_{file_index}.jsonl"
            outfile = open(current_path, "w", encoding="utf-8")

            for line_number, line in enumerate(infile, start=1):
                outfile.write(line)
                if line_number % max_lines_per_file == 0:
                    outfile.close()
                    file_index += 1
                    current_path = self.prompts_dir / f"prompts-batch_{file_index}.jsonl"
                    outfile = open(current_path, "w", encoding="utf-8")
            outfile.close()

    def build_retry_prompts_from_failed_requests(
        self,
        failed_custom_ids: Iterable[str],
        *,
        source_prompts_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict:
        """
        Build a retry prompts JSONL containing only requests whose `custom_id` failed.

        This is useful for retrying ratings or critiques requests after transient
        failures such as quota exhaustion.
        """
        failed_ids = {cid for cid in failed_custom_ids if cid}
        source_path = Path(source_prompts_file or self.prompts_file)
        output_path = Path(output_file or (source_path.parent / "prompts-retry-failed.jsonl"))

        if not source_path.exists():
            raise FileNotFoundError(f"Source prompts file not found: {source_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        source_lines = 0
        matched_ids = set()
        with source_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                source_lines += 1
                record = json.loads(line)
                custom_id = record.get("custom_id")
                if custom_id in failed_ids:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    matched_ids.add(custom_id)

        missing_ids = sorted(failed_ids - matched_ids)
        summary = {
            "source_file": str(source_path),
            "output_file": str(output_path),
            "source_lines": source_lines,
            "requested_failed_ids": len(failed_ids),
            "written": written,
            "missing_ids": missing_ids,
        }
        print(f"Wrote {written} retry prompt(s) to {output_path}. Missing from source: {len(missing_ids)}")
        return summary

    # -------------------- Batch Processing --------------------
    def upload_file(self, file_path: Path):
        """Upload a JSONL file to OpenAI and return its file ID."""
        try:
            with open(file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose="batch")
            print(f"Uploaded: {file_path.name} -> ID: {response.id}")
            self.input_file_id = response.id
            return self.input_file_id

        except Exception as e:
            print(f"Upload failed for {file_path}: {e}")
            return None

    def list_uploaded_files(self, purpose: str = "batch"):
        """List uploaded files with purpose, creation date, and name."""
        try:
            files = self.client.files.list(purpose=purpose) if purpose else self.client.files.list()
            data = files.data
            if not data:
                print("No uploaded files found.")
                return None

            print(f"Found {len(data)} file(s):\n")
            for f in data:
                created = self._convert_timestamp(f.created_at)
                print(f"ID: {f.id} | Purpose: {f.purpose:<10} | Created: {created} | Name: {f.filename}")
            return data

        except Exception as e:
            print(f"Failed to list files: {e}")
            return None

    def delete_file(self, file_id: str):
        """Delete a file by its ID and confirm deletion."""
        try:
            res = self.client.files.delete(file_id)
            print(f"Deleted file ID: {file_id}")
            return res
        except Exception as e:
            print(f"Failed to delete file '{file_id}': {e}")
            return None

    def create_batch(self, input_file_id: str | None = None) -> Optional[Batch]:
        """Create a batch job for the given input file and return the batch object."""
        self.input_file_id = input_file_id or self.input_file_id
        if not self.input_file_id:
            print("No input_file_id available to create a batch.")
            return None
        try:
            batch = self.client.batches.create(
                input_file_id=self.input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"Batch created | ID: {batch.id} | Status: {batch.status}")
            self.batch_id = batch.id
            return batch

        except Exception as e:
            print(f"Failed to create batch: {e}")
            return None

    def check_batch(self, batch_id: str | None = None, verbose: bool = False):
        """True: completed, False: running, None: failed/unavailable."""
        self.batch_id = batch_id or self.batch_id
        if not self.batch_id:
            if verbose:
                print("No batch_id.")
            return None, None

        try:
            batch = self.client.batches.retrieve(self.batch_id)
            st = getattr(batch, "status", None)
            if verbose:
                rc = getattr(batch, "request_counts", None)
                c = f = t = "-"
                if rc:
                    c, f, t = rc.completed, rc.failed, rc.total
                print(f"{self.batch_id} | {st} | completed={c} failed={f} total={t}")
                event_names = ["created_at", "expires_at", "completed_at", "expired_at", "failed_at", "cancelled_at"]
                for event_name in event_names:
                    self._print_label_and_timestamp(
                        f"Batch {event_name.replace('_', ' ')}", getattr(batch, event_name)
                    )
            state = True if st == "completed" else (None if st in {"failed", "cancelled", "expired"} else False)
            return state, batch

        except Exception as e:
            if verbose:
                print(f"retrieve: {e}")
            return None, None

    def list_batches(self, limit: int = 10):
        """List recent batches with status, creation time, and input file name."""
        try:
            batches = self.client.batches.list(limit=limit)
            data = batches.data
            if not data:
                print("No batches found.")
                return None

            print(f"Showing up to {limit} recent batches:\n")
            for b in data:
                file_id = getattr(b, "input_file_id", None)
                file_name = self._retrieve_file_name(file_id) if file_id else "N/A"
                created = self._convert_timestamp(b.created_at)
                print(f"ID: {b.id} | Status: {b.status:<10} | Created: {created} | Input: {file_name}")

            return data

        except Exception as e:
            print(f"Failed to list batches: {e}")
            return None

    def cancel_batch(self, batch_id: str):
        """Cancel a running batch by ID and show the result."""
        self.batch_id = batch_id

        try:
            batch = self.client.batches.cancel(batch_id)
            print(f"Batch ID: {batch.id} | New Status: {batch.status}")

            if getattr(batch, "errors", None):
                print(f"Errors: {batch.errors}")
            if getattr(batch, "cancelled_at", None):
                print(f"Cancelled at: {self._convert_timestamp(batch.cancelled_at)}")

            return batch

        except Exception as e:
            print(f"Failed to cancel batch '{batch_id}': {e}")
            return None

    def retrieve_batch_output(
        self,
        batch_id: Optional[str] = None,
        base_name: str = "responses",
        batch_obj=None,
    ) -> Optional[Path]:
        """Retrieve and save a completed batch output file for ratings or critiques."""
        self.batch_id = batch_id or self.batch_id
        if not self.batch_id:
            print("No batch_id available to retrieve output.")
            return None

        try:
            batch = batch_obj or self.client.batches.retrieve(self.batch_id)
            if batch.status != "completed":
                print(f"Batch {self.batch_id} not completed yet (status: {batch.status})")
                return None
            if not getattr(batch, "output_file_id", None):
                print(f"No output file for batch {self.batch_id}.")
                return None

            input_file_name = self._retrieve_file_name(getattr(batch, "input_file_id", ""))
            out_path = self._prepare_response_output_path(input_file_name, base_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            output_file_id = getattr(batch, "output_file_id", "")
            content = self.client.files.content(output_file_id)
            content.write_to_file(out_path)
            print(f"Saved batch output to {out_path.name}")
            return out_path

        except Exception as e:
            print(f"Failed to retrieve output for batch '{self.batch_id}': {e}")
            return None

    # -------------------- Main Automatic Batch Processing --------------------
    def _poll_then_retrieve(self, sleep_minutes: int, base_name: str, batch_id: Optional[str] = None):
        while True:
            state, batch = self.check_batch(batch_id=batch_id, verbose=False)
            if state is True:
                break
            if state is None:
                return None
            time.sleep(sleep_minutes * 60)
        return self.retrieve_batch_output(batch_obj=batch, base_name=base_name)

    def process_batches_from(
        self,
        start_from_batch: int,
        sleep_minutes: int = 2,
        base_name: str = "responses",
        resume_batch_id: Optional[str] = None,
    ):
        """Process all batch files starting from a specific batch number."""
        if resume_batch_id or self.batch_id:
            if resume_batch_id is not None:
                self.batch_id = resume_batch_id
            input_file_id = getattr(self.client.batches.retrieve(self.batch_id), "input_file_id", "")
            input_file_name = self._retrieve_file_name(input_file_id)
            k = self._extract_batch_index(input_file_name)
            print(f"Continuing from last batch ID: {self.batch_id} | batch {k}")
            output_path = self._poll_then_retrieve(sleep_minutes, base_name)
            if not output_path:
                return
            start_from_batch = k + 1
            print(f"Updating to start from batch {start_from_batch}.")

        batch_files = self._filter_files_by_batch_num(start_from_batch)
        for batch_file in batch_files:
            print(f"\nProcessing batch file: {batch_file.name}")

            file_id = self.upload_file(batch_file)
            if not file_id:
                return

            batch = self.create_batch(input_file_id=file_id)
            if not batch:
                return

            output_path = self._poll_then_retrieve(sleep_minutes, base_name)
            if not output_path:
                return

    # -------------------- Post-processing --------------------
    def extract_results_from_responses(
        self,
        responses_df,
        update_func,
        pattern: str = "responses*.jsonl",
    ) -> dict:
        """
        Read response JSONL files for ratings or critiques and update `responses_df`.

        Returns: {'files', 'lines', 'updated', 'skipped': [custom_id...], 'errors': [custom_id...]}.
        """
        files = sorted(self.responses_dir.glob(pattern), key=lambda p: p.name)
        summary = {"files": len(files), "lines": 0, "updated": 0, "skipped": [], "errors": []}

        for file in files:
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    summary["lines"] += 1
                    custom_id = None
                    try:
                        data = json.loads(line)
                        custom_id = data.get("custom_id", "")
                        resp = data.get("response", {})
                        if resp.get("status_code") != 200:
                            summary["skipped"].append(custom_id)
                            continue
                        body = resp.get("body") or {}
                        msg = (body.get("choices") or [{}])[0].get("message", {})
                        content = msg.get("content")
                        if not content:
                            summary["skipped"].append(custom_id)
                            continue

                        screen_task_id, trial, aspect = self._parse_custom_id(custom_id)
                        update_func(responses_df, screen_task_id, trial, aspect, content)
                        summary["updated"] += 1
                    except Exception:
                        summary["errors"].append(custom_id or "")
        return summary

    # -------------------- Helpers --------------------
    @staticmethod
    def _parse_custom_id(custom_id):
        """Extract screen_task_id, trial, and aspect from the custom_id."""
        parts = custom_id.split("&")
        screen_task_id = parts[0]
        aspect = parts[-1].replace("aspect=", "")
        if len(parts) == 3:
            trial = int(parts[1].replace("trial", ""))
        else:
            trial = None
        return screen_task_id, trial, aspect

    def _retrieve_file_name(self, file_id: str) -> str:
        """Retrieve the filename associated with a given file ID."""
        try:
            file_info = self.client.files.retrieve(file_id)
            return file_info.filename
        except Exception as e:
            print(f"Failed to retrieve filename for ID '{file_id}': {e}")
            return ""

    def _prepare_response_output_path(
        self,
        input_file_name: str | Path,
        base_name: str = "responses",
    ) -> Path:
        """Build response path as '{base_name}-batch_{k}.jsonl' if batch index exists."""
        input_file_name = Path(input_file_name).name if input_file_name else ""
        k = self._extract_batch_index(input_file_name)
        out_name = f"{base_name}-batch_{k}.jsonl" if k is not None else f"{base_name}.jsonl"
        return self.responses_dir / out_name

    def _filter_files_by_batch_num(self, start_from_batch):
        files = []
        for f in self.prompts_dir.iterdir():
            if f.is_file():
                idx = self._extract_batch_index(f.name)
                if idx is not None and idx >= start_from_batch:
                    files.append((idx, f))
        if not files:
            print(f"No batch files found starting from batch {start_from_batch}.")
            return []
        return [f for _, f in sorted(files, key=lambda t: t[0])]

    @staticmethod
    def _convert_timestamp(timestamp):
        """Convert a Unix timestamp to a readable datetime string."""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else None

    @staticmethod
    def _extract_batch_index(filename: str) -> Optional[int]:
        """Return batch index from name like 'prompts_batch_003.jsonl' -> 3."""
        m = re.search(r"(?:batch[_-]?)(\d+)(?:\.jsonl)?$", filename)
        return int(m.group(1)) if m else None

    def _print_label_and_timestamp(self, label, timestamp):
        """Print the provided label and timestamp in a human-readable format."""
        print(label + ":", self._convert_timestamp(timestamp)) if timestamp is not None else None
