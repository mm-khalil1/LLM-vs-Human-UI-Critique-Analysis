"""
Shared LLM batch management utilities.

Main classes:
- BatchConfig: provider/model/runtime settings.
- BatchPrompt: normalized prompt before provider-specific formatting.
- BatchResult: normalized parsed response.
- ProviderAdapter: interface for provider-specific batch APIs.
- BatchManager: shared workflow for request files, batch execution, polling, and result extraction.
- OpenAIBatchAdapter / GeminiBatchAdapter / ClaudeBatchAdapter: provider implementations.

Files under BatchConfig.batch_main_dir:
- Batch_Requests/requests-<selected_tasks>-batch_<N>.jsonl: size-capped request files that get submitted.
- Batch_Responses/responses-<selected_tasks>-batch_<N>.jsonl: downloaded output of each request file.
- batch_jobs-<selected_tasks>.json: job manifest that lets process_batches_* resume after a restart.
- requests-<selected_tasks>.jsonl / responses-<selected_tasks>.jsonl: optional single-file copies
  written by generate_requests_jsonl / combine_response_files.

BatchManager methods:
- generate_request_batches_jsonl: build requests from experiment DataFrames straight into batch files.
- generate_requests_jsonl: same, but into the single requests file (then use split_requests_jsonl).
- write_request_batches_jsonl / write_requests_jsonl: write already-built provider requests.
- split_requests_jsonl: split the single requests file into batch files.
- preview_request: inspect one request, optionally displaying parameters/text/images in notebooks.
- build_retry_requests_from_failed_requests: create a retry JSONL containing only failed custom_ids.
- upload_file / create_batch / check_batch / retrieve_batch_output: manual single-batch workflow.
- process_batches_sequentially: submit, poll, and retrieve batch files one at a time.
- process_batches_in_parallel: submit all batch files first, then poll and retrieve them together.
- extract_results_from_responses: parse response files and update the responses DataFrame.
- inspect_response_errors: list provider errors from response files.
- combine_response_files: concatenate response files into one root-level file.

ProviderAdapter methods:
- build_request: convert BatchPrompt to provider request JSON.
- preview_request: extract custom_id, text, and images from provider request JSON.
- get_request_id: read custom_id/key from provider request or response JSON.
- upload_file: upload or prepare provider batch input.
- create_batch: start provider batch job.
- get_batch_id: extract provider batch id/name.
- get_batch_input_name: recover the original request batch filename.
- check_batch: normalize provider batch status.
- retrieve_output: save provider output locally.
- parse_result: normalize one provider response record.
- extract_error: return provider error details for one response record.
"""
from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


def extract_message_content_from_batch_record(record: dict) -> str:
    """Return the first text output of an OpenAI or Gemini batch response record."""
    response = record.get("response") or {}
    candidates = response.get("candidates") or []
    if candidates:
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return (parts[0].get("text") or "") if parts else ""

    choices = (response.get("body") or {}).get("choices") or []
    return (choices[0].get("message", {}).get("content") or "") if choices else ""


def extract_batch_record_id(record: dict) -> Optional[str]:
    custom_id = record.get("custom_id", record.get("key"))
    return None if custom_id is None else str(custom_id)


def _label_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, image in enumerate(images):
        image["label"] = "target image" if index == len(images) - 1 else f"sample image {index + 1}"
    return images


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class BatchConfig:
    """Shared configuration for provider-backed batch runs."""

    provider: str
    model: str
    batch_main_dir: Path
    selected_tasks: str
    temperature: float = 1.0
    max_tokens: int = 4096
    system_message: str = "You are a UI expert"
    image_format: str = "jpeg"
    polling_interval_minutes: int = 2


@dataclass(frozen=True)
class BatchPrompt:
    """Provider-neutral prompt payload before it is converted to JSONL."""

    custom_id: str
    user_message: str
    base64_screen: str
    sample_screens: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    """Provider-neutral result returned after parsing a batch response."""

    custom_id: str
    text: Optional[str]
    raw_response: dict[str, Any]
    error: Optional[str] = None


class ProviderAdapter(ABC):
    """Base contract for OpenAI, Gemini, Claude, and future batch providers."""

    provider: str

    def __init__(self, client: Any, config: BatchConfig):
        self.client = client
        self.config = config

    @abstractmethod
    def build_request(self, prompt: BatchPrompt) -> dict[str, Any]:
        """Convert a provider-neutral prompt into one JSONL request object."""

    @abstractmethod
    def get_request_id(self, request: dict[str, Any]) -> Optional[str]:
        """Return the custom id from one provider request or response object."""

    @abstractmethod
    def preview_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Extract notebook-friendly text and images from one provider request object."""

    @abstractmethod
    def upload_file(self, file_path: Path) -> Optional[str]:
        """Upload a JSONL prompt file and return the provider file id."""

    @abstractmethod
    def create_batch(self, input_file_id: str) -> Any:
        """Create a provider batch job and return the provider batch object."""

    @abstractmethod
    def get_batch_id(self, batch: Any) -> str:
        """Return the provider batch id/name from a batch object."""

    @abstractmethod
    def get_batch_input_name(self, batch: Any) -> str:
        """Return the original input prompt filename/display name for a batch."""

    @abstractmethod
    def check_batch(self, batch_id: str, verbose: bool = False) -> tuple[Optional[bool], Any]:
        """Return (completed, batch_obj).

        completed is True when done, False while running, and None on terminal failure.
        batch_obj is None when the provider could not be queried.
        """

    @abstractmethod
    def retrieve_output(self, batch_obj: Any, output_path: Path) -> Path:
        """Download the completed batch output into output_path."""

    @abstractmethod
    def parse_result(self, response: dict[str, Any]) -> BatchResult:
        """Convert one provider response record into a normalized result."""

    def extract_error(self, response: dict[str, Any]) -> Optional[Any]:
        """Return provider error details for a failed or empty response, else None."""
        return self.parse_result(response).error

    def list_uploaded_files(self, *args, **kwargs):
        raise NotImplementedError(f"{self.provider} does not implement list_uploaded_files.")

    def delete_file(self, file_id: str):
        raise NotImplementedError(f"{self.provider} does not implement delete_file.")

    def list_batches(self, *args, **kwargs):
        raise NotImplementedError(f"{self.provider} does not implement list_batches.")

    def cancel_batch(self, batch_id: str):
        raise NotImplementedError(f"{self.provider} does not implement cancel_batch.")

    def delete_batch(self, batch_id: str):
        raise NotImplementedError(f"{self.provider} does not implement delete_batch.")


class _JobManifest:
    """batch_jobs-<selected_tasks>.json: one job record per request batch file."""

    def __init__(self, path: Path):
        self.path = path
        self.jobs: list[dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                self.jobs = json.load(fh)
            if not isinstance(self.jobs, list):
                raise ValueError(f"Batch jobs manifest must contain a list: {path}")

    def job_for(self, request_file: Path, response_path: Path, batch_index: Optional[int]) -> dict[str, Any]:
        request_path = str(request_file.resolve())
        for job in self.jobs:
            if job.get("request_path") == request_path:
                return job

        job = {
            "batch_index": batch_index,
            "request_path": request_path,
            "request_file": request_file.name,
            "response_path": str(response_path.resolve()),
            "input_file_id": None,
            "batch_id": None,
            "status": "pending",
            "created_at": None,
            "retrieved_at": None,
            "error": None,
            "check_failures": 0,
        }
        self.jobs.append(job)
        return job

    def has_batch(self, batch_id: str) -> bool:
        return any(job.get("batch_id") == batch_id for job in self.jobs)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self.jobs, fh, ensure_ascii=False, indent=2)
            fh.write("\n")


class BatchManager:
    """Provider-agnostic orchestration shared by all batch providers."""

    def __init__(self, adapter: ProviderAdapter):
        self.adapter = adapter
        self.config = adapter.config

        self.root = Path(self.config.batch_main_dir).expanduser().resolve()
        self.requests_dir = self.root / "Batch_Requests"
        self.responses_dir = self.root / "Batch_Responses"
        self.requests_file = self.root / f"requests-{self.config.selected_tasks}.jsonl"

        self.root.mkdir(parents=True, exist_ok=True)

        self.batch_id: Optional[str] = None
        self.input_file_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Request files
    # ------------------------------------------------------------------

    def generate_request_batches_jsonl(
        self,
        schedule_file,
        responses_df,
        screens_df,
        *,
        guidelines: str | list[str] = "Nielsen Norman 10 Usability Heuristics",
        prompting_type: str = "zero_shot",
        samples_df=None,
        build_prompt_fn=None,
        max_mb_per_file: int = 180,
        clear_existing: bool = False,
    ) -> list[dict[str, Any]]:
        """Build provider requests from experiment DataFrames directly into numbered batch files."""
        requests = self._prompt_requests(
            schedule_file, responses_df, screens_df, guidelines, prompting_type, samples_df, build_prompt_fn
        )
        return self.write_request_batches_jsonl(
            requests,
            max_mb_per_file=max_mb_per_file,
            clear_existing=clear_existing,
        )

    def generate_requests_jsonl(
        self,
        schedule_file,
        responses_df,
        screens_df,
        *,
        guidelines: str | list[str] = "Nielsen Norman 10 Usability Heuristics",
        prompting_type: str = "zero_shot",
        samples_df=None,
        build_prompt_fn=None,
    ) -> Path:
        """Build provider requests from experiment DataFrames into the single requests file."""
        requests = self._prompt_requests(
            schedule_file, responses_df, screens_df, guidelines, prompting_type, samples_df, build_prompt_fn
        )
        return self.write_requests_jsonl(requests)

    def write_requests_jsonl(
        self,
        requests: Iterable[dict[str, Any]],
        output_file: Optional[Path] = None,
    ) -> Path:
        """Write provider-ready request records into one JSONL file (default: requests_file)."""
        output_path = Path(output_file or self.requests_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as fh:
            for request in requests:
                fh.write(json.dumps(request, ensure_ascii=False) + "\n")
                count += 1

        print(f"Wrote {count} request(s) to: {output_path}")
        return output_path

    def write_request_batches_jsonl(
        self,
        requests: Iterable[dict[str, Any]],
        *,
        max_mb_per_file: int = 180,
        output_stem: Optional[str] = None,
        clear_existing: bool = False,
    ) -> list[dict[str, Any]]:
        """Stream provider-ready request records into numbered, size-capped batch files.

        process_batches_* only submits files whose stem is config.selected_tasks, which is the default.
        """
        return self._write_batch_lines(
            (json.dumps(request, ensure_ascii=False) + "\n" for request in requests),
            max_mb_per_file=max_mb_per_file,
            output_stem=output_stem or self.config.selected_tasks,
            clear_existing=clear_existing,
        )

    def split_requests_jsonl(self, max_mb_per_file: int = 180) -> list[Path]:
        """Split requests_file into numbered, size-capped batch files, replacing earlier ones."""
        if not self.requests_file.exists():
            print(f"Requests file '{self.requests_file}' does not exist.")
            return []

        with self.requests_file.open("r", encoding="utf-8") as fh:
            manifest_rows = self._write_batch_lines(
                (line if line.endswith("\n") else line + "\n" for line in fh if line.strip()),
                max_mb_per_file=max_mb_per_file,
                output_stem=self.config.selected_tasks,
                clear_existing=True,
            )
        return [row["path"] for row in manifest_rows]

    def preview_request(
        self,
        index: int = 0,
        custom_id: Optional[str] = None,
        source_requests_file: Optional[Path] = None,
        display: bool = True,
    ) -> dict[str, Any]:
        """Preview one request by index or custom_id, with notebook display when available.

        source_requests_file may be one requests file or a directory of batch files; it defaults
        to whichever of requests_file and Batch_Requests/ was written last.
        """
        if index < 0:
            raise ValueError("index must be >= 0.")
        source = Path(source_requests_file) if source_requests_file is not None else self._default_request_source()

        for position, (request, loaded_from) in enumerate(self._iter_request_records(source)):
            if custom_id is None and position == index:
                break
            if custom_id is not None and self.adapter.get_request_id(request) == custom_id:
                break
        else:
            if custom_id is not None:
                raise ValueError(f"No prompt found for custom_id: {custom_id}")
            raise IndexError(f"No prompt found at index {index}.")

        preview = self.adapter.preview_request(request)
        preview["raw_request"] = request
        preview["source_file"] = str(loaded_from)
        if display:
            self._display_request_preview(preview)
        return preview

    def build_retry_requests_from_failed_requests(
        self,
        failed_custom_ids: Iterable[str],
        *,
        source_requests_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Write a retry JSONL containing only the requests whose custom_id failed.

        source_requests_file may be one requests file or a directory of batch files; it defaults
        to whichever of requests_file and Batch_Requests/ was written last.
        """
        failed_ids = {custom_id for custom_id in failed_custom_ids if custom_id}
        source_path = Path(source_requests_file) if source_requests_file else self._default_request_source()
        output_path = Path(output_file) if output_file else self.root / "requests-retry-failed.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        source_lines = 0
        matched_ids = set()
        records = self._iter_request_records(source_path)
        with output_path.open("w", encoding="utf-8") as f_out:
            for record, _ in records:
                source_lines += 1
                custom_id = self.adapter.get_request_id(record)
                if custom_id in failed_ids:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    matched_ids.add(custom_id)

        summary = {
            "source_file": str(source_path),
            "output_file": str(output_path),
            "source_lines": source_lines,
            "requested_failed_ids": len(failed_ids),
            "written": written,
            "missing_ids": sorted(failed_ids - matched_ids),
        }
        print(f"Wrote {written} retry request(s) to {output_path}. Missing from source: {len(summary['missing_ids'])}")
        return summary

    # ------------------------------------------------------------------
    # Manual single-batch workflow
    # ------------------------------------------------------------------

    def upload_file(self, file_path: Path) -> Optional[str]:
        self.input_file_id = self.adapter.upload_file(file_path)
        return self.input_file_id

    def create_batch(self, input_file_id: Optional[str] = None) -> Any:
        self.input_file_id = input_file_id or self.input_file_id
        if not self.input_file_id:
            print("No input_file_id available to create a batch.")
            return None
        batch = self.adapter.create_batch(self.input_file_id)
        self.batch_id = self.adapter.get_batch_id(batch) if batch else None
        return batch

    def check_batch(self, batch_id: Optional[str] = None, verbose: bool = False) -> tuple[Optional[bool], Any]:
        self.batch_id = batch_id or self.batch_id
        if not self.batch_id:
            if verbose:
                print("No batch_id.")
            return None, None
        return self.adapter.check_batch(self.batch_id, verbose=verbose)

    def retrieve_batch_output(
        self,
        batch_id: Optional[str] = None,
        base_name: Optional[str] = None,
        batch_obj: Any = None,
    ) -> Optional[Path]:
        """Download a completed batch to Batch_Responses/<base_name>[-batch_<N>].jsonl."""
        if batch_obj is None:
            state, batch_obj = self.check_batch(batch_id)
            if state is not True or batch_obj is None:
                print(f"Batch {self.batch_id} is not completed.")
                return None

        out_path = self._response_path(self.adapter.get_batch_input_name(batch_obj), base_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return self.adapter.retrieve_output(batch_obj, out_path)

    def _poll_then_retrieve(
        self,
        sleep_minutes: Optional[int] = None,
        base_name: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Optional[Path]:
        """Wait for one batch (default: the last one created) and download its output."""
        sleep_minutes = self.config.polling_interval_minutes if sleep_minutes is None else sleep_minutes
        while True:
            state, batch = self.check_batch(batch_id=batch_id)
            if state is True:
                return self.retrieve_batch_output(batch_obj=batch, base_name=base_name)
            if state is None:
                return None
            time.sleep(sleep_minutes * 60)

    def list_uploaded_files(self, *args, **kwargs):
        return self.adapter.list_uploaded_files(*args, **kwargs)

    def delete_file(self, file_id: str):
        return self.adapter.delete_file(file_id)

    def list_batches(self, *args, **kwargs):
        return self.adapter.list_batches(*args, **kwargs)

    def cancel_batch(self, batch_id: str):
        self.batch_id = batch_id
        return self.adapter.cancel_batch(batch_id)

    def delete_batch(self, batch_id: str):
        self.batch_id = batch_id
        return self.adapter.delete_batch(batch_id)

    # ------------------------------------------------------------------
    # Automatic workflow over all batch files
    # ------------------------------------------------------------------

    def process_batches_sequentially(
        self,
        start_from_batch: int,
        sleep_minutes: Optional[int] = None,
        base_name: Optional[str] = None,
        resume_batch_id: Optional[str] = None,
        manifest_path: Optional[Path | str] = None,
        max_check_failures: int = 3,
    ) -> list[dict[str, Any]]:
        """Submit, poll, and retrieve batch files one at a time, resuming from the jobs manifest.

        resume_batch_id (or the last batch created in this session) is waited on first when the
        manifest does not track it; the run then continues after that batch's file.
        """
        manifest = self._open_manifest(manifest_path, max_check_failures)
        sleep_minutes = self.config.polling_interval_minutes if sleep_minutes is None else sleep_minutes

        resume_batch_id = resume_batch_id or self.batch_id
        if resume_batch_id and not manifest.has_batch(resume_batch_id):
            output_path = self._poll_then_retrieve(sleep_minutes, base_name, batch_id=resume_batch_id)
            if not output_path:
                return manifest.jobs
            start_from_batch = (self._extract_batch_index(output_path.name) or start_from_batch) + 1
            print(f"Updating to start from batch {start_from_batch}.")

        for batch_file in self._batch_files_from(start_from_batch):
            job = manifest.job_for(batch_file, self._response_path(batch_file.name, base_name), self._extract_batch_index(batch_file.name))
            action = self._reconcile_job(job, batch_file, max_check_failures)
            manifest.save()
            if action == "unverified":
                return manifest.jobs

            if action == "submit":
                submitted = self._submit_job(job, batch_file)
                manifest.save()
                if not submitted:
                    return manifest.jobs

            if action != "done" and not self._wait_for_job(job, manifest, sleep_minutes, max_check_failures):
                return manifest.jobs

        return manifest.jobs

    def process_batches_in_parallel(
        self,
        start_from_batch: int,
        sleep_minutes: Optional[int] = None,
        base_name: Optional[str] = None,
        manifest_path: Optional[Path | str] = None,
        max_check_failures: int = 3,
    ) -> list[dict[str, Any]]:
        """Submit all batch files first, then poll and retrieve every unfinished job together."""
        manifest = self._open_manifest(manifest_path, max_check_failures)
        sleep_minutes = self.config.polling_interval_minutes if sleep_minutes is None else sleep_minutes

        pending = []
        for batch_file in self._batch_files_from(start_from_batch):
            job = manifest.job_for(batch_file, self._response_path(batch_file.name, base_name), self._extract_batch_index(batch_file.name))
            action = self._reconcile_job(job, batch_file, max_check_failures)
            if action == "submit" and self._submit_job(job, batch_file):
                action = "running"
            manifest.save()
            if action == "running" or (action == "unverified" and job["status"] != "check_failed"):
                pending.append(job)

        if not pending:
            print("No unfinished submitted batches to poll.")
            return manifest.jobs

        while True:
            pending = [job for job in pending if self._poll_job(job, max_check_failures) == "pending"]
            manifest.save()
            if not pending:
                break
            print(f"Waiting on {len(pending)} batch(es).")
            time.sleep(sleep_minutes * 60)

        return manifest.jobs

    def _open_manifest(self, manifest_path: Optional[Path | str], max_check_failures: int) -> _JobManifest:
        if max_check_failures <= 0:
            raise ValueError("max_check_failures must be positive.")
        return _JobManifest(Path(manifest_path or (self.root / f"batch_jobs-{self.config.selected_tasks}.json")))

    def _reconcile_job(self, job: dict[str, Any], batch_file: Path, max_check_failures: int) -> str:
        """Sync a job with its response file and provider state.

        Returns "done", "running", "submit" (never submitted, or the last attempt failed),
        or "unverified" (the provider could not be queried).
        """
        if self._is_nonempty_file(job["response_path"]):
            self._mark_retrieved(job, job.get("retrieved_at"))
            print(f"Already retrieved: {batch_file.name}")
            return "done"

        if not job.get("batch_id"):
            return "submit"

        state, batch = self.adapter.check_batch(job["batch_id"])
        if batch is None:
            self._record_check_failure(job, max_check_failures)
            print(f"Could not verify existing batch: {batch_file.name} | {job['batch_id']} | {job['status']}")
            return "unverified"

        job["status"] = self._batch_status(batch) or job["status"]
        if state is True:
            self._retrieve_job(job, batch)
            return "done"
        if state is False:
            job.update(error=None, check_failures=0)
            print(f"Already submitted: {batch_file.name} | {job['batch_id']} | {job['status']}")
            return "running"

        print(f"Resubmitting failed batch file: {batch_file.name} | previous {job['batch_id']} | {job['status']}")
        job["error"] = self._batch_error(batch)
        self._archive_batch_attempt(job)
        job["status"] = "retry_pending"
        return "submit"

    def _submit_job(self, job: dict[str, Any], batch_file: Path) -> bool:
        print(f"\nSubmitting batch file: {batch_file.name}")
        file_id = self.upload_file(batch_file)
        job["input_file_id"] = file_id
        if not file_id:
            job.update(status="upload_failed", error="upload_failed")
            return False

        batch = self.create_batch(input_file_id=file_id)
        if not batch:
            job.update(status="create_failed", error="create_failed")
            return False

        job.update(
            batch_id=self.adapter.get_batch_id(batch),
            status=self._batch_status(batch) or "submitted",
            created_at=self._batch_created_at(batch),
            error=None,
        )
        return True

    def _poll_job(self, job: dict[str, Any], max_check_failures: int) -> str:
        """Check a submitted job once and retrieve it when done: "retrieved", "pending", or "failed"."""
        state, batch = self.adapter.check_batch(job["batch_id"])
        if batch is None:
            self._record_check_failure(job, max_check_failures)
            return "failed" if job["status"] == "check_failed" else "pending"

        job["status"] = self._batch_status(batch) or job["status"]
        job["check_failures"] = 0
        if state is True:
            return "retrieved" if self._retrieve_job(job, batch) else "failed"
        if state is None:
            job["error"] = self._batch_error(batch)
            return "failed"
        return "pending"

    def _wait_for_job(self, job: dict[str, Any], manifest: _JobManifest, sleep_minutes: int, max_check_failures: int) -> bool:
        while True:
            outcome = self._poll_job(job, max_check_failures)
            manifest.save()
            if outcome != "pending":
                return outcome == "retrieved"
            time.sleep(sleep_minutes * 60)

    def _retrieve_job(self, job: dict[str, Any], batch: Any) -> bool:
        try:
            response_path = Path(job["response_path"])
            response_path.parent.mkdir(parents=True, exist_ok=True)
            self.adapter.retrieve_output(batch, response_path)
        except Exception as exc:
            job.update(status="retrieve_failed", error=f"{type(exc).__name__}: {exc}")
            return False
        self._mark_retrieved(job)
        return True

    @staticmethod
    def _mark_retrieved(job: dict[str, Any], retrieved_at: Optional[str] = None) -> None:
        job.update(status="retrieved", retrieved_at=retrieved_at or _now(), error=None, check_failures=0)

    @staticmethod
    def _record_check_failure(job: dict[str, Any], max_check_failures: int) -> None:
        job["check_failures"] = int(job.get("check_failures") or 0) + 1
        job["status"] = "check_failed" if job["check_failures"] >= max_check_failures else "check_retry"
        job["error"] = "check_batch returned no batch object"

    @staticmethod
    def _archive_batch_attempt(job: dict[str, Any]) -> None:
        job.setdefault("attempts", []).append(
            {
                "input_file_id": job.get("input_file_id"),
                "batch_id": job.get("batch_id"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
                "retrieved_at": job.get("retrieved_at"),
                "error": job.get("error"),
                "check_failures": job.get("check_failures", 0),
                "archived_at": _now(),
            }
        )
        job.update(input_file_id=None, batch_id=None, created_at=None, retrieved_at=None, check_failures=0)

    # ------------------------------------------------------------------
    # Response files
    # ------------------------------------------------------------------

    def extract_results_from_responses(
        self,
        responses_df,
        update_func,
        pattern: str = "responses*.jsonl",
    ) -> dict[str, Any]:
        """Read response JSONL files (retry files last), normalize provider responses, and update responses_df."""
        files = self._response_files(pattern)
        summary = {"files": len(files), "lines": 0, "updated": 0, "skipped": [], "errors": [], "error_details": []}

        for file, line_number, line in self._iter_response_lines(files):
            summary["lines"] += 1
            custom_id = ""
            try:
                result = self.adapter.parse_result(json.loads(line))
                custom_id = result.custom_id
                if result.error or not result.text:
                    summary["skipped"].append(custom_id)
                    continue

                screen_task_id, trial, aspect = self._parse_custom_id(custom_id)
                update_func(responses_df, screen_task_id, trial, aspect, result.text)
                summary["updated"] += 1
            except Exception as exc:
                summary["errors"].append(custom_id)
                summary["error_details"].append(
                    {
                        "file": file.name,
                        "line_number": line_number,
                        "custom_id": custom_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        return summary

    def inspect_response_errors(self, pattern: str = "responses*.jsonl") -> list[dict[str, Any]]:
        """Return requests whose latest response record is an error, with provider error data as-is.

        A successful record in a later file (retry files are read last) clears an earlier error.
        """
        files = self._response_files(pattern)
        latest: dict[Any, Optional[dict[str, Any]]] = {}
        for file, line_number, line in self._iter_response_lines(files):
            record = json.loads(line)
            custom_id = self.adapter.get_request_id(record)
            error = self.adapter.extract_error(record)
            key = (file.name, line_number) if custom_id is None else custom_id
            latest[key] = None if error is None else {
                "file": file.name,
                "line": line_number,
                "custom_id": custom_id,
                "error": error,
            }
        return [entry for entry in latest.values() if entry is not None]

    def combine_response_files(
        self,
        output_file: Optional[Path | str] = None,
        pattern: str = "responses*.jsonl",
    ) -> Path:
        """Concatenate response JSONL files into one file stored under the batch root."""
        output_path = Path(output_file or f"responses-{self.config.selected_tasks}.jsonl").expanduser()
        if not output_path.is_absolute():
            output_path = self.root / output_path
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        files = [p for p in sorted(self.responses_dir.glob(pattern)) if p.is_file() and p.resolve() != output_path]
        if not files:
            print(f"No response files matched pattern '{pattern}' in {self.responses_dir}")

        with output_path.open("w", encoding="utf-8") as out_f:
            for file in files:
                with file.open("r", encoding="utf-8", errors="ignore") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line if line.endswith("\n") else line + "\n")

        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prompt_requests(
        self,
        schedule_file,
        responses_df,
        screens_df,
        guidelines,
        prompting_type,
        samples_df,
        build_prompt_fn,
    ) -> Iterator[dict[str, Any]]:
        """Prepare prompt rows now and return a generator of provider requests."""
        import pandas as pd

        if build_prompt_fn is None:
            raise ValueError("build_prompt_fn is required.")

        schedule_df = pd.read_parquet(schedule_file)
        df = responses_df.merge(schedule_df, on="screen_task_id", how="left").merge(screens_df, on="screen_id", how="left")

        if prompting_type == "zero_shot":
            samples_df = None
        elif prompting_type == "few_shot" and samples_df is not None:
            df = df[~df["screen_id"].isin(samples_df["screen_id"])]

        df = self._drop_invalid_prompt_rows(df)
        sample_screens = self._sample_screens(samples_df)

        def iter_requests():
            for row in df.itertuples(index=False):
                aspect = getattr(row, "aspect", None)
                prompt_text = build_prompt_fn(
                    task=row.task,
                    guidelines=guidelines,
                    evaluation_aspect=aspect,
                    prompting_type=prompting_type,
                    samples_df=samples_df,
                )
                yield self.adapter.build_request(
                    BatchPrompt(
                        custom_id=row.custom_id,
                        user_message=prompt_text,
                        base64_screen=row.base64_screen,
                        sample_screens=sample_screens,
                        metadata={"aspect": aspect, "task": row.task},
                    )
                )

        return iter_requests()

    def _write_batch_lines(
        self,
        lines: Iterable[str],
        *,
        max_mb_per_file: int,
        output_stem: str,
        clear_existing: bool,
    ) -> list[dict[str, Any]]:
        """Write JSONL lines into requests-<output_stem>-batch_<N>.jsonl files capped by size."""
        if max_mb_per_file <= 0:
            raise ValueError("max_mb_per_file must be positive.")
        max_bytes_per_file = max_mb_per_file * 1024 * 1024
        self.requests_dir.mkdir(parents=True, exist_ok=True)

        if clear_existing:
            for old_path in self._request_batch_files(output_stem):
                old_path.unlink()

        manifest_rows: list[dict[str, Any]] = []
        outfile = None
        try:
            for line in lines:
                line_size = len(line.encode("utf-8"))
                if outfile is not None and manifest_rows[-1]["bytes"] + line_size > max_bytes_per_file:
                    outfile.close()
                    outfile = None

                if outfile is None:
                    batch_index = len(manifest_rows) + 1
                    path = self.requests_dir / f"requests-{output_stem}-batch_{batch_index}.jsonl"
                    outfile = path.open("w", encoding="utf-8")
                    manifest_rows.append({"batch_index": batch_index, "path": path, "requests": 0, "bytes": 0})

                outfile.write(line)
                manifest_rows[-1]["requests"] += 1
                manifest_rows[-1]["bytes"] += line_size
        finally:
            if outfile is not None:
                outfile.close()

        total_requests = sum(row["requests"] for row in manifest_rows)
        print(f"Wrote {total_requests} request(s) across {len(manifest_rows)} batch file(s) in: {self.requests_dir}")
        return manifest_rows

    def _request_batch_files(self, stem: Optional[str] = None, directory: Optional[Path] = None) -> list[Path]:
        """Return requests-<stem>-batch_<N>.jsonl files in batch order (stem defaults to selected_tasks)."""
        directory = directory or self.requests_dir
        if not directory.is_dir():
            return []
        pattern = re.compile(rf"requests-{re.escape(stem or self.config.selected_tasks)}-batch_(\d+)\.jsonl")
        matches = [(int(match.group(1)), path) for path in directory.iterdir() if (match := pattern.fullmatch(path.name))]
        return [path for _, path in sorted(matches)]

    def _batch_files_from(self, start_from_batch: int) -> list[Path]:
        files = [path for path in self._request_batch_files() if self._extract_batch_index(path.name) >= start_from_batch]
        if not files:
            print(f"No request batch files found in {self.requests_dir} starting from batch {start_from_batch}.")
        return files

    def _default_request_source(self) -> Path:
        """Return whichever of requests_file and Batch_Requests/ was written last."""
        batch_files = self._request_batch_files()
        if not batch_files:
            return self.requests_file
        if self.requests_file.exists() and self.requests_file.stat().st_mtime > max(p.stat().st_mtime for p in batch_files):
            return self.requests_file
        return self.requests_dir

    def _iter_request_records(self, source: Path) -> Iterator[tuple[dict[str, Any], Path]]:
        """Yield (request, file) from one requests file or from a directory of batch files."""
        if not source.exists():
            raise FileNotFoundError(f"Request file or directory not found: {source}")
        files = self._request_batch_files(directory=source) if source.is_dir() else [source]
        if not files:
            raise FileNotFoundError(f"No request batch files found in: {source}")

        for path in files:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line), path

    def _response_files(self, pattern: str) -> list[Path]:
        """Response files matching pattern, with responses-retry_* files last so retried records win."""
        return sorted(self.responses_dir.glob(pattern), key=lambda path: (path.name.startswith("responses-retry"), path.name))

    @staticmethod
    def _iter_response_lines(files: Iterable[Path]) -> Iterator[tuple[Path, int, str]]:
        for file in files:
            with file.open("r", encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    if line.strip():
                        yield file, line_number, line

    def _response_path(self, request_file_name: str | Path, base_name: Optional[str] = None) -> Path:
        base_name = base_name or f"responses-{self.config.selected_tasks}"
        batch_index = self._extract_batch_index(Path(request_file_name).name) if request_file_name else None
        suffix = f"-batch_{batch_index}" if batch_index is not None else ""
        return self.responses_dir / f"{base_name}{suffix}.jsonl"

    def _drop_invalid_prompt_rows(self, df):
        import pandas as pd

        def missing(column):
            return df[column].isna() if column in df.columns else pd.Series(True, index=df.index)

        missing_custom_id = missing("custom_id")
        missing_screen = missing("base64_screen")
        print(
            "Prompt preparation summary | "
            f"rows={len(df)} | "
            f"missing_custom_id={int(missing_custom_id.sum())} | "
            f"missing_aspect={int(missing('aspect').sum())} | "
            f"missing_base64_screen={int(missing_screen.sum())}"
        )

        valid_mask = ~missing_custom_id & ~missing_screen
        dropped_rows = int((~valid_mask).sum())
        if dropped_rows:
            preview_cols = [c for c in ["screen_task_id", "screen_id", "custom_id", "aspect"] if c in df.columns]
            print(f"Dropping {dropped_rows} row(s) before writing prompts due to missing required fields.")
            if preview_cols:
                print(df.loc[~valid_mask, preview_cols].head(10).to_string(index=False))

        return df.loc[valid_mask].reset_index(drop=True)

    @staticmethod
    def _display_request_preview(preview: dict[str, Any]) -> None:
        try:
            from IPython.display import Image, Markdown, display
        except ImportError:
            print(f"custom_id: {preview.get('custom_id')}")
            print(f"parameters: {json.dumps(preview.get('parameters', {}), indent=2)}")
            print(preview.get("text", ""))
            print(f"images: {len(preview.get('images', []))}")
            return

        display(Markdown(f"**custom_id:** `{preview.get('custom_id', '')}`"))
        display(Markdown("**parameters**"))
        display(Markdown(f"```json\n{json.dumps(preview.get('parameters', {}), indent=2)}\n```"))
        display(Markdown("**prompt**"))
        display(Markdown(preview.get("text", "")))
        for image in preview.get("images", []):
            display(Markdown(f"**{image.get('label', 'image')}**"))
            image_data = base64.b64decode(image["data"]) if isinstance(image["data"], str) else image["data"]
            display(Image(data=image_data, format=image.get("format")))

    @staticmethod
    def _sample_screens(samples_df) -> tuple[str, ...]:
        if samples_df is None:
            return ()
        return tuple(row.base64_screen for row in samples_df.itertuples(index=False))

    @staticmethod
    def _parse_custom_id(custom_id: str):
        parts = custom_id.split("&")
        screen_task_id = parts[0]
        aspect = parts[-1].replace("aspect=", "")
        trial = int(parts[1].replace("trial", "")) if len(parts) == 3 else None
        return screen_task_id, trial, aspect

    @staticmethod
    def _is_nonempty_file(path: Path | str | None) -> bool:
        return bool(path) and Path(path).is_file() and Path(path).stat().st_size > 0

    @staticmethod
    def _batch_status(batch: Any) -> Optional[str]:
        if batch is None:
            return None
        for attr in ("status", "processing_status"):
            value = getattr(batch, attr, None)
            if value is not None:
                return str(value)
        state_name = getattr(getattr(batch, "state", None), "name", None)
        return None if state_name is None else str(state_name)

    @staticmethod
    def _batch_created_at(batch: Any) -> Optional[str]:
        for attr in ("created_at", "create_time"):
            value = getattr(batch, attr, None)
            if value is not None:
                return str(BatchManager._convert_timestamp(value))
        return None

    @staticmethod
    def _batch_error(batch: Any) -> Optional[str]:
        for attr in ("errors", "error"):
            value = getattr(batch, attr, None)
            if value:
                return str(value)
        return None

    @staticmethod
    def _convert_timestamp(timestamp):
        if isinstance(timestamp, datetime):
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S") if timestamp else None

    @staticmethod
    def _extract_batch_index(filename: str) -> Optional[int]:
        match = re.search(r"(?:batch[_-]?)(\d+)(?:\.jsonl)?$", filename)
        return int(match.group(1)) if match else None


class OpenAIBatchAdapter(ProviderAdapter):
    """OpenAI Batch API adapter."""

    provider = "openai"

    def build_request(self, prompt: BatchPrompt) -> dict[str, Any]:
        content = [{"type": "text", "text": prompt.user_message}]
        for base64_screen in (*prompt.sample_screens, prompt.base64_screen):
            if base64_screen:
                content.append(self._image_content(base64_screen))

        return {
            "custom_id": prompt.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": self.config.system_message},
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_tokens,
            },
        }

    def get_request_id(self, request: dict[str, Any]) -> Optional[str]:
        return request.get("custom_id")

    def preview_request(self, request: dict[str, Any]) -> dict[str, Any]:
        body = request.get("body") or {}
        messages = body.get("messages") or []

        def message_content(role):
            return next((message.get("content") for message in messages if message.get("role") == role), None)

        user_content = message_content("user") or []
        text_parts = []
        images = []
        if isinstance(user_content, str):
            text_parts.append(user_content)
        else:
            for part in user_content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image = self._parse_data_url((part.get("image_url") or {}).get("url", ""))
                    if image:
                        images.append(image)

        parameters = {
            "provider": self.provider,
            "method": request.get("method"),
            "url": request.get("url"),
            "model": body.get("model"),
            "temperature": body.get("temperature"),
            "max_completion_tokens": body.get("max_completion_tokens"),
            "system_message": message_content("system"),
        }
        return {
            "custom_id": self.get_request_id(request),
            "parameters": parameters,
            "text": "\n".join(text_parts),
            "images": _label_images(images),
        }

    def upload_file(self, file_path: Path) -> Optional[str]:
        try:
            with Path(file_path).open("rb") as fh:
                response = self.client.files.create(file=fh, purpose="batch")
            print(f"Uploaded: {Path(file_path).name} -> ID: {response.id}")
            return response.id
        except Exception as exc:
            print(f"Upload failed for {file_path}: {exc}")
            return None

    def create_batch(self, input_file_id: str) -> Any:
        try:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"Batch created | ID: {batch.id} | Status: {batch.status}")
            return batch
        except Exception as exc:
            print(f"Failed to create batch: {exc}")
            return None

    def get_batch_id(self, batch: Any) -> str:
        return getattr(batch, "id")

    def get_batch_input_name(self, batch: Any) -> str:
        return self._retrieve_file_name(getattr(batch, "input_file_id", ""))

    def check_batch(self, batch_id: str, verbose: bool = False) -> tuple[Optional[bool], Any]:
        try:
            batch = self.client.batches.retrieve(batch_id)
        except Exception as exc:
            if verbose:
                print(f"retrieve: {exc}")
            return None, None

        status = getattr(batch, "status", None)
        if verbose:
            counts = getattr(batch, "request_counts", None)
            completed, failed, total = (counts.completed, counts.failed, counts.total) if counts else ("-", "-", "-")
            print(f"{batch_id} | {status} | completed={completed} failed={failed} total={total}")

        if status == "completed":
            return True, batch
        return (None if status in {"failed", "cancelled", "expired"} else False), batch

    def retrieve_output(self, batch_obj: Any, output_path: Path) -> Path:
        """Save the output file followed by the error file, so failed requests become error records."""
        file_ids = [getattr(batch_obj, attr, None) for attr in ("output_file_id", "error_file_id")]
        file_ids = [file_id for file_id in file_ids if file_id]
        if not file_ids:
            raise ValueError(f"No output or error file for batch {self.get_batch_id(batch_obj)}.")

        with output_path.open("wb") as fh:
            for file_id in file_ids:
                content = self.client.files.content(file_id).content
                if content:
                    fh.write(content if content.endswith(b"\n") else content + b"\n")
        print(f"Saved batch output to {output_path.name}")
        return output_path

    def parse_result(self, response: dict[str, Any]) -> BatchResult:
        custom_id = response.get("custom_id", "")
        resp = response.get("response") or {}
        if resp.get("status_code") != 200:
            return BatchResult(custom_id=custom_id, text=None, raw_response=response, error=str(resp.get("status_code")))

        text = self._message_text(resp)
        return BatchResult(custom_id=custom_id, text=text, raw_response=response, error=None if text else "empty_response")

    def extract_error(self, response: dict[str, Any]) -> Optional[Any]:
        resp = response.get("response") or {}
        if resp.get("status_code") != 200:
            return resp
        return None if self._message_text(resp) else {"type": "empty_response", "response": resp}

    def list_uploaded_files(self, purpose: str = "batch"):
        try:
            data = (self.client.files.list(purpose=purpose) if purpose else self.client.files.list()).data
            if not data:
                print("No uploaded files found.")
                return None

            print(f"Found {len(data)} file(s):\n")
            for file in data:
                created = BatchManager._convert_timestamp(file.created_at)
                print(f"ID: {file.id} | Purpose: {file.purpose:<10} | Created: {created} | Name: {file.filename}")
            return data
        except Exception as exc:
            print(f"Failed to list files: {exc}")
            return None

    def delete_file(self, file_id: str):
        try:
            response = self.client.files.delete(file_id)
            print(f"Deleted file ID: {file_id}")
            return response
        except Exception as exc:
            print(f"Failed to delete file '{file_id}': {exc}")
            return None

    def list_batches(self, limit: int = 10):
        try:
            data = self.client.batches.list(limit=limit).data
            if not data:
                print("No batches found.")
                return None

            print(f"Showing up to {limit} recent batches:\n")
            for batch in data:
                file_id = getattr(batch, "input_file_id", None)
                file_name = self._retrieve_file_name(file_id) if file_id else "N/A"
                created = BatchManager._convert_timestamp(batch.created_at)
                print(f"ID: {batch.id} | Status: {batch.status:<10} | Created: {created} | Input: {file_name}")
            return data
        except Exception as exc:
            print(f"Failed to list batches: {exc}")
            return None

    def cancel_batch(self, batch_id: str):
        try:
            batch = self.client.batches.cancel(batch_id)
            print(f"Batch ID: {batch.id} | New Status: {batch.status}")
            return batch
        except Exception as exc:
            print(f"Failed to cancel batch '{batch_id}': {exc}")
            return None

    def _retrieve_file_name(self, file_id: str) -> str:
        try:
            return self.client.files.retrieve(file_id).filename
        except Exception as exc:
            print(f"Failed to retrieve filename for ID '{file_id}': {exc}")
            return ""

    def _image_content(self, base64_screen: str) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/{self.config.image_format};base64,{base64_screen}"},
        }

    @staticmethod
    def _message_text(resp: dict[str, Any]) -> Optional[str]:
        choices = (resp.get("body") or {}).get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content")

    @staticmethod
    def _parse_data_url(url: str) -> Optional[dict[str, Any]]:
        match = re.match(r"data:image/([^;]+);base64,(.*)", url)
        return {"format": match.group(1), "data": match.group(2)} if match else None


class GeminiBatchAdapter(ProviderAdapter):
    """Google Gemini Batch API adapter."""

    provider = "google"

    _FAILED_STATES = {"JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

    def build_request(self, prompt: BatchPrompt) -> dict[str, Any]:
        parts = [{"text": prompt.user_message}]
        for base64_screen in (*prompt.sample_screens, prompt.base64_screen):
            if base64_screen:
                parts.append(self._image_part(base64_screen))

        return {
            "key": prompt.custom_id,
            "request": {
                "model": f"models/{self.config.model}",
                "contents": [{"role": "user", "parts": parts}],
                "systemInstruction": {
                    "role": "user",
                    "parts": [{"text": self.config.system_message}],
                },
                "generation_config": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            },
        }

    def get_request_id(self, request: dict[str, Any]) -> Optional[str]:
        return request.get("key")

    def preview_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_body = request.get("request") or {}
        contents = request_body.get("contents") or []
        parts = contents[0].get("parts", []) if contents else []
        text_parts = []
        images = []

        for part in parts:
            if "text" in part:
                text_parts.append(part.get("text", ""))
            elif "inlineData" in part:
                inline_data = part.get("inlineData") or {}
                images.append(
                    {
                        "format": inline_data.get("mimeType", "image/jpeg").replace("image/", ""),
                        "data": inline_data.get("data", ""),
                    }
                )

        system_parts = (request_body.get("systemInstruction") or {}).get("parts") or []
        parameters = {
            "provider": self.provider,
            "model": request_body.get("model"),
            "generation_config": request_body.get("generation_config", {}),
            "system_message": "\n".join(part.get("text", "") for part in system_parts if "text" in part),
        }
        return {
            "custom_id": self.get_request_id(request),
            "parameters": parameters,
            "text": "\n".join(text_parts),
            "images": _label_images(images),
        }

    def upload_file(self, file_path: Path) -> Optional[str]:
        try:
            from google.genai import types

            uploaded_file = self.client.files.upload(
                file=file_path,
                config=types.UploadFileConfig(display_name=Path(file_path).name, mime_type="jsonl"),
            )
            print(f"Uploaded: {Path(file_path).name} -> ID: {uploaded_file.name}")
            return uploaded_file.name
        except Exception as exc:
            print(f"Upload failed for {file_path}: {exc}")
            return None

    def create_batch(self, input_file_id: str) -> Any:
        try:
            batch = self.client.batches.create(
                model=self.config.model,
                src=input_file_id,
                config={"display_name": self._retrieve_file_name(input_file_id)},
            )
            print(f"Batch created | ID: {batch.name} | Status: {batch.state.name}")
            return batch
        except Exception as exc:
            print(f"Failed to create batch: {exc}")
            return None

    def get_batch_id(self, batch: Any) -> str:
        return getattr(batch, "name")

    def get_batch_input_name(self, batch: Any) -> str:
        return getattr(batch, "display_name", "")

    def check_batch(self, batch_id: str, verbose: bool = False) -> tuple[Optional[bool], Any]:
        try:
            batch = self.client.batches.get(name=batch_id)
        except Exception as exc:
            if verbose:
                print(f"retrieve: {exc}")
            return None, None

        status = batch.state.name
        if verbose:
            if status == "JOB_STATE_SUCCEEDED" or status in self._FAILED_STATES:
                print(f"Job finished with state: {status}")
                if status == "JOB_STATE_FAILED":
                    print(f"Error: {batch.error}")
            else:
                print(f"Current state: {status}")

        if status == "JOB_STATE_SUCCEEDED":
            return True, batch
        return (None if status in self._FAILED_STATES else False), batch

    def retrieve_output(self, batch_obj: Any, output_path: Path) -> Path:
        if not getattr(getattr(batch_obj, "dest", None), "file_name", None):
            raise ValueError(f"No output file for batch {self.get_batch_id(batch_obj)}.")

        file_content = self.client.files.download(file=batch_obj.dest.file_name)
        with output_path.open("wb") as fh:
            fh.write(file_content)
        print(f"Saved batch output to {output_path.name}")
        return output_path

    def parse_result(self, response: dict[str, Any]) -> BatchResult:
        text = self._response_text(response)
        return BatchResult(
            custom_id=response.get("key", ""),
            text=text or None,
            raw_response=response,
            error=None if text else "empty_response",
        )

    def extract_error(self, response: dict[str, Any]) -> Optional[Any]:
        if self._response_text(response):
            return None
        return response.get("error") or {"type": "empty_response", "response": response.get("response") or {}}

    def list_uploaded_files(self):
        try:
            files = self.client.files.list()
            if not files:
                print("No uploaded files found.")
                return None

            print(f"Found {len(files)} file(s):\n")
            for file in files:
                created = BatchManager._convert_timestamp(file.create_time)
                print(f"ID: {file.name} | Created: {created} | Name: {file.display_name}")
            return files
        except Exception as exc:
            print(f"Failed to list files: {exc}")
            return None

    def delete_file(self, file_id: str):
        try:
            response = self.client.files.delete(name=file_id)
            print(f"Deleted file ID: {file_id}")
            return response
        except Exception as exc:
            print(f"Failed to delete file '{file_id}': {exc}")
            return None

    def list_batches(self):
        try:
            batches = self.client.batches.list()
            if not batches:
                print("No batches found.")
                return None

            for batch in batches:
                created = BatchManager._convert_timestamp(batch.create_time)
                print(f"ID: {batch.name} | Status: {batch.state.name:<10} | Created: {created} | Input: {batch.display_name}")
            return batches
        except Exception as exc:
            print(f"Failed to list batches: {exc}")
            return None

    def cancel_batch(self, batch_id: str):
        try:
            batch = self.client.batches.cancel(name=batch_id)
            print("Cancelled")
            return batch
        except Exception as exc:
            print(f"Failed to cancel batch '{batch_id}': {exc}")
            return None

    def delete_batch(self, batch_id: str):
        try:
            batch = self.client.batches.delete(name=batch_id)
            print("Deleted")
            return batch
        except Exception as exc:
            print(f"Failed to delete batch '{batch_id}': {exc}")
            return None

    def _retrieve_file_name(self, file_id: str) -> str:
        try:
            return self.client.files.get(name=file_id).display_name
        except Exception as exc:
            print(f"Failed to retrieve filename for ID '{file_id}': {exc}")
            return ""

    def _image_part(self, base64_screen: str) -> dict[str, Any]:
        return {
            "inlineData": {
                "mimeType": f"image/{self.config.image_format}",
                "data": base64_screen,
            }
        }

    @staticmethod
    def _response_text(response: dict[str, Any]) -> str:
        candidates = (response.get("response") or {}).get("candidates") or []
        parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
        return (parts[0].get("text") or "") if parts else ""


class ClaudeBatchAdapter(ProviderAdapter):
    """Anthropic Message Batches adapter."""

    provider = "anthropic"

    def __init__(self, client: Any, config: BatchConfig):
        super().__init__(client, config)
        # Claude batches do not keep the input filename, so it is only known within this session.
        self._batch_input_names: dict[str, str] = {}

    def build_request(self, prompt: BatchPrompt) -> dict[str, Any]:
        content = [{"type": "text", "text": prompt.user_message}]
        for base64_screen in (*prompt.sample_screens, prompt.base64_screen):
            if base64_screen:
                content.append(self._image_content(base64_screen))

        return {
            "custom_id": prompt.custom_id,
            "params": {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.config.system_message,
                "messages": [{"role": "user", "content": content}],
            },
        }

    def get_request_id(self, request: dict[str, Any]) -> Optional[str]:
        return request.get("custom_id")

    def preview_request(self, request: dict[str, Any]) -> dict[str, Any]:
        params = request.get("params") or {}
        messages = params.get("messages") or []
        content = messages[0].get("content", []) if messages else []
        text_parts = []
        images = []

        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image":
                source = part.get("source") or {}
                images.append(
                    {
                        "format": source.get("media_type", "image/jpeg").replace("image/", ""),
                        "data": source.get("data", ""),
                    }
                )

        parameters = {
            "provider": self.provider,
            "model": params.get("model"),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "system_message": params.get("system"),
        }
        return {
            "custom_id": self.get_request_id(request),
            "parameters": parameters,
            "text": "\n".join(text_parts),
            "images": _label_images(images),
        }

    def upload_file(self, file_path: Path) -> str:
        """Claude batches take requests inline, so the local path stands in for a file id."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        print(f"Prepared Claude batch input: {file_path.name}")
        return str(file_path)

    def create_batch(self, input_file_id: str) -> Any:
        input_path = Path(input_file_id)
        try:
            batch = self.client.messages.batches.create(requests=self._read_requests(input_path))
        except Exception as exc:
            print(f"Failed to create batch: {exc}")
            return None

        batch_id = self.get_batch_id(batch)
        self._batch_input_names[batch_id] = input_path.name
        print(f"Batch created | ID: {batch_id} | Status: {getattr(batch, 'processing_status', None)}")
        return batch

    def get_batch_id(self, batch: Any) -> str:
        return getattr(batch, "id")

    def get_batch_input_name(self, batch: Any) -> str:
        return self._batch_input_names.get(self.get_batch_id(batch), "")

    def check_batch(self, batch_id: str, verbose: bool = False) -> tuple[Optional[bool], Any]:
        try:
            batch = self.client.messages.batches.retrieve(batch_id)
        except Exception as exc:
            if verbose:
                print(f"retrieve: {exc}")
            return None, None

        status = getattr(batch, "processing_status", None)
        if verbose:
            print(f"{batch_id} | {status} | request_counts={getattr(batch, 'request_counts', None)}")

        if status == "ended":
            return True, batch
        return (False if status in {"in_progress", "canceling"} else None), batch

    def retrieve_output(self, batch_obj: Any, output_path: Path) -> Path:
        batch_id = self.get_batch_id(batch_obj)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for result in self.client.messages.batches.results(batch_id):
                fh.write(json.dumps(self._to_dict(result), ensure_ascii=False) + "\n")

        print(f"Saved batch output to {output_path.name}")
        return output_path

    def parse_result(self, response: dict[str, Any]) -> BatchResult:
        custom_id = response.get("custom_id", "")
        result = response.get("result") or {}
        if result.get("type") != "succeeded":
            error = result.get("error") or result.get("type") or "unknown_error"
            return BatchResult(custom_id=custom_id, text=None, raw_response=response, error=str(error))

        return BatchResult(custom_id=custom_id, text=self._result_text(result) or None, raw_response=response)

    def extract_error(self, response: dict[str, Any]) -> Optional[Any]:
        result = response.get("result") or {}
        if result.get("type") == "succeeded":
            return None if self._result_text(result) else {"type": "empty_response", "result": result}
        return result.get("error") if "error" in result else result

    def list_batches(self, limit: int = 10):
        try:
            batches = self.client.messages.batches.list(limit=limit)
            data = getattr(batches, "data", batches)
            for batch in data:
                created = BatchManager._convert_timestamp(getattr(batch, "created_at", None))
                print(f"ID: {batch.id} | Status: {batch.processing_status:<12} | Created: {created}")
            return data
        except Exception as exc:
            print(f"Failed to list Claude batches: {exc}")
            return None

    def cancel_batch(self, batch_id: str):
        try:
            batch = self.client.messages.batches.cancel(batch_id)
            print(f"Batch ID: {batch.id} | New Status: {batch.processing_status}")
            return batch
        except Exception as exc:
            print(f"Failed to cancel batch '{batch_id}': {exc}")
            return None

    def _image_content(self, base64_screen: str) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{self.config.image_format}",
                "data": base64_screen,
            },
        }

    @staticmethod
    def _result_text(result: dict[str, Any]) -> str:
        content = (result.get("message") or {}).get("content") or []
        return "\n".join(part.get("text", "") for part in content if part.get("type") == "text").strip()

    @staticmethod
    def _read_requests(input_path: Path) -> list[dict[str, Any]]:
        with input_path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        for method in ("to_dict", "model_dump", "dict"):
            if hasattr(value, method):
                return getattr(value, method)()
        raise TypeError(f"Cannot convert {type(value).__name__} to dict.")


def build_batch_adapter(client: Any, config: BatchConfig) -> ProviderAdapter:
    """Return the provider adapter for a batch config."""
    provider = "gemini" if config.provider == "google" else config.provider
    if provider in {"openai", "openrouter", "deepseek", "alibaba"}:
        return OpenAIBatchAdapter(client, config)
    if provider == "gemini":
        return GeminiBatchAdapter(client, config)
    if provider in {"anthropic", "claude"}:
        return ClaudeBatchAdapter(client, config)
    raise ValueError(f"Unsupported batch provider: {config.provider}")
