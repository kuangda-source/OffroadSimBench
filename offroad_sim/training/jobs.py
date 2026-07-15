"""Persistent subprocess jobs for local model training."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO


TRAINING_JOB_FILENAME = "training_job.json"
FINAL_JOB_STATUSES = {"completed", "failed", "canceled"}
_PROGRESS_PATTERN = re.compile(r"(?:epoch|step)?\s*(\d+)\s*[/\\]\s*(\d+)", re.IGNORECASE)


class ProcessTrainingJob:
    """One cancellable trainer process with streamed logs and persisted state."""

    def __init__(
        self,
        *,
        command: list[str],
        working_directory: Path,
        environment: dict[str, str],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
        finalizer: Callable[["ProcessTrainingJob"], dict[str, Any]] | None = None,
        job_id: str = "",
    ) -> None:
        self.job_id = job_id or f"train_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.command = [str(value) for value in command]
        self.working_directory = working_directory.resolve()
        self.environment = dict(environment)
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_path = self.output_dir / "stdout.log"
        self.stderr_path = self.output_dir / "stderr.log"
        self.state_path = self.output_dir / TRAINING_JOB_FILENAME
        self.metadata = dict(metadata or {})
        self.finalizer = finalizer
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._cancel_requested = False
        self._process: subprocess.Popen[str] | None = None
        self._state: dict[str, Any] = {
            "job_id": self.job_id,
            "status": "queued",
            "progress": 0.0,
            "current_step": None,
            "total_steps": None,
            "message": "Queued",
            "command": self.command,
            "working_directory": str(self.working_directory),
            "output_dir": str(self.output_dir),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "state_path": str(self.state_path),
            "pid": None,
            "return_code": None,
            "created_at": _timestamp(),
            "started_at": None,
            "finished_at": None,
            "error": "",
            "result": {},
            "metadata": self.metadata,
        }
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def wait(self, timeout: float | None = None) -> dict[str, Any]:
        self._done.wait(timeout)
        return self.snapshot()

    def cancel(self) -> bool:
        queued_cancel = False
        with self._lock:
            status = str(self._state["status"])
            if status in FINAL_JOB_STATUSES:
                return False
            self._cancel_requested = True
            process = self._process
            if status == "queued":
                self._state["status"] = "canceling"
                self._state["message"] = "Canceling"
                self._persist_locked()
                queued_cancel = True
            else:
                self._state["status"] = "canceling"
                self._state["message"] = "Canceling"
                self._persist_locked()
        if queued_cancel:
            result = self.finalizer(self) if self.finalizer is not None else {}
            self._finish(status="canceled", message="Canceled before start", result=result)
            return True
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        return True

    def _start(self) -> None:
        with self._lock:
            if self._cancel_requested or self._state["status"] == "canceled":
                return
            self._state.update(status="running", message="Running", started_at=_timestamp())
            self._persist_locked()
        thread = threading.Thread(target=self._run, name=f"training-job-{self.job_id}", daemon=True)
        thread.start()

    def _run(self) -> None:
        stdout_file = self.stdout_path.open("w", encoding="utf-8", buffering=1)
        stderr_file = self.stderr_path.open("w", encoding="utf-8", buffering=1)
        finalizer_called = False
        try:
            process_options: dict[str, Any] = {}
            if os.name == "nt":
                process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True
            process = subprocess.Popen(
                self.command,
                cwd=self.working_directory,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **process_options,
            )
            with self._lock:
                self._process = process
                self._state["pid"] = process.pid
                self._persist_locked()
                cancel_after_spawn = self._cancel_requested
            if cancel_after_spawn and process.poll() is None:
                _terminate_process_tree(process)
            stdout_thread = threading.Thread(
                target=self._stream,
                args=(process.stdout, stdout_file, True),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._stream,
                args=(process.stderr, stderr_file, False),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            return_code = process.wait()
            stdout_thread.join(timeout=5.0)
            stderr_thread.join(timeout=5.0)
            with self._lock:
                self._state["return_code"] = return_code
            finalizer_called = self.finalizer is not None
            result = self.finalizer(self) if self.finalizer is not None else {}
            if self._cancel_requested:
                self._finish(status="canceled", message="Canceled", result=result)
            elif return_code != 0:
                error = _tail_text(self.stderr_path) or _tail_text(self.stdout_path) or f"Exit code {return_code}"
                self._finish(status="failed", message="Trainer failed", error=error, result=result)
            else:
                self._finish(status="completed", message="Completed", progress=1.0, result=result)
        except Exception as exc:
            result: dict[str, Any] = {}
            if self.finalizer is not None and not finalizer_called:
                try:
                    result = self.finalizer(self)
                except Exception:
                    result = {}
            self._finish(status="failed", message="Trainer failed", error=str(exc), result=result)
        finally:
            stdout_file.close()
            stderr_file.close()
            with self._lock:
                self._process = None

    def _stream(self, stream: TextIO | None, target: TextIO, parse_progress: bool) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            target.write(line)
            if parse_progress:
                self._consume_progress(line)
        stream.close()

    def _consume_progress(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        progress: float | None = None
        current: int | None = None
        total: int | None = None
        message = stripped[-300:]
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_progress = payload.get("progress")
            if isinstance(raw_progress, (int, float)):
                progress = float(raw_progress)
                if progress > 1.0:
                    progress /= 100.0
            raw_current = payload.get("current_step", payload.get("step"))
            raw_total = payload.get("total_steps", payload.get("total"))
            if isinstance(raw_current, (int, float)):
                current = int(raw_current)
            if isinstance(raw_total, (int, float)):
                total = int(raw_total)
            message = str(payload.get("message") or payload.get("phase") or message)
        else:
            match = _PROGRESS_PATTERN.search(stripped)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
        if progress is None and current is not None and total:
            progress = current / total
        with self._lock:
            if progress is not None:
                self._state["progress"] = max(0.0, min(1.0, progress))
            if current is not None:
                self._state["current_step"] = current
            if total is not None:
                self._state["total_steps"] = total
            self._state["message"] = message
            self._persist_locked()

    def _finish(
        self,
        *,
        status: str,
        message: str,
        progress: float | None = None,
        error: str = "",
        result: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if self._state["status"] in FINAL_JOB_STATUSES:
                return
            self._state.update(
                status=status,
                message=message,
                finished_at=_timestamp(),
                error=error,
            )
            if progress is not None:
                self._state["progress"] = progress
            if result is not None:
                self._state["result"] = result
            self._persist_locked()
            self._done.set()

    def _persist(self) -> None:
        with self._lock:
            self._persist_locked()

    def _persist_locked(self) -> None:
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self.state_path)


class TrainingJobQueue:
    """FIFO queue with bounded parallel trainer processes."""

    def __init__(self, *, max_parallel: int = 1) -> None:
        if max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        self.max_parallel = int(max_parallel)
        self._jobs: dict[str, ProcessTrainingJob] = {}
        self._pending: deque[str] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._dispatcher = threading.Thread(target=self._dispatch, name="training-job-queue", daemon=True)
        self._dispatcher.start()

    def submit(
        self,
        *,
        command: list[str],
        working_directory: Path,
        environment: dict[str, str],
        output_dir: Path,
        metadata: dict[str, Any] | None = None,
        finalizer: Callable[[ProcessTrainingJob], dict[str, Any]] | None = None,
    ) -> ProcessTrainingJob:
        job = ProcessTrainingJob(
            command=command,
            working_directory=working_directory,
            environment=environment,
            output_dir=output_dir,
            metadata=metadata,
            finalizer=finalizer,
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("Training job queue is closed.")
            self._jobs[job.job_id] = job
            self._pending.append(job.job_id)
            self._condition.notify_all()
        return job

    def get(self, job_id: str) -> ProcessTrainingJob | None:
        with self._condition:
            return self._jobs.get(job_id)

    def snapshots(self) -> list[dict[str, Any]]:
        with self._condition:
            jobs = list(self._jobs.values())
        return sorted((job.snapshot() for job in jobs), key=lambda row: str(row["created_at"]), reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        return job.cancel() if job is not None else False

    def close(self, *, cancel_running: bool = False) -> None:
        with self._condition:
            self._closed = True
            jobs = list(self._jobs.values())
            self._condition.notify_all()
        if cancel_running:
            for job in jobs:
                job.cancel()
        self._dispatcher.join(timeout=2.0)

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                self._condition.wait(timeout=0.1)
                if not self._can_dispatch():
                    continue
                if self._closed and not self._pending:
                    return
                running = sum(
                    1 for job in self._jobs.values() if job.snapshot()["status"] in {"running", "canceling"}
                )
                while self._pending and running < self.max_parallel:
                    job_id = self._pending.popleft()
                    job = self._jobs[job_id]
                    if job.snapshot()["status"] != "queued":
                        continue
                    job._start()
                    running += 1
            time.sleep(0.02)

    def _can_dispatch(self) -> bool:
        if self._closed and not self._pending:
            return True
        if not self._pending:
            return False
        running = sum(
            1 for job in self._jobs.values() if job.snapshot()["status"] in {"running", "canceling"}
        )
        return running < self.max_parallel


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate the trainer and workers spawned below it."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 and process.poll() is None:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.terminate()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
