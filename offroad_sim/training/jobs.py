"""Persistent subprocess jobs for local model training."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

try:
    import psutil
except ImportError:  # pragma: no cover - exercised when the optional GUI dependency is absent.
    psutil = None


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
        self._resource_stop = threading.Event()
        self._cancel_requested = False
        self._process: subprocess.Popen[str] | None = None
        self._started_monotonic: float | None = None
        self._state: dict[str, Any] = {
            "job_id": self.job_id,
            "status": "queued",
            "progress": 0.0,
            "current_step": None,
            "total_steps": None,
            "eta_seconds": None,
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
            "resources": {},
            "resource_history": {},
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
            threading.Thread(
                target=_terminate_process_tree,
                args=(process,),
                name=f"training-cancel-{self.job_id}",
                daemon=True,
            ).start()
        return True

    def _start(self) -> None:
        with self._lock:
            if self._cancel_requested or self._state["status"] == "canceled":
                return
            self._state.update(status="running", message="Running", started_at=_timestamp())
            self._started_monotonic = time.monotonic()
            self._persist_locked()
        thread = threading.Thread(target=self._run, name=f"training-job-{self.job_id}", daemon=True)
        thread.start()

    def _run(self) -> None:
        stdout_file = self.stdout_path.open("w", encoding="utf-8", buffering=1)
        stderr_file = self.stderr_path.open("w", encoding="utf-8", buffering=1)
        finalizer_called = False
        resource_thread: threading.Thread | None = None
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
            self._resource_stop.clear()
            resource_thread = threading.Thread(
                target=self._sample_resources,
                args=(process.pid,),
                name=f"training-resources-{self.job_id}",
                daemon=True,
            )
            resource_thread.start()
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
            self._resource_stop.set()
            resource_thread.join(timeout=2.0)
            stdout_thread.join(timeout=5.0)
            stderr_thread.join(timeout=5.0)
            with self._lock:
                self._state["return_code"] = return_code
            finalizer_called = self.finalizer is not None
            result = self.finalizer(self) if self.finalizer is not None else {}
            result_status = str(result.get("status") or "") if isinstance(result, dict) else ""
            if self._cancel_requested:
                self._finish(status="canceled", message="Canceled", result=result)
            elif return_code != 0:
                error = _tail_text(self.stderr_path) or _tail_text(self.stdout_path) or f"Exit code {return_code}"
                self._finish(status="failed", message="Trainer failed", error=error, result=result)
            elif result_status == "failed":
                self._finish(
                    status="failed",
                    message="Trainer output validation failed",
                    error=str(result.get("error") or "Trainer output validation failed"),
                    result=result,
                )
            elif result_status == "canceled":
                self._finish(status="canceled", message="Canceled", result=result)
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
            self._resource_stop.set()
            if resource_thread is not None and resource_thread.is_alive():
                resource_thread.join(timeout=1.0)
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
                if self._started_monotonic is not None and 0.0 < progress < 1.0:
                    elapsed = max(0.0, time.monotonic() - self._started_monotonic)
                    self._state["eta_seconds"] = elapsed * (1.0 - progress) / progress
                elif progress >= 1.0:
                    self._state["eta_seconds"] = 0.0
            if current is not None:
                self._state["current_step"] = current
            if total is not None:
                self._state["total_steps"] = total
            self._state["message"] = message
            self._persist_locked()

    def _sample_resources(self, pid: int) -> None:
        sampler = _ProcessResourceSampler(pid)
        while not self._resource_stop.is_set():
            sample = sampler.sample()
            if sample:
                with self._lock:
                    previous = self._state.get("resources") if isinstance(self._state.get("resources"), dict) else {}
                    effective_sample = dict(sample)
                    if (
                        float(effective_sample.get("memory_mb", 0.0)) <= 0.0
                        and float(previous.get("memory_mb", 0.0)) > 0.0
                    ):
                        effective_sample.pop("memory_mb", None)
                    self._state["resources"] = {**previous, **effective_sample}
                    history = self._state.setdefault("resource_history", {})
                    for key, value in effective_sample.items():
                        if isinstance(value, (int, float)) and math.isfinite(float(value)):
                            values = history.setdefault(str(key), [])
                            values.append(float(value))
                            if len(values) > 2000:
                                del values[:-2000]
                    self._persist_locked()
            self._resource_stop.wait(0.5)

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


class _ProcessResourceSampler:
    """Collect bounded process telemetry without making GPU support mandatory."""

    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self.started = time.monotonic()
        self.process = None
        self._processes: dict[int, Any] = {}
        self._sample_count = 0
        self._last_gpu: dict[str, float] = {}
        if psutil is not None:
            try:
                self.process = psutil.Process(self.pid)
                self.process.cpu_percent(interval=None)
                self._processes[self.pid] = self.process
            except (psutil.Error, OSError):
                self.process = None

    def sample(self) -> dict[str, float]:
        values: dict[str, float] = {"elapsed_sec": max(0.0, time.monotonic() - self.started)}
        pids = {self.pid}
        if self.process is not None and psutil is not None:
            try:
                discovered = [self.process, *self.process.children(recursive=True)]
            except (psutil.Error, OSError):
                discovered = [self.process]
            for discovered_process in discovered:
                process_pid = int(discovered_process.pid)
                if process_pid not in self._processes:
                    try:
                        cached = psutil.Process(process_pid)
                        cached.cpu_percent(interval=None)
                        self._processes[process_pid] = cached
                    except (psutil.Error, OSError):
                        continue
            cpu_percent = 0.0
            memory_bytes = 0
            stale_pids: list[int] = []
            for process_pid, process in self._processes.items():
                try:
                    pids.add(process_pid)
                    cpu_percent += float(process.cpu_percent(interval=None))
                    memory_bytes += int(process.memory_info().rss)
                except (psutil.Error, OSError):
                    stale_pids.append(process_pid)
            for process_pid in stale_pids:
                self._processes.pop(process_pid, None)
            values["cpu_percent"] = cpu_percent
            values["memory_mb"] = memory_bytes / (1024.0 * 1024.0)

        self._sample_count += 1
        if self._sample_count == 1 or self._sample_count % 4 == 0:
            self._last_gpu = _nvidia_resource_sample(pids)
        values.update(self._last_gpu)
        return values


def _nvidia_resource_sample(pids: set[int]) -> dict[str, float]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {}
    try:
        apps = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    memory_mb = 0.0
    matched = False
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            process_id = int(parts[0])
            used_memory = float(parts[1])
        except ValueError:
            continue
        if process_id in pids:
            matched = True
            memory_mb += used_memory
    if not matched:
        return {}
    sample = {"gpu_memory_mb": memory_mb}
    try:
        devices = subprocess.run(
            [
                executable,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
        utilization = [float(line.strip()) for line in devices.stdout.splitlines() if line.strip()]
    except (OSError, ValueError, subprocess.SubprocessError):
        utilization = []
    if utilization:
        sample["gpu_percent"] = max(utilization)
    return sample


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Ask a trainer group to stop, then force its process tree after a grace period."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=3.0)
            return
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5.0,
            )
            taskkill_failed = completed.returncode != 0
        except (OSError, subprocess.TimeoutExpired):
            taskkill_failed = True
        if taskkill_failed and process.poll() is None:
            process.terminate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.kill()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
