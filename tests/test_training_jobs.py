from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_app import services
from desktop_app.qt_main import MainWindow
from offroad_sim.training import TRAINING_JOB_FILENAME, TrainingJobQueue
from offroad_sim.training import jobs as training_jobs


def _write_progress_trainer(root: Path, *, delay: float = 0.05, fail: bool = False) -> Path:
    script = root / "progress_trainer.py"
    script.write_text(
        f"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
for step in range(1, 5):
    print(json.dumps({{'progress': step / 4, 'current_step': step, 'total_steps': 4, 'message': f'epoch {{step}}'}}), flush=True)
    time.sleep({delay})
if {fail!r}:
    print('intentional failure', file=sys.stderr, flush=True)
    raise SystemExit(3)
checkpoint = output / 'progress.ckpt'
checkpoint.write_text('checkpoint', encoding='utf-8')
print(json.dumps({{
    'checkpoint_path': str(checkpoint.resolve()),
    'artifact_type': 'checkpoint',
    'metrics': {{'final_loss': 0.125}},
    'history': {{'loss': [0.8, 0.4, 0.2, 0.125]}},
}}), flush=True)
""",
        encoding="utf-8",
    )
    manifest = root / "trainer.yaml"
    manifest.write_text(
        """
schema_version: 1
trainer_id: progress_trainer
display_name: Progress Trainer
launch:
  kind: python_script
  entrypoint: progress_trainer.py
arguments:
  - "{dataset_root}"
  - --output
  - "{output_dir}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    return manifest


def _wait_for_status(job, statuses: set[str], timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = job.snapshot()
        if snapshot["status"] in statuses:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"Job did not reach {statuses}: {job.snapshot()}")


def _process_until(predicate, timeout: float = 10.0) -> bool:
    app = QApplication.instance() or QApplication([])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(predicate())


def test_manifest_training_job_streams_progress_and_finalizes_record(tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(dataset_root),
            output_dir=str(tmp_path / "run"),
        )
        running = _wait_for_status(job, {"running", "completed"})
        assert Path(running["state_path"]).is_file()
        finished = job.wait(timeout=10.0)
    finally:
        queue.close(cancel_running=True)

    assert finished["status"] == "completed"
    assert finished["progress"] == 1.0
    assert finished["current_step"] == 4
    assert finished["total_steps"] == 4
    assert "epoch 4" in Path(finished["stdout_path"]).read_text(encoding="utf-8")
    assert Path(finished["result"]["artifact_path"]).name == "progress.ckpt"
    record = json.loads((tmp_path / "run" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["metrics"]["final_loss"] == 0.125
    assert record["history"]["loss"] == [0.8, 0.4, 0.2, 0.125]


def test_training_job_queue_runs_fifo_with_single_worker(tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path, delay=0.1)
    queue = TrainingJobQueue(max_parallel=1)
    try:
        first = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "first"),
        )
        second = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "second"),
        )
        _wait_for_status(first, {"running"})
        assert second.snapshot()["status"] == "queued"
        assert first.wait(timeout=10.0)["status"] == "completed"
        assert second.wait(timeout=10.0)["status"] == "completed"
    finally:
        queue.close(cancel_running=True)


def test_training_job_cancel_updates_job_and_training_run(tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path, delay=1.0)
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "canceled"),
        )
        _wait_for_status(job, {"running"})
        assert job.cancel() is True
        finished = job.wait(timeout=10.0)
    finally:
        queue.close(cancel_running=True)

    assert finished["status"] == "canceled"
    record = json.loads((tmp_path / "canceled" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "canceled"


def test_cancel_queued_training_job_finalizes_record_without_starting(tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path, delay=1.0)
    queue = TrainingJobQueue(max_parallel=1)
    try:
        first = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "running"),
        )
        second = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "queued_cancel"),
        )
        _wait_for_status(first, {"running"})
        assert second.snapshot()["status"] == "queued"
        assert second.cancel() is True
        finished = second.wait(timeout=5.0)
    finally:
        queue.close(cancel_running=True)

    assert finished["status"] == "canceled"
    assert finished["pid"] is None
    record = json.loads((tmp_path / "queued_cancel" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "canceled"


def test_cancel_terminates_trainer_worker_process_tree(tmp_path) -> None:
    marker = tmp_path / "worker_survived.txt"
    worker_code = (
        "import time; from pathlib import Path; "
        f"time.sleep(2.0); Path({str(marker)!r}).write_text('survived', encoding='utf-8')"
    )
    script = tmp_path / "tree_trainer.py"
    script.write_text(
        "from __future__ import annotations\n"
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {worker_code!r}])\n"
        "print('worker started', flush=True)\n"
        "time.sleep(30.0)\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "trainer_id: tree_trainer",
                "launch:",
                "  kind: python_script",
                "  entrypoint: tree_trainer.py",
            ]
        ),
        encoding="utf-8",
    )
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "tree_run"),
        )
        _wait_for_status(job, {"running"})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            stdout_path = Path(job.snapshot()["stdout_path"])
            if stdout_path.is_file() and "worker started" in stdout_path.read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(0.02)
        assert job.cancel() is True
        assert job.wait(timeout=10.0)["status"] == "canceled"
        time.sleep(2.5)
    finally:
        queue.close(cancel_running=True)

    assert marker.exists() is False


def test_cancel_during_process_spawn_terminates_immediately(monkeypatch, tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path, delay=5.0)
    real_popen = training_jobs.subprocess.Popen
    spawn_entered = threading.Event()
    allow_spawn = threading.Event()

    def delayed_popen(*args, **kwargs):
        spawn_entered.set()
        assert allow_spawn.wait(timeout=5.0)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(training_jobs.subprocess, "Popen", delayed_popen)
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "spawn_cancel"),
        )
        assert spawn_entered.wait(timeout=5.0)
        assert job.snapshot()["status"] == "running"
        assert job.snapshot()["pid"] is None
        started = time.monotonic()
        assert job.cancel() is True
        allow_spawn.set()
        finished = job.wait(timeout=10.0)
        elapsed = time.monotonic() - started
    finally:
        allow_spawn.set()
        queue.close(cancel_running=True)

    assert finished["status"] == "canceled"
    assert elapsed < 3.0


def test_training_job_failure_keeps_diagnostics(tmp_path) -> None:
    manifest = _write_progress_trainer(tmp_path, fail=True)
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = services.queue_trainer_manifest_job(
            queue,
            manifest,
            dataset_root=str(tmp_path),
            output_dir=str(tmp_path / "failed"),
        )
        finished = job.wait(timeout=10.0)
    finally:
        queue.close(cancel_running=True)

    assert finished["status"] == "failed"
    assert finished["return_code"] == 3
    assert "intentional failure" in finished["error"]
    record = json.loads((tmp_path / "failed" / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "intentional failure" in record["summary"]["error"]


def test_training_job_discovery_marks_stale_active_state_interrupted(tmp_path) -> None:
    run_dir = tmp_path / "stale"
    run_dir.mkdir()
    state_path = run_dir / TRAINING_JOB_FILENAME
    state_path.write_text(json.dumps({"job_id": "old", "status": "running"}), encoding="utf-8")

    rows = services.training_job_entries(tmp_path)

    assert rows[0]["job_id"] == "old"
    assert rows[0]["status"] == "interrupted"
    assert "Previous desktop session" in rows[0]["message"]


def test_gui_tracks_manifest_job_progress_logs_and_completion(tmp_path) -> None:
    QApplication.instance() or QApplication([])
    manifest = _write_progress_trainer(tmp_path, delay=0.05)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "gui_run"
    window = MainWindow()
    window.catalog["training_presets"] = [
        {
            "id": "progress_trainer",
            "label": "Progress Trainer",
            "available": True,
            "manifest_path": str(manifest),
            "parameters": {},
        }
    ]
    window._fill_training_preset_combo()
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("")
    window.sequence_combo.setCurrentText("")
    window.training_output_edit.setText(str(output_dir))

    window.run_training_preset()

    assert window._current_training_job_id
    assert _process_until(
        lambda: window._current_training_job_id in window._handled_training_job_ids,
        timeout=10.0,
    )
    assert window.training_job_progress.value() == 100
    assert "completed" in window.training_job_status_label.text()
    assert "epoch 4" in window.latest_training_log.toPlainText()
    assert window.training_job_table.rowCount() >= 1
    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["history"]["loss"][-1] == 0.125
    window.close()


def test_gui_can_pause_log_refresh_without_pausing_metrics(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    window = MainWindow()
    job = {
        "job_id": "active-job",
        "status": "running",
        "message": "training",
        "progress": 0.5,
        "stdout_path": "stdout.log",
        "stderr_path": "stderr.log",
    }
    window._current_training_job_id = "active-job"
    monkeypatch.setattr(window.training_job_queue, "snapshots", lambda: [job])
    monkeypatch.setattr(window, "_training_job_log_tail", lambda *_args: "new log line")
    live_calls: list[dict] = []
    monkeypatch.setattr(services, "live_training_metric_record", lambda _job: {"history": {"loss": [1.0]}})
    monkeypatch.setattr(window, "_set_live_training_metric_views", live_calls.append)
    window.latest_training_log.setPlainText("kept log")
    window.pause_training_log_button.setChecked(True)

    window._refresh_training_jobs()

    assert window.latest_training_log.toPlainText() == "kept log"
    assert live_calls == [{"history": {"loss": [1.0]}}]
    window.close()


def test_gui_close_stops_generic_qthreads() -> None:
    QApplication.instance() or QApplication([])
    window = MainWindow()
    completed = threading.Event()
    cancel_called = threading.Event()
    callbacks = []

    def bounded_task() -> str:
        time.sleep(0.3)
        completed.set()
        return "done"

    window._run_task(
        bounded_task,
        callbacks.append,
        "long task",
        cancel_hook=cancel_called.set,
    )
    assert _process_until(lambda: bool(window.threads), timeout=1.0)
    active = list(window.threads)

    window.close()

    assert window.threads == []
    assert cancel_called.is_set()
    assert completed.is_set()
    assert callbacks == []
    assert all(not thread.is_alive() for thread in active)


def test_gui_close_retains_and_reports_noncooperative_detached_task() -> None:
    QApplication.instance() or QApplication([])
    window = MainWindow()
    release = threading.Event()
    window._run_task(lambda: release.wait(timeout=5.0), lambda _: None, "blocking task")
    assert _process_until(lambda: bool(window.threads), timeout=1.0)

    window.close()

    entries = window.detached_task_entries()
    assert entries and entries[0]["alive"] is True
    assert window.detached_task_names == [entries[0]["name"]]
    release.set()
    assert _process_until(lambda: not window.threads, timeout=2.0)
    assert window.detached_task_entries() == []
