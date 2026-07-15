from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_app import services
from desktop_app.qt_main import MainWindow, TrainingCurveWidget
from offroad_sim.training import TrainingJobQueue
from offroad_sim.training import jobs as training_jobs


def _ensure_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_metric_diagnostics_detect_nonfinite_explosion_and_stall(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text('{"step": 1, "custom_score": NaN}\n', encoding="utf-8")
    record = {
        "status": "completed",
        "history": {
            "train_loss": [0.5, 0.4, 20.0],
            "validation_loss": [0.7] * 20,
            "throughput": [12.0, 13.0],
        },
        "summary": {"events_path": str(events)},
    }

    diagnostics = services.training_metric_diagnostics(record)

    codes = {(row["code"], row["metric"]) for row in diagnostics["warnings"]}
    assert diagnostics["status"] == "critical"
    assert ("non_finite", "custom_score") in codes
    assert ("loss_explosion", "train_loss") in codes
    assert ("metric_stalled", "validation_loss") in codes


def test_metric_export_writes_json_and_rectangular_csv(tmp_path: Path) -> None:
    record = {
        "run_id": "run_a",
        "preset_id": "tiny",
        "status": "completed",
        "metrics": {"loss": 0.2},
        "history": {"loss": [0.8, 0.4, 0.2], "learning_rate": [0.01, 0.005]},
    }

    payload = services.export_training_metrics(record, tmp_path / "export")

    exported = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))
    csv_lines = Path(payload["csv_path"]).read_text(encoding="utf-8").splitlines()
    assert exported["history"]["loss"] == [0.8, 0.4, 0.2]
    assert exported["diagnostics"]["status"] == "healthy"
    assert csv_lines[0] == "step,learning_rate,loss"
    assert len(csv_lines) == 4


def test_nonfinite_metric_filter_preserves_original_step_alignment(tmp_path: Path) -> None:
    record = services.write_training_run_record(
        tmp_path / "run",
        preset_id="metric_alignment",
        status="completed",
        history={"loss": [1.0, float("nan"), 0.5]},
        history_steps={"loss": [10, 20, 30]},
    )

    assert record["history"]["loss"] == [1.0, 0.5]
    assert record["history_steps"]["loss"] == [10.0, 30.0]
    assert services.training_metric_steps(record)["loss"] == [10.0, 30.0]


def test_live_metric_record_combines_events_and_resource_samples(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                '{"step": 1, "train_loss": 0.8, "learning_rate": 0.01}',
                '{"step": 2, "train_loss": 0.4, "learning_rate": 0.005}',
            ]
        ),
        encoding="utf-8",
    )
    record = services.live_training_metric_record(
        {
            "job_id": "job_a",
            "status": "running",
            "output_dir": str(tmp_path),
            "metadata": {"events_path": str(events)},
            "resource_history": {"cpu_percent": [10.0, 15.0], "memory_mb": [100.0, 101.0]},
        }
    )

    assert record["history"]["train_loss"] == [0.8, 0.4]
    assert record["history"]["learning_rate"] == [0.01, 0.005]
    assert record["history"]["resource.cpu_percent"] == [10.0, 15.0]
    assert record["metric_diagnostics"]["resource_metrics"] == [
        "resource.cpu_percent",
        "resource.memory_mb",
    ]


def test_live_resource_metric_filter_preserves_elapsed_step_alignment(tmp_path: Path) -> None:
    record = services.live_training_metric_record(
        {
            "job_id": "resource_alignment",
            "status": "running",
            "output_dir": str(tmp_path),
            "resource_history": {
                "elapsed_sec": [1.0, 2.0, 3.0],
                "cpu_percent": [10.0, float("nan"), 15.0],
            },
        }
    )

    assert record["history"]["resource.cpu_percent"] == [10.0, 15.0]
    assert record["history_steps"]["resource.cpu_percent"] == [1.0, 3.0]


def test_training_job_records_eta_and_process_resources(tmp_path: Path) -> None:
    script = tmp_path / "trainer.py"
    script.write_text(
        "import json, time\n"
        "for step in range(1, 4):\n"
        " print(json.dumps({'progress': step / 3, 'step': step, 'total': 3}), flush=True)\n"
        " time.sleep(0.35)\n",
        encoding="utf-8",
    )
    queue = TrainingJobQueue(max_parallel=1)
    try:
        job = queue.submit(
            command=[sys.executable, str(script)],
            working_directory=tmp_path,
            environment=dict(os.environ),
            output_dir=tmp_path / "run",
        )
        finished = job.wait(timeout=10.0)
    finally:
        queue.close(cancel_running=True)

    assert finished["status"] == "completed"
    assert finished["eta_seconds"] == 0.0
    assert finished["resource_history"]["elapsed_sec"]
    assert finished["resource_history"]["memory_mb"]
    assert finished["resources"]["memory_mb"] > 0.0


def test_resource_sampler_keeps_cpu_baselines_for_busy_child_process(tmp_path: Path) -> None:
    if training_jobs.psutil is None:
        return
    script = tmp_path / "busy.py"
    script.write_text(
        "import time\nend=time.monotonic()+2.5\nx=0\nwhile time.monotonic()<end: x=(x+1)%1000003\n",
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(script)])
    try:
        sampler = training_jobs._ProcessResourceSampler(process.pid)
        samples = []
        for _ in range(6):
            time.sleep(0.25)
            samples.append(sampler.sample().get("cpu_percent", 0.0))
    finally:
        process.terminate()
        process.wait(timeout=5.0)

    assert max(samples) > 0.0


def test_curve_selects_train_and_validation_loss_and_custom_metrics() -> None:
    _ensure_app()
    curve = TrainingCurveWidget()
    curve.set_history(
        {
            "train_loss": [0.8, 0.4],
            "validation_loss": [0.9, 0.5],
            "custom_iou": [0.2, 0.4],
        }
    )

    assert curve.primary_metric == "train_loss"
    assert curve.selected_metrics == ["train_loss", "validation_loss"]
    curve.set_primary_metric("custom_iou")
    assert curve.selected_metrics == ["custom_iou"]


def test_curve_uses_per_metric_steps_and_preserves_live_zoom() -> None:
    _ensure_app()
    curve = TrainingCurveWidget()
    history = {
        "train_loss": [1.0 - index * 0.005 for index in range(100)],
        "validation_loss": [0.9 - index * 0.02 for index in range(10)],
    }
    steps = {
        "train_loss": list(range(1, 101)),
        "validation_loss": list(range(10, 101, 10)),
    }
    curve.set_history(history, steps)
    curve.x_zoom = 6.0

    history["train_loss"].append(0.49)
    steps["train_loss"].append(101)
    curve.set_history(history, steps)
    curve.resize(700, 260)
    curve.grab()

    assert curve.x_zoom == 6.0
    assert curve.steps["validation_loss"][-1] == 100.0
    assert curve._projected_points["validation_loss"]


def test_gui_metric_panel_selects_metrics_shows_diagnostics_and_exports(tmp_path: Path, monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    record = {
        "run_id": "metric_demo",
        "preset_id": "tiny",
        "status": "completed",
        "history": {
            "train_loss": [0.5, 0.4, 20.0],
            "validation_loss": [0.6, 0.5, 0.4],
            "resource.memory_mb": [100.0, 101.0],
        },
    }
    window._set_training_run_views(record)

    assert window.training_metric_combo.findText("train_loss") >= 0
    assert window.training_metric_combo.findText("resource.memory_mb") >= 0
    assert window.training_curve.selected_metrics == ["train_loss", "validation_loss"]
    assert "critical" in window.training_metric_warning_label.text()

    export_dir = tmp_path / "gui_export"
    monkeypatch.setattr(
        "desktop_app.qt_main.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(export_dir),
    )
    window.export_selected_training_metrics()

    assert (export_dir / "training_metrics.json").is_file()
    assert (export_dir / "training_metrics.csv").is_file()
    assert (export_dir / "training_curves.png").is_file()
    window.close()
