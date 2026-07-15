from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from desktop_app import services
from desktop_app.qt_main import MainWindow


def _ensure_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _run(run_id: str, value: float, *, dataset: str = "dataset_a", status: str = "completed") -> dict:
    return {
        "run_id": run_id,
        "preset_id": "tiny_trainer",
        "preset_label": "Tiny Trainer",
        "status": status,
        "dataset_root": dataset,
        "adapter": "manifest_dataset",
        "sequence_id": "seq_001",
        "created_at": "2026-07-15T10:00:00+0800",
        "artifact_path": f"models/{run_id}.ckpt",
        "parameters": {"learning_rate": 0.01 if run_id == "run_a" else 0.001, "epochs": 3},
        "metrics": {"validation_loss": value},
        "history": {"validation_loss": [value + 0.4, value], "train_loss": [value + 0.3, value - 0.1]},
    }


def test_filter_training_runs_combines_model_dataset_status_and_date() -> None:
    runs = [
        _run("run_a", 0.2),
        {**_run("run_b", 0.3, dataset="dataset_b", status="failed"), "created_at": "2026-07-10T10:00:00+0800"},
    ]

    assert [row["run_id"] for row in services.filter_training_runs(runs, query="run_a")] == ["run_a"]
    assert [row["run_id"] for row in services.filter_training_runs(runs, dataset_root="dataset_b")] == ["run_b"]
    assert [row["run_id"] for row in services.filter_training_runs(runs, status="completed")] == ["run_a"]
    assert [row["run_id"] for row in services.filter_training_runs(runs, date_from="2026-07-12")] == ["run_a"]


def test_compare_training_runs_ranks_metric_and_returns_parameter_evidence() -> None:
    comparison = services.compare_training_runs([_run("run_a", 0.2), _run("run_b", 0.35)])

    assert comparison["metric"] == "validation_loss"
    assert comparison["direction"] == "min"
    assert comparison["best_run_id"] == "run_a"
    assert [row["run_id"] for row in comparison["rows"]] == ["run_a", "run_b"]
    assert comparison["rows"][0]["best"] is True
    assert comparison["parameter_names"] == ["epochs", "learning_rate"]
    assert len(comparison["curves"]) == 2


def test_compare_training_runs_infers_max_direction_for_accuracy() -> None:
    first = _run("run_a", 0.2)
    second = _run("run_b", 0.3)
    first["history"] = {"accuracy": [0.5, 0.8]}
    second["history"] = {"accuracy": [0.5, 0.7]}

    comparison = services.compare_training_runs([first, second], metric="accuracy")

    assert comparison["direction"] == "max"
    assert comparison["best_run_id"] == "run_a"


def test_clone_config_mark_best_and_export_reports(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    record = _run("run_a", 0.2)
    run_path = run_dir / services.TRAINING_RUN_FILENAME
    run_path.write_text(json.dumps(record), encoding="utf-8")

    clone = services.clone_training_config_from_run(record, path=tmp_path / "training_configs.json")
    marked = services.mark_best_training_run(run_path, metric="validation_loss", direction="min", value=0.2)
    comparison = services.compare_training_runs([record, _run("run_b", 0.3)])
    report = services.export_experiment_report(comparison, tmp_path / "report")

    assert clone["training_preset_id"] == "tiny_trainer"
    assert clone["parameters"]["learning_rate"] == 0.01
    assert marked["best_marks"]["validation_loss"]["value"] == 0.2
    assert "run_a" in Path(report["markdown_path"]).read_text(encoding="utf-8")
    assert "<table>" in Path(report["html_path"]).read_text(encoding="utf-8")


def test_markdown_report_escapes_headers_cells_and_newlines(tmp_path: Path) -> None:
    first = {
        **_run("run|one\nline", 0.2),
        "parameters": {"optimizer|name": "adam\nfast"},
        "metrics": {"validation|loss": 0.2},
        "history": {"validation|loss": [0.4, 0.2]},
    }
    second = {
        **_run("run_two", 0.3),
        "parameters": {"optimizer|name": "sgd"},
        "metrics": {"validation|loss": 0.3},
        "history": {"validation|loss": [0.5, 0.3]},
    }
    comparison = services.compare_training_runs(
        [first, second],
        metric="validation|loss",
    )
    report = services.export_experiment_report(comparison, tmp_path / "escaped", title="Title|A\nB")
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert "Title\\|A B" in markdown
    assert "validation\\|loss" in markdown
    assert "optimizer\\|name" in markdown
    assert "adam<br>fast" in markdown


def test_markdown_report_escapes_backticks_outside_code_spans(tmp_path: Path) -> None:
    first = _run("run_a", 0.2)
    second = _run("run_b", 0.3)
    first["history"] = {"val`loss": [0.4, 0.2]}
    second["history"] = {"val`loss": [0.5, 0.3]}
    comparison = services.compare_training_runs(
        [first, second],
        metric="val`loss",
    )
    report = services.export_experiment_report(comparison, tmp_path / "backticks")
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert "- Metric: val\\`loss" in markdown
    assert "- Metric: `" not in markdown


def test_cloning_same_run_twice_creates_distinct_configs(tmp_path: Path) -> None:
    record = _run("run_a", 0.2)
    config_path = tmp_path / "training_configs.json"

    first = services.clone_training_config_from_run(record, path=config_path)
    second = services.clone_training_config_from_run(record, path=config_path)

    assert first["id"] != second["id"]
    ids = {row["id"] for row in services.training_config_entries(config_path)}
    assert {first["id"], second["id"]}.issubset(ids)


def test_cleanup_only_removes_unreferenced_failed_or_invalid_runs(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "outputs"
    config_root = tmp_path / "configs"
    config_root.mkdir()
    monkeypatch.setattr(services, "CONFIG_ROOT", config_root)

    failed_dir = output_root / "failed"
    failed_dir.mkdir(parents=True)
    failed_path = failed_dir / services.TRAINING_RUN_FILENAME
    failed_path.write_text(json.dumps({"status": "failed", "artifact_path": str(failed_dir)}), encoding="utf-8")

    valid_dir = output_root / "valid"
    artifact = valid_dir / "model.ckpt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("checkpoint", encoding="utf-8")
    valid_path = valid_dir / services.TRAINING_RUN_FILENAME
    valid_path.write_text(
        json.dumps({"status": "completed", "artifact_path": str(artifact)}),
        encoding="utf-8",
    )

    referenced_dir = output_root / "referenced"
    referenced_dir.mkdir(parents=True)
    referenced_path = referenced_dir / services.TRAINING_RUN_FILENAME
    referenced_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    (config_root / "models.json").write_text(str(referenced_path.resolve()), encoding="utf-8")

    dry_run = services.cleanup_training_runs(
        [failed_path, valid_path, referenced_path],
        output_root=output_root,
        dry_run=True,
    )
    removed = services.cleanup_training_runs([failed_path], output_root=output_root, dry_run=False)

    assert str(failed_dir.resolve()) in dry_run["candidates"]
    assert any(row["reason"] == "completed run has a valid artifact" for row in dry_run["refused"])
    assert any(row["reason"] == "referenced by configuration" for row in dry_run["refused"])
    assert removed["removed"] == [str(failed_dir.resolve())]
    assert not failed_dir.exists()


def test_gui_filters_compares_and_exports_selected_experiments(tmp_path: Path, monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    first = _run("run_a", 0.2)
    second = _run("run_b", 0.35)
    failed = _run("run_failed", 0.8, status="failed")
    window.catalog["training_runs"] = [first, second, failed]
    window._fill_training_run_list()

    window.training_run_list.item(0).setSelected(True)
    window.training_run_list.item(1).setSelected(True)
    window.compare_selected_training_runs()

    assert window.experiment_comparison_table.rowCount() == 2
    assert "run_a" in window.experiment_comparison_summary.text()
    assert len(window.experiment_comparison_curve.selected_metrics) == 2
    assert window.training_result_tabs.currentIndex() == 1

    report_dir = tmp_path / "gui_report"
    monkeypatch.setattr(
        "desktop_app.qt_main.QFileDialog.getExistingDirectory",
        lambda *args, **kwargs: str(report_dir),
    )
    window.export_experiment_comparison_report()
    assert (report_dir / "experiment_report.md").is_file()
    assert (report_dir / "experiment_report.html").is_file()
    assert (report_dir / "experiment_curves.png").is_file()

    failed_index = window.experiment_status_filter.findData("failed")
    window.experiment_status_filter.setCurrentIndex(failed_index)
    assert window.training_run_list.count() == 1
    assert "run_failed" in window.training_run_list.item(0).text()
    window.close()
