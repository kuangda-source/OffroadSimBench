from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from desktop_app import services
from desktop_app.qt_main import MainWindow, NavigationTaskCanvas, NavigationTaskDialog, STYLESHEET, TrainingCurveWidget


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _process_until(predicate, timeout_sec: float = 2.0) -> bool:
    app = _ensure_app()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def test_visible_demo_request_keeps_dataset_model_and_backend_switchable() -> None:
    request = services.VisibleBeamNGDemoRequest(
        dataset_root="datasets/ORFD_Dataset_ICRA2022_ZIP",
        adapter="orfd",
        sequence_id="training/c2021_0228_1819",
        world_model_type="le_wm",
        world_model_path="outputs/models/lewm_orfd",
        planner="le_wm_cem",
        scenario="beamng_visible_autodrive",
        vehicle="configs/vehicles/ugv_medium.yaml",
    )

    payload = services.build_visible_beamng_demo_request(request)

    assert payload.agent == "route_world_model"
    assert payload.backend == "beamng"
    assert payload.world_model_type == "le_wm"
    assert payload.world_model_path == "outputs/models/lewm_orfd"


def test_visible_demo_service_runs_route_world_model_episode() -> None:
    request = services.VisibleBeamNGDemoRequest(world_model_type="simple_kinematic", planner="")
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "visible", "metrics": {"connected": True}}

        payload = services.run_visible_beamng_demo(request)

    assert payload["episode_id"] == "visible"
    assert run_episode.call_args.kwargs["backend_name"] == "beamng"
    assert run_episode.call_args.kwargs["agent_name"] == "route_world_model"
    assert run_episode.call_args.kwargs["vehicle"] == "configs/vehicles/ugv_medium.yaml"
    assert run_episode.call_args.kwargs["backend_options"]["connection"].gfx == "vk"


def test_visible_demo_defaults_are_human_visible() -> None:
    request = services.VisibleBeamNGDemoRequest(world_model_type="simple_kinematic", planner="")
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "visible", "metrics": {"connected": True}}

        services.run_visible_beamng_demo(request)

    kwargs = run_episode.call_args.kwargs
    assert kwargs["pre_run_hold_sec"] >= 5.0
    assert kwargs["step_delay_sec"] > 0.0
    assert kwargs["close_backend"] is False


def test_beamng_page_uses_generic_actions_only() -> None:
    _ensure_app()
    window = MainWindow()

    texts = [button.text() for button in window.findChildren(QPushButton)]

    assert "编辑/预览区域与起终点" in texts
    assert "运行当前区域任务" in texts
    assert "Johnson Valley LE-WM 演示" not in texts
    assert "启动 BeamNG 可视自动驾驶" not in texts
    assert "BeamNG LE-WM 闭环训练评估" not in texts
    assert "编辑区域/起终点" not in texts
    assert "BeamNG 预览区域/起终点" not in texts
    window.close()


def test_gui_busy_indicator_lifecycle() -> None:
    _ensure_app()
    window = MainWindow()

    window._set_busy(True, "开始测试")

    assert window.busy_label.isHidden() is False
    assert window.busy_bar.isHidden() is False
    assert window.busy_bar.minimum() == 0
    assert window.busy_bar.maximum() == 0
    assert "开始测试" in window.busy_label.text()

    window._set_busy(False)

    assert window.busy_label.isHidden() is True
    assert window.busy_bar.isHidden() is True
    window.close()


def test_gui_run_task_shows_and_hides_busy_indicator() -> None:
    _ensure_app()
    window = MainWindow()
    release = threading.Event()
    payload = {"ok": True}
    seen: list[object] = []

    def run_slow_task() -> dict[str, bool]:
        release.wait(2.0)
        return payload

    window._run_task(run_slow_task, seen.append, "demo failed", task_label="开始测试")

    assert window.busy_label.isHidden() is False
    assert window.busy_bar.isHidden() is False
    assert "开始测试" in window.busy_label.text()

    release.set()

    assert _process_until(lambda: window.busy_bar.isHidden() and not window.threads and not window.workers)
    assert seen == [payload]
    window.close()


def test_gui_exposes_algorithm_choice() -> None:
    _ensure_app()
    window = MainWindow()

    values = [window.algorithm_combo.itemData(index) for index in range(window.algorithm_combo.count())]

    assert "local_lewm_cost" in values
    window.close()


def test_gui_overview_uses_world_model_config_selector() -> None:
    _ensure_app()
    window = MainWindow()

    overview = window.page_stack.widget(0)
    labels = [label.text() for label in overview.findChildren(QLabel)]

    assert "World model config" in labels
    assert "Model path" not in labels
    assert "Algorithm" not in labels
    assert "World model" not in labels
    window.close()


def test_gui_uses_guided_demo_and_two_workbenches() -> None:
    _ensure_app()
    window = MainWindow()

    nav_texts = [button.text() for button in window.nav_buttons]

    assert nav_texts == ["总览", "数据集与训练", "BeamNG 仿真", "实验记录"]
    window.close()


def test_gui_overview_is_guided_demo_launcher() -> None:
    _ensure_app()
    window = MainWindow()

    overview = window.page_stack.widget(0)
    labels = [label.text() for label in overview.findChildren(QLabel)]
    buttons = [button.text() for button in overview.findChildren(QPushButton)]

    assert "Demo preset" in labels
    assert "BeamNG region task" in labels
    assert "World model config" in labels
    assert "Run guided demo" in buttons
    assert "Open Dataset & Training" in buttons
    assert "Open BeamNG Simulation" in buttons
    assert "Backend" not in labels
    assert "Scenario" not in labels
    assert "Agent" not in labels
    window.close()


def test_gui_dataset_training_page_exposes_training_studio_controls() -> None:
    _ensure_app()
    window = MainWindow()

    dataset_page = window.page_stack.widget(1)
    labels = [label.text() for label in dataset_page.findChildren(QLabel)]
    buttons = [button.text() for button in dataset_page.findChildren(QPushButton)]

    assert "Training config" in labels
    assert "Training preset" in labels
    assert "Training output" in labels
    assert "Training config summary" in labels
    assert "Latest metric curve" in labels
    assert "Dataset name" in labels
    assert "Manifest sequences" in labels
    assert "Auto-detect sequences" in buttons
    assert "Save dataset manifest" in buttons
    assert "Preview dataset frame" in buttons
    assert "预览 ORFD 图像" not in buttons
    assert "Save training config" in buttons
    assert "Start training/export" in buttons
    assert hasattr(window, "training_preset_summary")
    assert hasattr(window, "training_output_edit")
    assert hasattr(window, "training_run_list")
    assert hasattr(window, "training_run_summary")
    window.close()


def test_gui_dataset_preview_has_titles_and_readable_tabs() -> None:
    _ensure_app()
    window = MainWindow()

    dataset_page = window.page_stack.widget(1)
    labels = [label.text() for label in dataset_page.findChildren(QLabel)]

    assert "RGB preview" in labels
    assert "Depth / Label preview" in labels
    assert "Frame metadata" in labels
    assert "QTabBar::tab" in STYLESHEET
    assert "QTabBar::tab:selected" in STYLESHEET
    assert "#f5f5f7" in STYLESHEET
    assert "#ffffff" in STYLESHEET
    assert "#1d1d1f" in STYLESHEET
    assert "#007aff" in STYLESHEET
    window.close()


def test_gui_training_preset_summary_updates_for_manifest_trainer(tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text("trainer_id: echo_trainer\nentrypoint: echo_trainer.py\n", encoding="utf-8")
    window = MainWindow()
    window.catalog["training_presets"] = [
        {
            "id": "echo_trainer",
            "label": "Echo Trainer",
            "kind": "training",
            "available": True,
            "description": "Train an external echo model.",
            "manifest_path": str(manifest),
            "parameters": {"epochs": {"type": "int", "default": 3}},
        }
    ]

    window._fill_training_preset_combo()

    summary = window.training_preset_summary.toPlainText()
    assert "Echo Trainer" in summary
    assert "Train an external echo model." in summary
    assert str(manifest) in summary
    assert '"epochs": 3' in window.trainer_params_edit.toPlainText()
    window.close()


def test_gui_imports_trainer_manifest(monkeypatch, tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text("trainer_id: echo_trainer\nentrypoint: echo_trainer.py\n", encoding="utf-8")
    window = MainWindow()
    imported = {
        "id": "echo_trainer",
        "label": "Echo Trainer",
        "kind": "training",
        "available": True,
        "description": "Imported trainer.",
        "manifest_path": str(manifest),
        "parameters": {},
    }

    monkeypatch.setattr("desktop_app.qt_main.QFileDialog.getOpenFileName", lambda *args, **kwargs: (str(manifest), ""))
    monkeypatch.setattr(services, "import_trainer_manifest", lambda path: imported)
    def fake_refresh_catalogs() -> None:
        window.catalog["training_presets"] = [imported]
        window._fill_training_preset_combo()

    monkeypatch.setattr(window, "refresh_catalogs", fake_refresh_catalogs)

    window.import_trainer_manifest()

    assert window.training_preset_combo.currentData()["id"] == "echo_trainer"
    assert "Imported trainer." in window.training_preset_summary.toPlainText()
    window.close()


def test_gui_imports_dataset_manifest(monkeypatch, tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "dataset_manifest.yaml"
    manifest.write_text("adapter: manifest_dataset\ndataset_id: custom_drive\nsequences: []\n", encoding="utf-8")
    window = MainWindow()
    imported = {
        "id": "custom_drive",
        "label": "Custom Drive",
        "adapter": "manifest_dataset",
        "dataset_root": str(tmp_path / "installed" / "custom_drive"),
        "manifest_path": str(manifest),
        "sequences": ["clip_001"],
    }

    monkeypatch.setattr("desktop_app.qt_main.QFileDialog.getOpenFileName", lambda *args, **kwargs: (str(manifest), ""))
    monkeypatch.setattr(services, "import_dataset_manifest", lambda path: imported)

    def fake_refresh_catalogs() -> None:
        window.catalog["dataset_manifests"] = [imported]
        window._fill_dataset_manifest_combo()

    monkeypatch.setattr(window, "refresh_catalogs", fake_refresh_catalogs)

    window.import_dataset_manifest()

    assert window.dataset_catalog_combo.currentData()["id"] == "custom_drive"
    assert window.dataset_root_edit.text() == imported["dataset_root"]
    assert window.adapter_edit.text() == "manifest_dataset"
    assert window.sequence_combo.currentText() == "clip_001"
    assert "custom_drive" in window.dataset_summary.toPlainText()
    window.close()


def test_gui_saves_dataset_manifest_from_current_dataset_fields(monkeypatch, tmp_path) -> None:
    _ensure_app()
    window = MainWindow()
    dataset_root = tmp_path / "drive_dataset"
    sequence_root = dataset_root / "clip_001"
    sequence_root.mkdir(parents=True)
    window.dataset_manifest_name_edit.setText("Custom Drive")
    window.dataset_root_edit.setText(str(dataset_root))
    window.dataset_manifest_sequences_edit.setPlainText(
        '[{"id": "clip_001", "root": "clip_001", "assets": {"front_rgb": "images/*.png"}}]'
    )
    captured: dict[str, object] = {}
    saved = {
        "id": "custom_drive",
        "label": "Custom Drive",
        "adapter": "manifest_dataset",
        "dataset_root": str(tmp_path / "configs" / "datasets" / "custom_drive"),
        "sequences": ["clip_001"],
    }

    def fake_save_dataset_manifest(**kwargs):
        captured.update(kwargs)
        return saved

    def fake_refresh_catalogs() -> None:
        window.catalog["dataset_manifests"] = [saved]
        window._fill_dataset_manifest_combo()

    monkeypatch.setattr(services, "save_dataset_manifest", fake_save_dataset_manifest)
    monkeypatch.setattr(window, "refresh_catalogs", fake_refresh_catalogs)

    window.save_dataset_manifest_from_gui()

    assert captured["dataset_id"] == "Custom Drive"
    assert captured["dataset_root"] == str(dataset_root)
    assert captured["sequences"][0]["id"] == "clip_001"
    assert captured["sequences"][0]["assets"]["front_rgb"] == "images/*.png"
    assert window.dataset_catalog_combo.currentData()["id"] == "custom_drive"
    assert window.adapter_edit.text() == "manifest_dataset"
    assert window.sequence_combo.currentText() == "clip_001"
    window.close()


def test_gui_auto_detects_dataset_sequences(monkeypatch, tmp_path) -> None:
    _ensure_app()
    window = MainWindow()
    dataset_root = tmp_path / "drive_dataset"
    dataset_root.mkdir()
    suggested = [
        {
            "id": "clip_001",
            "root": "clip_001",
            "pose_csv": "poses.csv",
            "assets": {"front_rgb": "images/*.png"},
        }
    ]
    monkeypatch.setattr(services, "suggest_dataset_manifest_sequences", lambda root: suggested)
    window.dataset_root_edit.setText(str(dataset_root))

    window.suggest_dataset_manifest_sequences_from_gui()

    assert '"id": "clip_001"' in window.dataset_manifest_sequences_edit.toPlainText()
    assert window.dataset_manifest_name_edit.text() == "drive_dataset"
    assert window.adapter_edit.text() == "manifest_dataset"
    assert "suggested_sequences" in window.dataset_summary.toPlainText()
    window.close()


def test_gui_saves_and_applies_training_config(monkeypatch, tmp_path) -> None:
    _ensure_app()
    monkeypatch.setattr(services, "TRAINING_CONFIGS_PATH", tmp_path / "training_configs.json")
    window = MainWindow()
    window.training_config_name_edit.setText("Custom Tiny Train")
    window.dataset_root_edit.setText("D:/datasets/custom_drive")
    window.adapter_edit.setText("manifest_dataset")
    window.sequence_combo.setCurrentText("clip_001")
    window.home_model_combo.setCurrentText("outputs/models/custom_tiny")
    window._select_training_preset("tiny_world_model")
    window.trainer_params_edit.setPlainText('{"epochs": 5}')

    window.save_training_config()

    data = window.training_config_combo.currentData()
    assert data["id"] == "custom_tiny_train"
    assert data["training_preset_id"] == "tiny_world_model"
    assert window.dataset_root_edit.text() == "D:/datasets/custom_drive"
    assert window.adapter_edit.text() == "manifest_dataset"
    assert window.sequence_combo.currentText() == "clip_001"
    assert window.home_model_combo.currentText() == "outputs/models/custom_tiny"
    assert '"epochs": 5' in window.trainer_params_edit.toPlainText()
    window.close()


def test_gui_can_start_training_from_script_path_without_manifest(monkeypatch, tmp_path) -> None:
    _ensure_app()
    script = tmp_path / "train_any.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "run"
    window = MainWindow()
    window.training_config_name_edit.setText("Script Config")
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("manifest_dataset")
    window.sequence_combo.setCurrentText("clip_001")
    window.training_output_edit.setText(str(output_dir))
    window.trainer_entrypoint_edit.setText(str(script))
    window.trainer_params_edit.setPlainText('{"epochs": 5, "learning_rate": 0.01}')
    window.trainer_schema_edit.setPlainText("")
    window.trainer_arguments_edit.setPlainText(json.dumps(["{dataset_root}", "--output", "{output_dir}"], indent=2))
    saved = {
        "trainer": {
            "id": "script_config",
            "label": "Script Config",
            "manifest_path": str(tmp_path / "trainers" / "script_config.yaml"),
        },
        "training_config": {
            "id": "script_config",
            "label": "Script Config",
            "training_preset_id": "script_config",
            "dataset_root": str(dataset_root),
            "adapter": "manifest_dataset",
            "sequence_id": "clip_001",
            "output_path": str(output_dir),
            "parameters": {"epochs": 5, "learning_rate": 0.01},
        },
    }
    captured: dict[str, object] = {}

    def fake_save_script_training_config(**kwargs):
        captured["save"] = kwargs
        return saved

    def fake_run_training_config_job(config, **kwargs):
        captured["run_config"] = config
        captured["run_kwargs"] = kwargs
        return {"output_dir": str(output_dir), "training_run_path": str(output_dir / "training_run.json")}

    monkeypatch.setattr(services, "save_script_training_config", fake_save_script_training_config)
    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_script_training_config()

    save_kwargs = captured["save"]
    assert save_kwargs["trainer_entrypoint"] == str(script)
    assert save_kwargs["dataset_root"] == str(dataset_root)
    assert save_kwargs["adapter"] == "manifest_dataset"
    assert save_kwargs["sequence_id"] == "clip_001"
    assert save_kwargs["output_path"] == str(output_dir)
    assert save_kwargs["parameters"] == {"epochs": 5, "learning_rate": 0.01}
    assert save_kwargs["arguments"] is None
    assert captured["run_config"]["training_preset_id"] == "script_config"
    assert captured["run_kwargs"]["trainer_root"] == str((tmp_path / "trainers").resolve())
    window.close()


def test_gui_script_training_leaves_output_empty_for_service_default(monkeypatch, tmp_path) -> None:
    _ensure_app()
    script = tmp_path / "train_any.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    window = MainWindow()
    window.training_config_name_edit.setText("Script Config")
    window.dataset_root_edit.setText(str(dataset_root))
    window.training_output_edit.setText("")
    window.model_path_edit.setText("outputs/existing_checkpoint.ckpt")
    window.trainer_entrypoint_edit.setText(str(script))
    captured: dict[str, object] = {}

    def fake_save_script_training_config(**kwargs):
        captured.update(kwargs)
        return {
            "trainer": {"id": "script_config", "manifest_path": str(tmp_path / "trainers" / "script_config.yaml")},
            "training_config": {"id": "script_config", "training_preset_id": "script_config"},
        }

    monkeypatch.setattr(services, "save_script_training_config", fake_save_script_training_config)

    window.save_script_training_config_from_gui()

    assert captured["output_path"] == ""
    window.close()


def test_gui_training_preset_dispatches_existing_actions(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    calls: list[str] = []

    def fake_run_training_config_job(config, **kwargs):
        calls.append(config["training_preset_id"])
        return {"output_dir": "outputs/models/test", "training_run_path": "outputs/models/test/training_run.json"}

    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    for preset_id in ["stablewm_hdf5", "lewm_cost_model", "tiny_world_model"]:
        for index in range(window.training_preset_combo.count()):
            data = window.training_preset_combo.itemData(index)
            if data["id"] == preset_id:
                window.training_preset_combo.setCurrentIndex(index)
                break
        window.run_training_preset()

    assert calls == ["stablewm_hdf5", "lewm_cost_model", "tiny_world_model"]
    window.close()


def test_gui_training_preset_dispatches_manifest_trainer(monkeypatch, tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text("trainer_id: echo_trainer\nentrypoint: echo_trainer.py\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    window = MainWindow()
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("manifest_dataset")
    window.sequence_combo.setCurrentText("clip_001")
    window.model_path_edit.setText(str(tmp_path / "run"))
    window.trainer_params_edit.setPlainText('{"epochs": 4}')
    window.catalog["training_presets"] = [
        {
            "id": "echo_trainer",
            "label": "Echo Trainer",
            "available": True,
            "manifest_path": str(manifest),
            "parameters": {"epochs": {"type": "int", "default": 3}},
        }
    ]
    window._fill_training_preset_combo()
    captured: dict[str, object] = {}

    def fake_run_training_config_job(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"output_dir": str(tmp_path / "run"), "training_run_path": str(tmp_path / "run" / "training_run.json")}

    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_training_preset()

    assert captured["config"]["training_preset_id"] == "echo_trainer"
    assert captured["config"]["dataset_root"] == str(dataset_root)
    assert captured["config"]["adapter"] == "manifest_dataset"
    assert captured["config"]["sequence_id"] == "clip_001"
    assert captured["config"]["parameters"] == {"epochs": 4}
    assert captured["kwargs"]["trainer_root"] == str(tmp_path)
    window.close()


def test_gui_training_config_dispatches_manifest_trainer_with_saved_output(monkeypatch, tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text("trainer_id: echo_trainer\nentrypoint: echo_trainer.py\n", encoding="utf-8")
    output_dir = tmp_path / "configured_run"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    window = MainWindow()
    window.catalog["training_presets"] = [
        {
            "id": "echo_trainer",
            "label": "Echo Trainer",
            "available": True,
            "manifest_path": str(manifest),
            "parameters": {"epochs": {"type": "int", "default": 3}},
        }
    ]
    window.catalog["training_configs"] = [
        {
            "id": "echo_config",
            "label": "Echo Config",
            "training_preset_id": "echo_trainer",
            "dataset_root": str(dataset_root),
            "adapter": "manifest_dataset",
            "sequence_id": "clip_001",
            "output_path": str(output_dir),
            "parameters": {"epochs": 7},
        }
    ]
    window._fill_training_preset_combo()
    window._fill_training_config_combo()
    window._select_training_config("echo_config")
    captured: dict[str, object] = {}

    def fake_run_training_config_job(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"output_dir": str(output_dir), "training_run_path": str(output_dir / "training_run.json")}

    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_training_preset()

    assert captured["config"]["training_preset_id"] == "echo_trainer"
    assert captured["config"]["dataset_root"] == str(dataset_root)
    assert captured["config"]["adapter"] == "manifest_dataset"
    assert captured["config"]["sequence_id"] == "clip_001"
    assert captured["config"]["output_path"] == str(output_dir)
    assert captured["config"]["parameters"] == {"epochs": 7}
    assert captured["kwargs"]["trainer_root"] == str(tmp_path)
    window.close()


def test_gui_start_training_runs_current_training_config_through_unified_service(monkeypatch, tmp_path) -> None:
    _ensure_app()
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text("trainer_id: echo_trainer\nentrypoint: echo_trainer.py\n", encoding="utf-8")
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "run"
    window = MainWindow()
    window.catalog["training_presets"] = [
        {
            "id": "echo_trainer",
            "label": "Echo Trainer",
            "available": True,
            "manifest_path": str(manifest),
            "parameters": {"epochs": {"type": "int", "default": 3}},
        }
    ]
    window._fill_training_preset_combo()
    window.training_config_name_edit.setText("Echo Config")
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("manifest_dataset")
    window.sequence_combo.setCurrentText("clip_001")
    window.training_output_edit.setText(str(output_dir))
    window.trainer_params_edit.setPlainText('{"epochs": 8}')
    captured: dict[str, object] = {}

    def fake_run_training_config_job(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"output_dir": str(output_dir), "training_run_path": str(output_dir / "training_run.json")}

    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_training_preset()

    row = captured["config"]
    assert row["training_preset_id"] == "echo_trainer"
    assert row["dataset_root"] == str(dataset_root)
    assert row["adapter"] == "manifest_dataset"
    assert row["sequence_id"] == "clip_001"
    assert row["output_path"] == str(output_dir)
    assert row["parameters"] == {"epochs": 8}
    window.close()


def test_gui_imports_training_config_bundle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(services, "TRAINING_CONFIGS_PATH", tmp_path / "training_configs.json")
    monkeypatch.setattr(services, "DATASET_MANIFEST_DIRS", (tmp_path / "catalog" / "datasets",))
    monkeypatch.setattr(services, "TRAINER_MANIFEST_DIRS", (tmp_path / "catalog" / "trainers",))
    dataset_root = tmp_path / "dataset"
    trainer_root = tmp_path / "trainer"
    (dataset_root / "seq").mkdir(parents=True)
    trainer_root.mkdir(parents=True)
    (dataset_root / "dataset_manifest.yaml").write_text(
        """
adapter: manifest_dataset
dataset_id: gui_dataset
display_name: GUI Dataset
sequences:
  - id: clip_001
    root: seq
    assets:
      front_rgb: rgb_{frame_id}.png
""",
        encoding="utf-8",
    )
    (trainer_root / "train.py").write_text("print('{}')\n", encoding="utf-8")
    (trainer_root / "trainer.yaml").write_text(
        """
trainer_id: gui_trainer
display_name: GUI Trainer
runtime: python
entrypoint: train.py
parameters:
  epochs:
    type: int
    default: 1
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    config = tmp_path / "training_config.yaml"
    config.write_text(
        """
id: gui_bundle
label: GUI Bundle
dataset_manifest: dataset/dataset_manifest.yaml
trainer_manifest: trainer/trainer.yaml
sequence_id: clip_001
output_path: outputs/models/gui_bundle
parameters:
  epochs: 6
""",
        encoding="utf-8",
    )
    _ensure_app()
    window = MainWindow()

    monkeypatch.setattr("desktop_app.qt_main.QFileDialog.getOpenFileName", lambda *args, **kwargs: (str(config), ""))

    window.import_training_config()

    row = window.training_config_combo.currentData()
    buttons = [button.text() for button in window.page_stack.widget(1).findChildren(QPushButton)]
    assert "Import training config" in buttons
    assert row["id"] == "gui_bundle"
    assert row["training_preset_id"] == "gui_trainer"
    assert window.training_preset_combo.currentData()["id"] == "gui_trainer"
    assert Path(window.dataset_root_edit.text()).name == "gui_dataset"
    assert window.adapter_edit.text() == "manifest_dataset"
    assert window.sequence_combo.currentText() == "clip_001"
    assert '"epochs": 6' in window.trainer_params_edit.toPlainText()
    window.close()


def test_gui_tiny_training_uses_training_parameters_and_output_path(monkeypatch, tmp_path) -> None:
    _ensure_app()
    output_dir = tmp_path / "tiny_run"
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    window = MainWindow()
    window.dataset_root_edit.setText(str(dataset_root))
    window.adapter_edit.setText("manifest_dataset")
    window.sequence_combo.setCurrentText("clip_001")
    window._select_training_preset("tiny_world_model")
    window.training_output_edit.setText(str(output_dir))
    window.trainer_params_edit.setPlainText('{"ridge": 0.25}')
    captured: dict[str, object] = {}

    def fake_run_training_config_job(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return {"output_dir": str(output_dir), "training_run_path": str(output_dir / "training_run.json")}

    monkeypatch.setattr(services, "run_training_config_job", fake_run_training_config_job)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_training_preset()

    assert captured["config"]["training_preset_id"] == "tiny_world_model"
    assert captured["config"]["dataset_root"] == str(dataset_root)
    assert captured["config"]["output_path"] == str(output_dir)
    assert captured["config"]["adapter"] == "manifest_dataset"
    assert captured["config"]["sequence_id"] == "clip_001"
    assert captured["config"]["parameters"] == {"ridge": 0.25}
    window.close()


def test_gui_training_run_list_loads_selected_summary() -> None:
    _ensure_app()
    window = MainWindow()
    window.catalog["training_runs"] = [
        {
            "run_id": "demo_train",
            "preset_label": "Train tiny world model",
            "status": "completed",
            "artifact_path": "outputs/demo/model",
            "metrics": {"loss": 0.25},
            "history": {"loss": [0.8, 0.25]},
            "logs": {"stdout": "outputs/demo/stdout.log", "stderr": "outputs/demo/stderr.log"},
            "summary": {
                "world_model_config": {
                    "id": "demo_self_supervised_world_model",
                    "model_path": "outputs/demo/model",
                    "validation": {"goal_success": True},
                }
            },
        }
    ]

    window._fill_training_run_list()
    item = window.training_run_list.item(0)
    window._load_selected_training_run(item)

    assert "demo_train" in window.training_run_summary.toPlainText()
    assert "loss" in window.training_run_summary.toPlainText()
    assert "Demo Train" in window.training_run_overview.toPlainText()
    assert "loss: 0.25" in window.training_run_overview.toPlainText()
    assert "artifact: outputs/demo/model" in window.training_run_overview.toPlainText()
    assert "world_model_config: demo_self_supervised_world_model" in window.training_run_overview.toPlainText()
    assert "config_goal_success: true" in window.training_run_overview.toPlainText()
    assert "stdout: outputs/demo/stdout.log" in window.training_run_overview.toPlainText()
    assert "stderr: outputs/demo/stderr.log" in window.training_run_overview.toPlainText()
    assert window.training_curve.history["loss"] == [0.8, 0.25]
    window.close()


def test_gui_training_finished_loads_training_run_record_overview(tmp_path) -> None:
    _ensure_app()
    run_dir = tmp_path / "run"
    record = services.write_training_run_record(
        run_dir,
        preset_id="tiny_world_model",
        status="completed",
        artifact_path=str(run_dir / "model"),
        artifact_type="world_model",
        metrics={"loss": 0.1},
        history={"loss": [0.5, 0.2, 0.1]},
    )
    window = MainWindow()

    window._training_finished({"output_dir": str(run_dir / "model"), "training_run_path": record["path"]})

    assert "Train tiny world model" in window.training_run_overview.toPlainText()
    assert "loss: 0.1" in window.training_run_overview.toPlainText()
    assert window.latest_training_curve.history["loss"] == [0.5, 0.2, 0.1]
    assert window.training_curve.history["loss"] == [0.5, 0.2, 0.1]
    window.close()


def test_gui_training_finished_shows_available_metric_curves(tmp_path) -> None:
    _ensure_app()
    run_dir = tmp_path / "run_metrics"
    record = services.write_training_run_record(
        run_dir,
        preset_id="tiny_world_model",
        status="completed",
        artifact_path=str(run_dir / "model"),
        artifact_type="world_model",
        history={"loss": [0.8, 0.2], "val_loss": [0.9, 0.3]},
    )
    window = MainWindow()

    window._training_finished({"output_dir": str(run_dir / "model"), "training_run_path": record["path"]})

    assert "loss" in window.latest_metric_summary.text()
    assert "val_loss" in window.latest_metric_summary.text()
    assert "loss" in window.training_run_metric_summary.text()
    assert "val_loss" in window.training_run_metric_summary.text()
    window.close()


def test_training_curve_widget_accepts_metric_history() -> None:
    _ensure_app()
    curve = TrainingCurveWidget()

    curve.set_history({"loss": [0.8, 0.4, 0.2], "train_rmse": [0.1]})

    assert curve.history["loss"] == [0.8, 0.4, 0.2]
    assert curve.primary_metric == "loss"


def test_gui_world_model_page_saves_config_for_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    _ensure_app()
    window = MainWindow()
    window.model_config_name_edit.setText("Test LE-WM")
    window.home_model_combo.setCurrentText("outputs/test/model/lewm_cost_object.ckpt")
    window._select_combo_value(window.algorithm_combo, "stablewm_lewm")
    window._select_combo_value(window.world_model_combo, "le_wm")

    window.save_world_model_config()

    ids = [
        window.world_model_config_combo.itemData(index)["id"]
        for index in range(window.world_model_config_combo.count())
    ]
    assert "Test_LE-WM" in ids
    assert window.world_model_config_combo.currentData()["model_path"] == "outputs/test/model/lewm_cost_object.ckpt"
    window.close()


def test_gui_imports_world_model_config_for_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    _ensure_app()
    checkpoint = tmp_path / "external_lewm_object.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    window = MainWindow()

    monkeypatch.setattr("desktop_app.qt_main.QFileDialog.getOpenFileName", lambda *args, **kwargs: (str(checkpoint), ""))

    window.import_world_model_config()

    row = window.world_model_config_combo.currentData()
    buttons = [button.text() for button in window.page_stack.widget(1).findChildren(QPushButton)]
    assert "Import model/checkpoint" in buttons
    assert row["id"] == "external_lewm_object"
    assert row["algorithm"] == "stablewm_lewm"
    assert row["world_model"] == "le_wm"
    assert row["model_path"] == str(checkpoint.resolve())
    assert window.model_path_edit.text() == str(checkpoint.resolve())
    window.close()


def test_gui_imports_world_model_directory_for_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    _ensure_app()
    model_dir = tmp_path / "external_tiny_dir"
    model_dir.mkdir()
    (model_dir / "model.json").write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")
    window = MainWindow()

    monkeypatch.setattr("desktop_app.qt_main.QFileDialog.getExistingDirectory", lambda *args, **kwargs: str(model_dir))

    window.import_world_model_directory_config()

    row = window.world_model_config_combo.currentData()
    buttons = [button.text() for button in window.page_stack.widget(1).findChildren(QPushButton)]
    assert "Import model folder" in buttons
    assert row["id"] == "external_tiny_dir"
    assert row["algorithm"] == "world_model_direct"
    assert row["world_model"] == "tiny_learned"
    assert row["model_path"] == str(model_dir.resolve())
    assert window.model_path_edit.text() == str(model_dir.resolve())
    window.close()


def test_gui_visible_demo_uses_minimum_human_visible_steps(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    captured: dict[str, services.VisibleBeamNGDemoRequest] = {}

    monkeypatch.setattr(services, "run_visible_beamng_demo", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.run_visible_beamng_demo()

    assert captured["request"].max_steps >= 600
    window.close()


def test_gui_closed_loop_uses_beamng_map_request(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    captured: dict[str, services.BeamNGMapLeWMClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_beamng_map_lewm_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.run_beamng_lewm_closed_loop()

    assert captured["request"].collect_steps >= 160
    assert captured["request"].eval_steps >= 80
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_pipeline_finished_reads_closed_loop_evaluation() -> None:
    _ensure_app()
    window = MainWindow()

    window._pipeline_finished(
        {
            "hdf5_path": "map.h5",
            "model_dir": "model",
            "evaluation": {"metrics": {"steps": 12}, "episode_path": ""},
        }
    )

    assert window.metric_cards["steps"].value_label.text() == "12"
    window.close()


def test_gui_pipeline_finished_registers_self_supervised_world_model_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    _ensure_app()
    window = MainWindow()

    window._pipeline_finished(
        {
            "status": "completed",
            "model_dir": "outputs/region_self_supervised/nav_demo/model",
            "task": {"task_id": "nav_demo"},
            "training": {"status": "completed", "model_type": "tiny_learned"},
            "acceptance": {"goal_success": True, "goal_reached": True, "min_goal_distance": 2.0, "final_goal_distance": 3.0, "collision_count": 0},
            "evaluation": {"metrics": {"steps": 12}, "episode_path": ""},
        }
    )

    row = window.world_model_config_combo.currentData()
    assert row["algorithm"] == "world_model_direct"
    assert row["world_model"] == "tiny_learned"
    assert row["model_path"] == "outputs/region_self_supervised/nav_demo/model"
    assert row["validation"]["goal_success"] is True
    assert "nav_demo" in row["id"]
    window.close()


def test_gui_pipeline_finished_shows_region_acceptance_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    _ensure_app()
    window = MainWindow()

    window._pipeline_finished(
        {
            "status": "completed",
            "model_dir": "outputs/region_self_supervised/nav_demo/model",
            "training_run_path": "outputs/region_self_supervised/nav_demo/training_run.json",
            "summary_path": "outputs/region_self_supervised/nav_demo/summary.json",
            "task": {"task_id": "nav_demo"},
            "training": {"status": "completed", "model_type": "tiny_learned"},
            "quality_gate": {"passed": True, "progress_ratio": 0.72, "reason": "passed"},
            "acceptance": {
                "goal_success": True,
                "goal_reached": True,
                "final_goal_reached": True,
                "min_goal_distance": 3.2,
                "final_goal_distance": 4.1,
                "model_controlled": True,
                "collision_count": 0,
            },
            "region_navigation": {
                "evaluation_agent": "world_model_direct",
                "route_free": True,
                "evaluation_route_mode": "route_free",
            },
            "evaluation": {"metrics": {"steps": 40, "collision_count": 0}, "episode_path": ""},
        }
    )

    text = window.beamng_summary.toPlainText()
    assert "Region world-model evaluation" in text
    assert "goal_success: true" in text
    assert "goal_reached: true" in text
    assert "final_goal_reached: true" in text
    assert "model_controlled: true" in text
    assert "min_goal_distance: 3.2" in text
    assert "collision_count: 0" in text
    assert "quality_progress_ratio: 0.72" in text
    assert "model_dir: outputs/region_self_supervised/nav_demo/model" in text
    assert "training_run_path: outputs/region_self_supervised/nav_demo/training_run.json" in text
    window.close()


def test_gui_region_navigation_loop_uses_task_path(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    window.task_path_edit.setText("configs/tasks/beamng_johnson_valley_nav_test.yaml")
    captured: dict[str, services.RegionNavigationClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_region_navigation_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.run_region_navigation_loop()

    assert captured["request"].task_path == "configs/tasks/beamng_johnson_valley_nav_test.yaml"
    assert captured["request"].collect_steps >= 160
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_exposes_region_self_supervised_training(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 5
    window.task_path_edit.setText("configs/tasks/beamng_johnson_valley_nav_test.yaml")
    captured: dict[str, services.RegionSelfSupervisedWorldModelRequest] = {}

    monkeypatch.setattr(services, "run_region_self_supervised_world_model", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.train_region_self_supervised_world_model()
    button_texts = [button.text() for button in window.page_stack.widget(2).findChildren(QPushButton)]

    assert "区域自监督训练 world model" in button_texts
    assert captured["request"].task_path == "configs/tasks/beamng_johnson_valley_nav_test.yaml"
    assert captured["request"].world_model_type == "tiny_learned"
    assert captured["request"].evaluation_agent == "world_model_direct"
    assert captured["request"].evaluation_route_mode == "route_free"
    assert captured["request"].collect_steps >= 1000
    assert captured["request"].collect_rollouts >= 3
    assert captured["request"].min_collection_goal_progress_ratio >= 0.25
    assert captured["request"].collection_coverage_grid_size >= 4
    assert captured["request"].collection_coverage_target_interval == 1
    assert captured["request"].collection_max_target_steps <= 40
    assert captured["request"].eval_steps >= 1000
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_home_start_uses_selected_task_and_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    services.save_world_model_config(
        config_id="test_home_lewm",
        label="Test Home LE-WM",
        algorithm="stablewm_lewm",
        world_model="le_wm",
        model_path="outputs/region_navigation/model/lewm_cost_object.ckpt",
    )
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 9
    window.home_task_combo.setCurrentText("configs/tasks/beamng_johnson_valley_nav_test.yaml")
    window._select_world_model_config("test_home_lewm")
    captured: dict[str, services.RegionNavigationClosedLoopRequest] = {}

    monkeypatch.setattr(services, "run_region_navigation_closed_loop", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.run_home_region_model_test()

    assert captured["request"].task_path == "configs/tasks/beamng_johnson_valley_nav_test.yaml"
    assert captured["request"].algorithm == "stablewm_lewm"
    assert captured["request"].algorithm_model_path == "outputs/region_navigation/model/lewm_cost_object.ckpt"
    assert captured["request"].planner == "navigation_mpc"
    assert captured["request"].collect_steps >= 900
    assert captured["request"].eval_steps >= 900
    assert captured["request"].step_delay_sec == 0.02
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_home_start_uses_direct_world_model_evaluation_for_tiny_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", tmp_path / "world_model_configs.json")
    services.save_world_model_config(
        config_id="tiny_direct",
        label="Tiny Direct",
        algorithm="world_model_direct",
        world_model="tiny_learned",
        model_path="outputs/region_self_supervised/model",
    )
    _ensure_app()
    window = MainWindow()
    window.settings.max_steps = 9
    window.home_task_combo.setCurrentText("configs/tasks/beamng_johnson_valley_nav_test.yaml")
    window._select_world_model_config("tiny_direct")
    captured: dict[str, services.RegionWorldModelEvaluationRequest] = {}

    monkeypatch.setattr(services, "run_region_world_model_evaluation", lambda request: captured.setdefault("request", request))
    monkeypatch.setattr(services, "run_region_navigation_closed_loop", lambda request: captured.setdefault("wrong_request", request))
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: task())

    window.run_home_region_model_test()

    assert "wrong_request" not in captured
    assert captured["request"].task_path == "configs/tasks/beamng_johnson_valley_nav_test.yaml"
    assert captured["request"].world_model_type == "tiny_learned"
    assert captured["request"].world_model_path == "outputs/region_self_supervised/model"
    assert captured["request"].eval_steps >= 900
    assert captured["request"].step_delay_sec == 0.02
    assert captured["request"].close_beamng is False
    window.close()


def test_gui_home_start_respects_non_beamng_backend(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    window._select_combo_value(window.backend_combo, "gym_heightmap")
    window._select_combo_value(window.scenario_combo, "forest_trail_001")
    window._select_combo_value(window.agent_combo, "rule_based")
    captured: dict[str, object] = {}

    def fake_run_episode(request: services.RunRequest) -> dict[str, object]:
        captured["request"] = request
        return {"episode_id": "gym", "metrics": {"steps": 3}, "episode_path": ""}

    monkeypatch.setattr(services, "run_episode_from_request", fake_run_episode)
    monkeypatch.setattr(
        services,
        "run_region_navigation_closed_loop",
        lambda request: (_ for _ in ()).throw(AssertionError("BeamNG region loop should not run")),
    )
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window.run_home_start()

    assert isinstance(captured["request"], services.RunRequest)
    assert captured["request"].backend == "gym_heightmap"
    assert captured["request"].scenario == "forest_trail_001"
    window.close()


def test_gui_navigation_preview_uses_editor_callback(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    captured: dict[str, object] = {}

    def fake_preview(task_path: str, **kwargs):
        captured["task_path"] = task_path
        captured.update(kwargs)
        return {"preview": {"realtime": True}, "analysis": {"start_in_region": True}}

    monkeypatch.setattr(window.navigation_preview_session, "update", fake_preview)
    monkeypatch.setattr(window, "_run_task", lambda task, callback, label, **kwargs: callback(task()))

    window._preview_task_from_editor("configs/tasks/beamng_johnson_valley_nav_test.yaml", "topdown", 120.0)

    assert captured["task_path"] == "configs/tasks/beamng_johnson_valley_nav_test.yaml"
    assert captured["camera_mode"] == "topdown"
    assert captured["camera_height_m"] == 120.0
    assert "preview" in window.beamng_summary.toPlainText()
    window.close()


def test_gui_navigation_preview_coalesces_requests_while_loading(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    started: list[dict[str, object]] = []

    def fake_run_task(task, callback, label, **kwargs):
        started.append({"task": task, "callback": callback, "label": label, "kwargs": kwargs})

    monkeypatch.setattr(window, "_run_task", fake_run_task)
    monkeypatch.setattr(
        window.navigation_preview_session,
        "update",
        lambda task_path, **kwargs: {"task_path": task_path, "kwargs": kwargs, "analysis": {}},
    )

    window._preview_task_from_editor("configs/tasks/beamng_johnson_valley_nav_test.yaml", "topdown", 90.0)
    window._preview_task_from_editor("configs/tasks/beamng_johnson_valley_nav_test.yaml", "orbit", 150.0)

    assert len(started) == 1

    first_payload = started[0]["task"]()
    started[0]["callback"](first_payload)

    assert len(started) == 2
    second_payload = started[1]["task"]()
    assert second_payload["kwargs"]["camera_mode"] == "orbit"
    assert second_payload["kwargs"]["camera_height_m"] == 150.0
    window.close()


def test_navigation_task_dialog_previews_draft_from_same_editor(tmp_path) -> None:
    _ensure_app()
    preview_calls: list[dict[str, object]] = []
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        preview_callback=lambda task_path, camera_mode, camera_height_m: preview_calls.append(
            {"task_path": task_path, "camera_mode": camera_mode, "camera_height_m": camera_height_m}
        ),
    )

    dialog.task_id_edit.setText("draft")
    dialog.level_edit.setText("gridmap_v2")
    dialog.canvas.region = [(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)]
    dialog.canvas.start = (2.0, -170.0, 100.6)
    dialog.canvas.goal = (6.0, -210.0)
    dialog.canvas.route = [(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)]

    dialog.preview_task()

    assert dialog.result() == 0
    assert preview_calls == [
        {"task_path": str(task_path.resolve()), "camera_mode": "topdown", "camera_height_m": 150.0}
    ]
    assert task_path.exists()


def test_navigation_task_dialog_realtime_preview_uses_same_draft(tmp_path) -> None:
    _ensure_app()
    preview_calls: list[dict[str, object]] = []
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        preview_callback=lambda task_path, camera_mode, camera_height_m: preview_calls.append(
            {"task_path": task_path, "camera_mode": camera_mode, "camera_height_m": camera_height_m}
        ),
    )

    dialog.task_id_edit.setText("draft")
    dialog.level_edit.setText("gridmap_v2")
    dialog.canvas.region = [(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)]
    dialog.canvas.start = (2.0, -170.0, 100.6)
    dialog.canvas.goal = (6.0, -210.0)
    dialog.canvas.route = [(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)]
    assert dialog.realtime_preview_check.isChecked() is True
    dialog.preview_height_spin.setValue(120.0)

    dialog._run_realtime_preview()

    assert preview_calls == [
        {"task_path": str(task_path.resolve()), "camera_mode": "topdown", "camera_height_m": 120.0}
    ]
    assert task_path.exists()


def test_navigation_task_dialog_uses_beamng_pose_for_task_points(tmp_path) -> None:
    _ensure_app()
    task_path = tmp_path / "draft.yaml"
    dialog = NavigationTaskDialog(
        str(task_path),
        pose_callback=lambda: {"available": True, "x": 12.5, "y": -34.25, "z": 101.2, "yaw": 0.75},
    )
    dialog.canvas.clear_region()
    dialog.canvas.clear_route()

    dialog.refresh_beamng_pose()
    dialog._use_beamng_pose_as_region_point()
    dialog._use_beamng_pose_as_start()
    dialog._use_beamng_pose_as_goal()
    dialog._use_beamng_pose_as_route_point()

    assert dialog.current_beamng_pose["available"] is True
    assert "12.500" in dialog.beamng_pose_label.text()
    assert dialog.canvas.beamng_pose == (12.5, -34.25)
    assert dialog.canvas.region == [(12.5, -34.25)]
    assert dialog.canvas.start == (12.5, -34.25, 101.2)
    assert dialog.start_z_spin.value() == 101.2
    assert dialog.start_yaw_spin.value() == 0.75
    assert dialog.canvas.goal == (12.5, -34.25)
    assert dialog.canvas.route == [(12.5, -34.25), (12.5, -34.25), (12.5, -34.25)]


def test_navigation_task_dialog_applies_beamng_window_picks_by_current_mode(tmp_path) -> None:
    _ensure_app()
    task_path = tmp_path / "draft.yaml"
    picks = iter(
        [
            {"available": True, "sequence": 1, "x": 10.0, "y": 20.0, "z": 30.0},
            {"available": True, "sequence": 2, "x": 11.0, "y": 21.0, "z": 31.0, "yaw": 0.5},
            {"available": True, "sequence": 3, "x": 12.0, "y": 22.0, "z": 32.0},
            {"available": True, "sequence": 4, "x": 13.0, "y": 23.0, "z": 33.0},
        ]
    )
    dialog = NavigationTaskDialog(str(task_path), pick_callback=lambda: next(picks))
    dialog.canvas.clear_region()
    dialog.canvas.clear_route()

    dialog.canvas.set_mode("region")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("start")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("goal")
    dialog._poll_beamng_pick()
    dialog.canvas.set_mode("route")
    dialog._poll_beamng_pick()

    assert dialog.canvas.region == [(10.0, 20.0)]
    assert dialog.canvas.start == (11.0, 21.0, 31.0)
    assert dialog.start_z_spin.value() == 31.0
    assert dialog.start_yaw_spin.value() == 0.5
    assert dialog.canvas.goal == (12.0, 22.0)
    assert dialog.canvas.route == [(11.0, 21.0), (12.0, 22.0), (13.0, 23.0)]
    assert "sequence=4" in dialog.beamng_pick_label.text()


def test_navigation_task_dialog_mode_buttons_toggle_and_highlight(tmp_path) -> None:
    _ensure_app()
    dialog = NavigationTaskDialog(str(tmp_path / "draft.yaml"))

    assert dialog.canvas.mode == "region"
    assert dialog.mode_buttons["region"].isCheckable()
    assert dialog.mode_buttons["region"].isChecked()
    assert not dialog.mode_buttons["start"].isChecked()

    dialog.mode_buttons["region"].click()

    assert dialog.canvas.mode is None
    assert all(not button.isChecked() for button in dialog.mode_buttons.values())

    dialog.mode_buttons["goal"].click()

    assert dialog.canvas.mode == "goal"
    assert dialog.mode_buttons["goal"].isChecked()
    assert not dialog.mode_buttons["region"].isChecked()
    dialog.close()


def test_region_task_editor_opens_non_modal(monkeypatch) -> None:
    _ensure_app()
    window = MainWindow()
    calls = {"exec": 0}

    def fake_exec(self: NavigationTaskDialog) -> QDialog.DialogCode:
        calls["exec"] += 1
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(NavigationTaskDialog, "exec", fake_exec)

    window.open_region_task_editor()

    assert calls["exec"] == 0
    assert window.region_task_dialog is not None
    assert window.region_task_dialog.windowModality() == Qt.WindowModality.NonModal
    window.region_task_dialog.close()
    window.close()


def test_navigation_task_canvas_drag_moves_region_point_without_adding() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF) -> None:
            self._point = point

        def position(self) -> QPointF:
            return self._point

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (9.0, 9.0)]
    canvas._fit_bounds()
    canvas.set_mode("region")

    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((0.0, 0.0))))
    canvas.mouseMoveEvent(FakeMouseEvent(canvas._to_canvas((2.0, 3.0))))
    canvas.mouseReleaseEvent(FakeMouseEvent(canvas._to_canvas((2.0, 3.0))))

    assert len(canvas.region) == 4
    assert canvas.region[0] == (2.0, 3.0)


def test_navigation_task_canvas_right_click_deletes_active_region_point() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
            self._point = point
            self._button = button

        def position(self) -> QPointF:
            return self._point

        def button(self) -> Qt.MouseButton:
            return self._button

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (9.0, 9.0)]
    canvas._fit_bounds()
    canvas.set_mode("region")

    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((10.0, 0.0)), Qt.MouseButton.RightButton))

    assert canvas.region == [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_navigation_task_canvas_right_click_deletes_start_and_goal_points() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
            self._point = point
            self._button = button

        def position(self) -> QPointF:
            return self._point

        def button(self) -> Qt.MouseButton:
            return self._button

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    canvas.start = (1.0, 1.0, 0.0)
    canvas.goal = (9.0, 9.0)
    canvas.route = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]
    canvas._fit_bounds()

    canvas.set_mode("start")
    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((1.0, 1.0)), Qt.MouseButton.RightButton))
    canvas.set_mode("goal")
    canvas.mousePressEvent(FakeMouseEvent(canvas._to_canvas((9.0, 9.0)), Qt.MouseButton.RightButton))

    assert canvas.start is None
    assert canvas.goal is None
    assert canvas.route == [(5.0, 5.0)]


def test_navigation_task_canvas_ignores_clicks_without_active_mode() -> None:
    _ensure_app()

    class FakeMouseEvent:
        def __init__(self, point: QPointF) -> None:
            self._point = point

        def position(self) -> QPointF:
            return self._point

    canvas = NavigationTaskCanvas()
    canvas.resize(480, 360)
    canvas.region = []
    canvas.route = []
    canvas.set_mode(None)

    canvas.mousePressEvent(FakeMouseEvent(QPointF(120.0, 120.0)))

    assert canvas.region == []
    assert canvas.route == []


def test_navigation_task_canvas_preserves_world_aspect_ratio() -> None:
    _ensure_app()
    canvas = NavigationTaskCanvas()
    canvas.resize(640, 360)
    canvas.bounds = (0.0, 100.0, 0.0, 100.0)

    origin = canvas._to_canvas((0.0, 0.0))
    east = canvas._to_canvas((100.0, 0.0))
    north = canvas._to_canvas((0.0, 100.0))
    roundtrip = canvas._from_canvas(canvas._to_canvas((25.0, 75.0)))

    east_distance = abs(east.x() - origin.x())
    north_distance = abs(north.y() - origin.y())
    assert abs(east_distance - north_distance) < 1e-6
    assert roundtrip == (25.0, 75.0)


def test_navigation_task_dialog_delays_invalid_region_warning_until_save(tmp_path, monkeypatch) -> None:
    _ensure_app()
    task_path = tmp_path / "invalid.yaml"
    dialog = NavigationTaskDialog(str(task_path), preview_callback=lambda *_args: None)
    dialog.canvas.region = [(0.0, 0.0), (1.0, 0.0)]
    warnings: list[str] = []
    monkeypatch.setattr("desktop_app.qt_main.QMessageBox.warning", lambda *_args: warnings.append(str(_args[-1])))

    dialog.preview_task()
    assert warnings == []

    dialog.save_task()
    assert warnings
