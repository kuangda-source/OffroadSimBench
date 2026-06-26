from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from unittest.mock import Mock, patch

from offroad_sim.datasets import create_mock_orfd_dataset
from offroad_sim.tasks import load_navigation_region_task
from offroad_sim.utils.yaml_io import load_yaml_file

from desktop_app import services


def _write_manifest_dataset(root: Path) -> Path:
    sequence = root / "seq_001"
    sequence.mkdir(parents=True)
    (sequence / "poses.csv").write_text(
        "frame_id,timestamp,x,y,z,yaw,speed\n"
        "000001,0.0,0.0,0.0,0.0,0.0,1.0\n"
        "000002,0.1,1.0,0.0,0.0,0.0,1.2\n",
        encoding="utf-8",
    )
    (sequence / "rgb_000001.png").write_bytes(b"fake")
    (sequence / "rgb_000002.png").write_bytes(b"fake")
    manifest = root / "dataset_manifest.yaml"
    manifest.write_text(
        """
adapter: manifest_dataset
dataset_id: custom_drive
display_name: Custom Drive
dataset_type: camera_pose
sequences:
  - id: clip_001
    root: seq_001
    pose_csv: poses.csv
    assets:
      front_rgb: rgb_{frame_id}.png
""",
        encoding="utf-8",
    )
    return manifest


def test_desktop_catalog_snapshot_exposes_runtime_choices() -> None:
    catalog = services.catalog_snapshot()

    assert any(item["name"] == "gym_heightmap" for item in catalog["backends"])
    assert any(item["name"] == "world_model" for item in catalog["agents"])
    assert any(item["name"] == "world_model_cem" for item in catalog["planners"])
    assert any(item["name"] == "tiny_learned" for item in catalog["world_models"])
    assert any(item["name"] == "local_lewm_cost" for item in catalog["algorithms"])
    assert "navigation_tasks" in catalog
    assert "model_checkpoints" in catalog
    assert "world_model_configs" in catalog
    assert "dataset_manifests" in catalog
    assert "training_configs" in catalog
    assert "training_presets" in catalog
    assert "training_runs" in catalog


def test_import_dataset_manifest_rewrites_relative_roots_and_registers_dataset(tmp_path) -> None:
    source_root = tmp_path / "external_dataset"
    source = _write_manifest_dataset(source_root)
    destination_root = tmp_path / "configs" / "datasets"

    row = services.import_dataset_manifest(source, destination_root=destination_root)

    installed_root = destination_root / "custom_drive"
    installed_manifest = installed_root / "dataset_manifest.yaml"
    copied = load_yaml_file(installed_manifest)
    assert row["id"] == "custom_drive"
    assert row["label"] == "Custom Drive"
    assert row["adapter"] == "manifest_dataset"
    assert row["dataset_root"] == str(installed_root.resolve())
    assert copied["imported_from"] == str(source.resolve())
    assert copied["sequences"][0]["root"] == str((source_root / "seq_001").resolve())

    entries = services.dataset_manifest_entries(destination_root)
    assert entries[0]["id"] == "custom_drive"
    inspected = services.inspect_dataset(row["dataset_root"], adapter=row["adapter"], sequence_id="clip_001")
    assert inspected["dataset_id"] == "custom_drive"
    assert inspected["frame_count"] == 2


def test_training_preset_entries_include_available_and_future_models() -> None:
    presets = {row["id"]: row for row in services.training_preset_entries()}

    assert presets["stablewm_hdf5"]["available"] is True
    assert presets["lewm_cost_model"]["available"] is True
    assert presets["tiny_world_model"]["available"] is True
    assert presets["lewm_full_self_supervised"]["available"] is False
    assert presets["lewm_full_self_supervised"]["status"] == services.UNFINISHED_TEXT
    assert presets["tdmpc2_adapter"]["available"] is False
    assert presets["dreamerv3_adapter"]["available"] is False


def test_training_run_record_is_discoverable(tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "demo_train"
    record = services.write_training_run_record(
        run_dir,
        preset_id="tiny_world_model",
        status="completed",
        dataset_root="dataset",
        adapter="orfd",
        sequence_id="training/seq_0001",
        artifact_path=str(run_dir / "model.json"),
        artifact_type="world_model",
        metrics={"loss": 0.25},
        parameters={"ridge": 1e-4},
    )

    record_path = run_dir / services.TRAINING_RUN_FILENAME
    runs = services.training_run_entries(tmp_path)

    assert record["path"] == str(record_path.resolve())
    assert record_path.exists()
    assert runs[0]["preset_id"] == "tiny_world_model"
    assert runs[0]["dataset_root"] == "dataset"
    assert runs[0]["sequence_id"] == "training/seq_0001"
    assert runs[0]["artifact_type"] == "world_model"
    assert runs[0]["metrics"]["loss"] == 0.25
    assert runs[0]["path"] == str(record_path.resolve())


def test_training_run_record_preserves_history_for_curves(tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "curve_train"

    services.write_training_run_record(
        run_dir,
        preset_id="lewm_cost_model",
        status="completed",
        artifact_path=str(run_dir / "model.ckpt"),
        artifact_type="checkpoint",
        metrics={"final_loss": 0.2, "train_rmse": 0.1},
        history={"loss": [0.8, 0.45, 0.2]},
    )

    run = services.training_run_entries(tmp_path)[0]
    history = services.training_metric_history(run)

    assert run["history"]["loss"] == [0.8, 0.45, 0.2]
    assert history["loss"] == [0.8, 0.45, 0.2]
    assert history["final_loss"] == [0.2]
    assert history["train_rmse"] == [0.1]


def test_world_model_config_save_and_list(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"

    defaults = services.world_model_config_entries(config_path)
    assert any(row["id"] == services.DEFAULT_WORLD_MODEL_CONFIG_ID for row in defaults)

    saved = services.save_world_model_config(
        config_id="my_lewm_config",
        label="My LE-WM Config",
        algorithm="stablewm_lewm",
        world_model="le_wm",
        model_path="outputs/region_navigation/model/lewm_cost_object.ckpt",
        path=config_path,
    )
    rows = services.world_model_config_entries(config_path)
    row = next(item for item in rows if item["id"] == "my_lewm_config")

    assert saved["id"] == "my_lewm_config"
    assert row["label"] == "My LE-WM Config"
    assert row["algorithm"] == "stablewm_lewm"
    assert row["world_model"] == "le_wm"
    assert row["model_path"] == "outputs/region_navigation/model/lewm_cost_object.ckpt"


def test_import_world_model_config_infers_tiny_learned_model_json(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"
    model_dir = tmp_path / "external_tiny"
    model_dir.mkdir()
    model_path = model_dir / "model.json"
    model_path.write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")

    row = services.import_world_model_config(model_path, path=config_path)

    rows = services.world_model_config_entries(config_path)
    saved = next(item for item in rows if item["id"] == "external_tiny")
    assert row["id"] == "external_tiny"
    assert saved["algorithm"] == "world_model_direct"
    assert saved["world_model"] == "tiny_learned"
    assert saved["model_path"] == str(model_path.resolve())


def test_import_world_model_config_accepts_model_directory(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"
    model_dir = tmp_path / "external_tiny_dir"
    model_dir.mkdir()
    (model_dir / "model.json").write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")

    row = services.import_world_model_config(model_dir, path=config_path)

    assert row["id"] == "external_tiny_dir"
    assert row["algorithm"] == "world_model_direct"
    assert row["world_model"] == "tiny_learned"
    assert row["model_path"] == str(model_dir.resolve())


def test_import_world_model_config_defaults_checkpoint_to_lewm(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"
    checkpoint = tmp_path / "custom_lewm_object.ckpt"
    checkpoint.write_bytes(b"checkpoint")

    row = services.import_world_model_config(checkpoint, label="Custom LE-WM", path=config_path)

    assert row["id"] == "Custom_LE-WM"
    assert row["algorithm"] == "stablewm_lewm"
    assert row["world_model"] == "le_wm"
    assert row["model_path"] == str(checkpoint.resolve())


def test_training_config_save_and_list(tmp_path) -> None:
    config_path = tmp_path / "training_configs.json"

    defaults = services.training_config_entries(config_path)
    default_ids = {row["id"] for row in defaults}
    assert "orfd_stablewm_hdf5" in default_ids
    assert "orfd_tiny_world_model" in default_ids

    saved = services.save_training_config(
        config_id="My Custom Train",
        label="My Custom Train",
        training_preset_id="echo_trainer",
        dataset_root="D:/datasets/custom_drive",
        adapter="manifest_dataset",
        sequence_id="clip_001",
        output_path="outputs/models/custom_echo",
        parameters={"epochs": 4, "batch_size": 8},
        path=config_path,
    )
    rows = services.training_config_entries(config_path)
    row = next(item for item in rows if item["id"] == "my_custom_train")

    assert saved["id"] == "my_custom_train"
    assert row["label"] == "My Custom Train"
    assert row["training_preset_id"] == "echo_trainer"
    assert row["dataset_root"] == "D:/datasets/custom_drive"
    assert row["adapter"] == "manifest_dataset"
    assert row["sequence_id"] == "clip_001"
    assert row["output_path"] == "outputs/models/custom_echo"
    assert row["parameters"] == {"epochs": 4, "batch_size": 8}


def test_import_training_config_installs_dataset_and_trainer_manifests(tmp_path) -> None:
    source_root = tmp_path / "bundle"
    dataset_root = source_root / "dataset"
    trainer_root = source_root / "trainer"
    dataset_root.mkdir(parents=True)
    trainer_root.mkdir(parents=True)
    dataset_manifest = _write_manifest_dataset(dataset_root)
    trainer_script = trainer_root / "train.py"
    trainer_script.write_text("print('{}')\n", encoding="utf-8")
    trainer_manifest = trainer_root / "trainer.yaml"
    trainer_manifest.write_text(
        """
trainer_id: bundle_trainer
display_name: Bundle Trainer
runtime: python
entrypoint: train.py
parameters:
  epochs:
    type: int
    default: 2
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    training_config = source_root / "training_config.yaml"
    training_config.write_text(
        """
id: bundle_config
label: Bundle Config
dataset_manifest: dataset/dataset_manifest.yaml
trainer_manifest: trainer/trainer.yaml
sequence_id: clip_001
output_path: outputs/models/bundle_config
parameters:
  epochs: 5
""",
        encoding="utf-8",
    )

    row = services.import_training_config(
        training_config,
        path=tmp_path / "training_configs.json",
        dataset_destination_root=tmp_path / "datasets",
        trainer_destination_root=tmp_path / "trainers",
    )

    installed_dataset = load_yaml_file(tmp_path / "datasets" / "custom_drive" / "dataset_manifest.yaml")
    installed_trainer = load_yaml_file(tmp_path / "trainers" / "bundle_trainer.yaml")
    configs = services.training_config_entries(tmp_path / "training_configs.json")

    assert row["id"] == "bundle_config"
    assert row["training_preset_id"] == "bundle_trainer"
    assert row["dataset_root"] == str((tmp_path / "datasets" / "custom_drive").resolve())
    assert row["adapter"] == "manifest_dataset"
    assert row["sequence_id"] == "clip_001"
    assert row["parameters"] == {"epochs": 5}
    assert installed_dataset["imported_from"] == str(dataset_manifest.resolve())
    assert installed_trainer["entrypoint"] == str(trainer_script.resolve())
    assert any(config["id"] == "bundle_config" for config in configs)


def test_desktop_services_list_navigation_tasks_and_checkpoints(tmp_path) -> None:
    task_path = tmp_path / "task.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(task_path),
            task_id="menu_task",
            level="johnson_valley",
            region_polygon=[(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)],
            start_pos=(2.0, 2.0, 1.0),
            goal_pos=(20.0, 20.0),
            expert_route=[(2.0, 2.0), (20.0, 20.0)],
        )
    )
    checkpoint = tmp_path / "runs" / "model" / "lewm_cost_object.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    tasks = services.navigation_task_entries(tmp_path)
    checkpoints = services.model_checkpoint_entries(tmp_path)

    assert tasks[0]["id"] == "menu_task"
    assert tasks[0]["path"] == str(task_path.resolve())
    assert checkpoints[0]["path"] == str(checkpoint.resolve())
    assert checkpoints[0]["label"] == "runs/model/lewm_cost_object.ckpt"


def test_desktop_request_builds_dataset_and_planner_options() -> None:
    request = services.RunRequest(
        backend="dataset_replay",
        agent="world_model",
        dataset_root="outputs/mock_orfd_phase3",
        adapter="orfd",
        sequence_id="training/seq_0001",
        load_assets=True,
        world_model_type="tiny_learned",
        world_model_path="outputs/models/phase3_tiny_world_model",
        planner="world_model_cem",
        planner_horizon=4,
        planner_samples=16,
        planner_iterations=2,
    )

    assert services.backend_options(request) == {
        "dataset_root": "outputs/mock_orfd_phase3",
        "sequence_id": "training/seq_0001",
        "adapter": "orfd",
        "load_assets": True,
    }
    assert services.agent_options(request)["planner_config"] == {
        "horizon": 4,
        "num_samples": 16,
        "iterations": 2,
    }
    assert services.scenario_for_request(request)["scenario_id"] == "dataset_mock_orfd_phase3"


def test_desktop_display_value_uses_nan_for_missing_values() -> None:
    assert services.display_value(math.nan) == "NaN"
    assert services.display_value(None) == "NaN"
    assert services.display_value(False) == "false"


def test_desktop_lewm_command_services_require_paths() -> None:
    try:
        services.export_lewm_hdf5("", "out.h5")
    except ValueError as exc:
        assert "Dataset root" in str(exc)
    else:
        raise AssertionError("Expected dataset root validation.")

    try:
        services.train_lewm_cost_model("", "out")
    except ValueError as exc:
        assert "Input HDF5" in str(exc)
    else:
        raise AssertionError("Expected HDF5 path validation.")


def test_run_json_command_ignores_surrounding_logs() -> None:
    completed = Mock(returncode=0, stdout="log before\n{\"ok\": true}\nlog after", stderr="")
    with patch("desktop_app.services.subprocess.run", return_value=completed):
        assert services._run_json_command(["python", "script.py"]) == {"ok": True}


def test_desktop_preview_and_terrain_draft_export(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "orfd", frame_count=3)

    preview = services.preview_dataset_frame(str(dataset_root), adapter="orfd", sequence_id="training/seq_0001")
    assert preview["frame_id"] == "000000"
    assert "front_rgb" in preview["previews"]

    terrain = services.export_orfd_beamng_terrain_draft(
        str(dataset_root),
        adapter="orfd",
        sequence_id="training/seq_0001",
        output_dir=tmp_path / "terrain",
        grid_size=16,
    )
    assert terrain["status"] == "draft_ready"
    assert terrain["beamng_import_ready"] is False
    assert (tmp_path / "terrain" / "heightmap.png").exists()
    assert (tmp_path / "terrain" / "terrain_mesh.obj").exists()


def test_train_tiny_world_model_writes_training_run_record(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "orfd", frame_count=3)
    output_dir = tmp_path / "tiny_model"

    payload = services.train_tiny_world_model(
        str(dataset_root),
        str(output_dir),
        adapter="orfd",
        sequence_id="training/seq_0001",
    )

    record_path = output_dir / services.TRAINING_RUN_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert payload["training_run_path"] == str(record_path.resolve())
    assert record["preset_id"] == "tiny_world_model"
    assert record["artifact_path"] == str(output_dir.resolve())
    assert record["sequence_id"] == "training/seq_0001"


def test_export_lewm_hdf5_writes_training_run_record(tmp_path) -> None:
    output_hdf5 = tmp_path / "stablewm" / "orfd.h5"
    with patch("desktop_app.services._run_json_command", return_value={"output_hdf5": str(output_hdf5), "frame_count": 3}):
        payload = services.export_lewm_hdf5(
            "dataset",
            str(output_hdf5),
            adapter="orfd",
            sequence_id="training/seq_0001",
            image_size=32,
        )

    record_path = output_hdf5.with_suffix("") / services.TRAINING_RUN_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert payload["training_run_path"] == str(record_path.resolve())
    assert record["preset_id"] == "stablewm_hdf5"
    assert record["artifact_path"] == str(output_hdf5.resolve())
    assert record["artifact_type"] == "hdf5"
    assert record["parameters"]["image_size"] == 32


def test_train_lewm_cost_model_writes_training_run_record(tmp_path) -> None:
    output_dir = tmp_path / "lewm_cost"
    checkpoint = output_dir / "lewm_cost_object.ckpt"
    with patch(
        "desktop_app.services._run_json_command",
        return_value={"output_dir": str(output_dir), "checkpoint_path": str(checkpoint), "loss": 0.1},
    ):
        payload = services.train_lewm_cost_model("input.h5", str(output_dir))

    record_path = output_dir / services.TRAINING_RUN_FILENAME
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert payload["training_run_path"] == str(record_path.resolve())
    assert record["preset_id"] == "lewm_cost_model"
    assert record["artifact_path"] == str(checkpoint.resolve())
    assert record["artifact_type"] == "checkpoint"
    assert record["parameters"]["input_hdf5"] == "input.h5"


def test_train_lewm_cost_model_copies_source_hdf5_record(tmp_path) -> None:
    hdf5_path = tmp_path / "stablewm" / "orfd_small.h5"
    services.write_training_run_record(
        hdf5_path.with_suffix(""),
        preset_id="stablewm_hdf5",
        status="completed",
        dataset_root="dataset_root",
        adapter="orfd",
        sequence_id="training/seq_0001",
        artifact_path=str(hdf5_path),
        artifact_type="hdf5",
        metrics={"total_frames": 6},
    )
    output_dir = tmp_path / "lewm_cost"
    checkpoint = output_dir / "lewm_cost_object.ckpt"
    with patch(
        "desktop_app.services._run_json_command",
        return_value={"output_dir": str(output_dir), "checkpoint_path": str(checkpoint)},
    ):
        services.train_lewm_cost_model(str(hdf5_path), str(output_dir))

    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))

    assert record["dataset_root"] == "dataset_root"
    assert record["adapter"] == "orfd"
    assert record["sequence_id"] == "training/seq_0001"
    assert record["summary"]["source_training_run_path"].endswith("training_run.json")


def test_save_manual_navigation_task_writes_valid_task(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"

    payload = services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            goal_radius=7.5,
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
            evaluation_drive_mode="manual",
        )
    )

    task = load_navigation_region_task(output)
    assert payload["task_path"] == str(output.resolve())
    assert task.task_id == "manual_region_test"
    assert task.contains_point((2.0, -170.0)) is True
    assert task.contains_point((6.0, -210.0)) is True
    assert task.beamng["evaluation_drive_mode"] == "manual"
    assert task.cost["out_of_region_weight"] == 250.0
    assert task.to_beamng_scenario(mode="evaluation")["metadata"]["beamng"]["drive_mode"] == "manual"


def test_save_manual_navigation_task_rejects_start_outside_region(tmp_path) -> None:
    try:
        services.save_manual_navigation_task(
            services.ManualNavigationTaskRequest(
                output_path=str(tmp_path / "bad.yaml"),
                task_id="bad",
                region_polygon=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
                start_pos=(8.0, 1.0, 100.6),
                goal_pos=(1.0, 1.0),
            )
        )
    except ValueError as exc:
        assert "inside the selected region" in str(exc)
    else:
        raise AssertionError("Expected manual task validation failure.")


def test_navigation_task_analysis_checks_region_and_route(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )

    payload = services.analyze_navigation_task(str(output))

    assert payload["start_in_region"] is True
    assert payload["goal_in_region"] is True
    assert payload["route_in_region"] is True
    assert payload["route_waypoint_count"] == 3
    assert payload["route_length_m"] > payload["straight_line_m"]


def test_preview_navigation_task_in_beamng_uses_manual_preview(tmp_path) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )
    with patch("desktop_app.services.run_episode") as run_episode:
        run_episode.return_value.to_dict.return_value = {"episode_id": "preview", "metrics": {"drive_mode": "manual"}}

        payload = services.preview_navigation_task_in_beamng(str(output), hold_open_sec=1.0)

    scenario = run_episode.call_args.kwargs["scenario"]
    assert payload["analysis"]["start_in_region"] is True
    assert run_episode.call_args.kwargs["agent_name"] == "stop"
    assert run_episode.call_args.kwargs["close_backend"] is False
    assert scenario["metadata"]["beamng"]["drive_mode"] == "manual"
    assert scenario["metadata"]["beamng"]["preview_mode"] is True
    assert scenario["metadata"]["beamng"]["camera_mode"] == "topdown"
    assert scenario["metadata"]["beamng"]["camera_height_m"] == 150.0
    assert scenario["metadata"]["beamng"]["route"] == [[2.0, -170.0], [5.0, -190.0], [6.0, -210.0]]


def test_realtime_navigation_preview_session_reuses_backend(tmp_path, monkeypatch) -> None:
    output = tmp_path / "manual_region.yaml"
    services.save_manual_navigation_task(
        services.ManualNavigationTaskRequest(
            output_path=str(output),
            task_id="manual_region_test",
            level="gridmap_v2",
            region_polygon=[(0.0, -160.0), (20.0, -160.0), (20.0, -220.0), (0.0, -220.0)],
            start_pos=(2.0, -170.0, 100.6),
            start_yaw=-1.57,
            goal_pos=(6.0, -210.0),
            expert_route=[(2.0, -170.0), (5.0, -190.0), (6.0, -210.0)],
        )
    )
    calls: list[str] = []

    class FakeBackend:
        def __init__(self, *, connection, vehicle_config=None):
            calls.append("init")
            calls.append(f"picker={connection.enable_point_picker}")
            self.metrics = {"level": "gridmap_v2", "route_waypoint_count": 0}

        def reset(self, scenario):
            calls.append("reset")
            self.metrics = {"level": scenario["metadata"]["beamng"]["level"], "route_waypoint_count": len(scenario["metadata"]["beamng"]["route"])}

        def update_navigation_preview(self, scenario):
            calls.append("update")
            self.metrics = {"level": scenario["metadata"]["beamng"]["level"], "route_waypoint_count": len(scenario["metadata"]["beamng"]["route"])}

        def get_metrics(self):
            return dict(self.metrics)

        def consume_point_picker(self):
            return {"available": True, "sequence": 3, "x": 12.5, "y": -34.25, "z": 101.2}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(services, "BeamNGBackend", FakeBackend)
    session = services.BeamNGNavigationPreviewSession()

    first = session.update(str(output), camera_mode="topdown", camera_height_m=100.0)
    second = session.update(str(output), camera_mode="topdown", camera_height_m=120.0)
    pick = session.consume_picker_pick()
    session.close()

    assert calls == ["init", "picker=True", "reset", "update", "close"]
    assert first["preview"]["realtime"] is True
    assert second["metrics"]["route_waypoint_count"] == 3
    assert pick["available"] is True


def test_realtime_navigation_preview_session_reports_current_pose(monkeypatch) -> None:
    class FakeBackend:
        def __init__(self, *, connection, vehicle_config=None):
            pass

        def get_current_vehicle_pose(self):
            return {"available": True, "x": 12.5, "y": -34.25, "z": 101.2, "yaw": 0.75}

        def close(self):
            pass

    session = services.BeamNGNavigationPreviewSession()
    assert session.current_pose()["available"] is False
    session._backend = FakeBackend(connection=None)
    session._level = "johnson_valley"

    pose = session.current_pose()

    assert pose["available"] is True
    assert pose["x"] == 12.5
    assert pose["y"] == -34.25
    assert pose["level"] == "johnson_valley"


def test_realtime_navigation_preview_session_pose_and_picker_do_not_block_while_busy() -> None:
    class FakeBackend:
        def get_current_vehicle_pose(self):
            raise AssertionError("busy preview should not call pose API")

        def consume_point_picker(self):
            raise AssertionError("busy preview should not call picker API")

    session = services.BeamNGNavigationPreviewSession()
    session._backend = FakeBackend()
    session._level = "johnson_valley"
    session._lock = threading.Lock()
    session._lock.acquire()
    try:
        pose = session.current_pose()
        pick = session.consume_picker_pick()
    finally:
        session._lock.release()

    assert pose["available"] is False
    assert pick["available"] is False
    assert "busy" in pose["message"].lower()
    assert "busy" in pick["message"].lower()


def test_orfd_lewm_pipeline_uses_selected_sequence_and_options() -> None:
    with (
        patch("desktop_app.services.inspect_dataset", return_value={"selected_sequence": "training/seq_0001"}),
        patch("desktop_app.services.export_lewm_hdf5", return_value={"output_hdf5": "out.h5"}) as export,
        patch("desktop_app.services.train_lewm_cost_model", return_value={"output_dir": "model"}) as train,
        patch("desktop_app.services.run_episode_from_request", return_value={"metrics": {}, "episode_path": ""}) as run_episode,
    ):
        payload = services.run_orfd_lewm_pipeline(
            services.PipelineRequest(
                dataset_root="dataset",
                sequence_id="training/seq_0001",
                hdf5_path="custom.h5",
                model_dir="model",
                image_size=32,
                run_beamng=False,
            )
        )

    export.assert_called_once()
    assert export.call_args.kwargs["image_size"] == 32
    train.assert_called_once_with("out.h5", "model")
    assert run_episode.call_count == 1
    assert payload["beamng"] is None
