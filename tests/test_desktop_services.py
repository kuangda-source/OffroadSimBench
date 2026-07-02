from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.datasets import create_mock_orfd_dataset
from offroad_sim.replay import EpisodeRecorder
from offroad_sim.tasks import NavigationRegionTask, load_navigation_region_task
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


def test_episode_trace_preserves_action_gear_for_training(tmp_path: Path) -> None:
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": "reverse_episode", "backend": "beamng"})
    for index, gear in enumerate([-1, 1]):
        recorder.record_step(
            observation=Observation(
                timestamp=float(index),
                vehicle_state=VehicleState(x=float(index), y=0.0, z=1.0, yaw=0.0, speed=0.2),
                goal=(10.0, 0.0),
            ),
            action=Action(throttle=0.5, gear=gear),
            reward=0.0,
            done=False,
            info={},
        )
    recorder.end_episode({"steps": 2})
    episode_path = recorder.save(tmp_path / "reverse_episode")
    task = NavigationRegionTask(
        task_id="reverse_task",
        map_id="johnson_valley",
        level="johnson_valley",
        region_polygon=[(-1.0, -1.0), (12.0, -1.0), (12.0, 2.0), (-1.0, 2.0)],
        start_pos=(0.0, 0.0, 1.0),
        start_yaw=0.0,
        goal_pos=(10.0, 0.0),
        goal_radius=2.0,
        expert_route=[(0.0, 0.0), (10.0, 0.0)],
    )

    trace = services.load_episode_trace(episode_path)
    sequence = services._episode_trace_to_dataset_sequence(episode_path, task)

    assert trace[0]["gear"] == -1
    assert sequence.frames[0].action is not None
    assert sequence.frames[0].action.gear == -1


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


def test_save_dataset_manifest_from_directory_registers_generic_dataset(tmp_path) -> None:
    dataset_root = tmp_path / "drive_dataset"
    sequence_root = dataset_root / "clip_001"
    sequence_root.mkdir(parents=True)
    destination_root = tmp_path / "configs" / "datasets"

    row = services.save_dataset_manifest(
        dataset_id="custom_drive",
        display_name="Custom Drive",
        dataset_root=str(dataset_root),
        sequences=[
            {
                "id": "clip_001",
                "root": str(sequence_root),
                "assets": {"front_rgb": "images/*.png", "mask": "masks/*.png"},
            }
        ],
        destination_root=destination_root,
    )

    installed_root = destination_root / "custom_drive"
    installed = load_yaml_file(installed_root / "dataset_manifest.yaml")

    assert row["id"] == "custom_drive"
    assert row["label"] == "Custom Drive"
    assert row["adapter"] == "manifest_dataset"
    assert row["dataset_root"] == str(installed_root.resolve())
    assert row["sequences"] == ["clip_001"]
    assert installed["dataset_type"] == "manifest_dataset"
    assert installed["source_root"] == str(dataset_root.resolve())
    assert installed["sequences"][0]["root"] == str(sequence_root.resolve())
    assert installed["sequences"][0]["assets"]["front_rgb"] == "images/*.png"


def test_suggest_dataset_manifest_sequences_detects_common_driving_layout(tmp_path) -> None:
    dataset_root = tmp_path / "drive_dataset"
    sequence_root = dataset_root / "clip_001"
    (sequence_root / "images").mkdir(parents=True)
    (sequence_root / "depth").mkdir()
    (sequence_root / "masks").mkdir()
    (sequence_root / "images" / "000001.png").write_bytes(b"fake")
    (sequence_root / "depth" / "000001.npy").write_bytes(b"fake")
    (sequence_root / "masks" / "000001.png").write_bytes(b"fake")
    (sequence_root / "poses.csv").write_text("frame_id,timestamp\n000001,0.0\n", encoding="utf-8")
    (sequence_root / "actions.csv").write_text("frame_id,steer\n000001,0.0\n", encoding="utf-8")

    sequences = services.suggest_dataset_manifest_sequences(dataset_root)

    assert sequences == [
        {
            "id": "clip_001",
            "root": "clip_001",
            "pose_csv": "poses.csv",
            "actions_csv": "actions.csv",
            "assets": {
                "front_rgb": "images/*.png",
                "depth": "depth/*.npy",
                "label": "masks/*.png",
            },
        }
    ]


def test_suggest_dataset_manifest_sequences_handles_single_root_sequence(tmp_path) -> None:
    dataset_root = tmp_path / "single_drive"
    (dataset_root / "rgb").mkdir(parents=True)
    (dataset_root / "rgb" / "000001.jpg").write_bytes(b"fake")
    (dataset_root / "poses.csv").write_text("frame_id,timestamp\n000001,0.0\n", encoding="utf-8")

    sequences = services.suggest_dataset_manifest_sequences(dataset_root)

    assert sequences[0]["id"] == "single_drive"
    assert sequences[0]["root"] == "."
    assert sequences[0]["pose_csv"] == "poses.csv"
    assert sequences[0]["assets"] == {"front_rgb": "rgb/*.jpg"}


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


def test_register_training_run_artifact_promotes_world_model_config(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"
    model_dir = tmp_path / "trained_tiny"
    model_dir.mkdir()
    (model_dir / "model.json").write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")
    record = services.write_training_run_record(
        tmp_path / "run",
        preset_id="tiny_world_model",
        status="completed",
        artifact_path=str(model_dir),
        artifact_type="world_model",
        metrics={"loss": 0.05},
        summary={"goal_success": True, "collision_count": 0},
    )

    row = services.register_training_run_artifact_as_world_model_config(
        record["path"],
        label="Promoted Tiny",
        path=config_path,
    )

    assert row["id"] == "Promoted_Tiny"
    assert row["algorithm"] == "world_model_direct"
    assert row["world_model"] == "tiny_learned"
    assert row["model_path"] == str(model_dir.resolve())
    assert row["validation"]["goal_success"] is True
    assert row["validation"]["loss"] == 0.05
    saved = services.world_model_config_entries(config_path)
    assert any(item["id"] == "Promoted_Tiny" for item in saved)
    refreshed = json.loads(Path(record["path"]).read_text(encoding="utf-8"))
    assert refreshed["summary"]["world_model_config"]["id"] == "Promoted_Tiny"


def test_world_model_configs_mark_only_route_free_success_as_demo_ready(tmp_path) -> None:
    config_path = tmp_path / "world_model_configs.json"
    model_dir = tmp_path / "trained_tiny"
    model_dir.mkdir()
    (model_dir / "model.json").write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")
    services.save_world_model_config(
        config_id="route_free_success",
        label="Route Free Success",
        algorithm="world_model_direct",
        world_model="tiny_learned",
        model_path=str(model_dir),
        validation={
            "goal_success": True,
            "route_free": True,
            "route_free_direct": True,
            "model_controlled": True,
            "route_waypoint_count": 0,
            "collision_count": 0,
            "final_goal_distance": 11.8,
            "goal_radius": 12.0,
        },
        path=config_path,
    )
    services.save_world_model_config(
        config_id="experience_corridor_success",
        label="Experience Corridor Success",
        algorithm="world_model_direct",
        world_model="tiny_learned",
        model_path=str(model_dir),
        validation={
            "goal_success": True,
            "route_free": True,
            "route_free_direct": False,
            "experience_corridor": True,
            "model_controlled": True,
            "route_waypoint_count": 0,
            "collision_count": 0,
            "final_goal_distance": 11.8,
            "goal_radius": 12.0,
        },
        path=config_path,
    )
    services.save_world_model_config(
        config_id="trained_only",
        label="Trained Only",
        algorithm="world_model_direct",
        world_model="tiny_learned",
        model_path=str(model_dir),
        validation={"train_rmse": 0.1, "collection_rollout_count": 6},
        path=config_path,
    )

    rows = {row["id"]: row for row in services.world_model_config_entries(config_path)}

    assert rows["route_free_success"]["demo_ready"] is True
    assert rows["experience_corridor_success"]["demo_ready"] is False
    assert rows["trained_only"]["demo_ready"] is False
    assert rows[services.DEFAULT_WORLD_MODEL_CONFIG_ID]["demo_ready"] is True
    assert [row["id"] for row in services.demo_ready_world_model_config_entries(config_path)] == [
        services.DEFAULT_WORLD_MODEL_CONFIG_ID,
        "route_free_success",
    ]


def test_demo_acceptance_rejects_non_demo_ready_world_model_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "world_model_configs.json"
    model_dir = tmp_path / "trained_tiny"
    model_dir.mkdir()
    (model_dir / "model.json").write_text('{"model_type": "tiny_learned", "weights": "weights.npz"}', encoding="utf-8")
    services.save_world_model_config(
        config_id="trained_only",
        label="Trained Only",
        algorithm="world_model_direct",
        world_model="tiny_learned",
        model_path=str(model_dir),
        validation={"train_rmse": 0.1},
        path=config_path,
    )
    monkeypatch.setattr(services, "WORLD_MODEL_CONFIGS_PATH", config_path)
    monkeypatch.setattr(
        services,
        "demo_config_entries",
        lambda: [
            {
                "id": "unsafe_demo",
                "label": "Unsafe Demo",
                "task_path": str(tmp_path / "task.yaml"),
                "world_model_config_id": "trained_only",
                "planner": "navigation_mpc",
            }
        ],
    )

    with pytest.raises(ValueError, match="not demo-ready"):
        services.run_demo_acceptance(services.DemoAcceptanceRequest(demo_config_id="unsafe_demo", max_steps=1))


def test_register_training_run_artifact_rejects_hdf5_export(tmp_path) -> None:
    hdf5 = tmp_path / "dataset.h5"
    hdf5.write_bytes(b"hdf5")
    record = services.write_training_run_record(
        tmp_path / "run",
        preset_id="export_stablewm_hdf5",
        status="completed",
        artifact_path=str(hdf5),
        artifact_type="hdf5",
    )

    with pytest.raises(ValueError, match="not runnable"):
        services.register_training_run_artifact_as_world_model_config(record["path"], path=tmp_path / "configs.json")


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


def test_import_training_config_accepts_inline_trainer_definition(tmp_path) -> None:
    source_root = tmp_path / "inline_bundle"
    source_root.mkdir()
    trainer_script = source_root / "train_inline.py"
    trainer_script.write_text("print('{}')\n", encoding="utf-8")
    training_config = source_root / "training_config.yaml"
    training_config.write_text(
        """
id: inline_config
label: Inline Config
dataset_root: D:/datasets/custom_drive
adapter: manifest_dataset
sequence_id: clip_001
trainer:
  trainer_id: inline_trainer
  display_name: Inline Trainer
  runtime: python
  entrypoint: train_inline.py
  arguments:
    - "{dataset_root}"
    - "--output"
    - "{output_dir}"
    - "--epochs"
    - "{params.epochs}"
  parameters:
    epochs:
      type: int
      default: 3
  outputs:
    artifact_type: checkpoint
output_path: outputs/models/inline_config
parameters:
  epochs: 6
""",
        encoding="utf-8",
    )

    row = services.import_training_config(
        training_config,
        path=tmp_path / "training_configs.json",
        trainer_destination_root=tmp_path / "trainers",
    )

    installed_trainer = load_yaml_file(tmp_path / "trainers" / "inline_trainer.yaml")
    configs = services.training_config_entries(tmp_path / "training_configs.json")

    assert row["training_preset_id"] == "inline_trainer"
    assert row["dataset_root"] == "D:/datasets/custom_drive"
    assert row["adapter"] == "manifest_dataset"
    assert row["parameters"] == {"epochs": 6}
    assert installed_trainer["entrypoint"] == str(trainer_script.resolve())
    assert installed_trainer["arguments"][-1] == "{params.epochs}"
    assert any(config["id"] == "inline_config" and config["training_preset_id"] == "inline_trainer" for config in configs)


def test_run_trainer_manifest_job_reads_sidecar_metrics_without_stdout_json(tmp_path) -> None:
    trainer_script = tmp_path / "train_sidecar.py"
    trainer_script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
(out / "model.ckpt").write_text("checkpoint", encoding="utf-8")
(out / "metrics.json").write_text(json.dumps({"loss": 0.25, "accuracy": 0.75}), encoding="utf-8")
(out / "history.json").write_text(json.dumps({"loss": [0.9, 0.5, 0.25]}), encoding="utf-8")
print("training complete")
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        f"""
trainer_id: sidecar_trainer
display_name: Sidecar Trainer
runtime: python
entrypoint: {trainer_script}
arguments:
  - "--output"
  - "{{output_dir}}"
outputs:
  artifact_type: checkpoint
  artifact_path: model.ckpt
  metrics_file: metrics.json
  history_file: history.json
""",
        encoding="utf-8",
    )

    payload = services.run_trainer_manifest_job(
        manifest,
        dataset_root="D:/datasets/custom_drive",
        output_dir=str(tmp_path / "run"),
        parameters={},
        adapter="manifest_dataset",
        sequence_id="clip_001",
    )

    record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert payload["metrics"] == {"loss": 0.25, "accuracy": 0.75}
    assert record["artifact_path"] == str((tmp_path / "run" / "model.ckpt").resolve())
    assert record["history"] == {"loss": [0.9, 0.5, 0.25]}
    assert services.training_metric_history(record)["loss"] == [0.9, 0.5, 0.25]


def test_run_trainer_manifest_job_builds_history_from_jsonl_events(tmp_path) -> None:
    trainer_script = tmp_path / "train_events.py"
    trainer_script.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
events = [
    {"step": 1, "loss": 0.8, "val_loss": 0.9},
    {"step": 2, "loss": 0.35, "val_loss": 0.5},
]
(out / "events.jsonl").write_text("\\n".join(json.dumps(row) for row in events), encoding="utf-8")
print("finished")
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "trainer.yaml"
    manifest.write_text(
        f"""
trainer_id: event_trainer
display_name: Event Trainer
runtime: python
entrypoint: {trainer_script}
arguments:
  - "--output"
  - "{{output_dir}}"
outputs:
  artifact_type: checkpoint
  events_file: events.jsonl
""",
        encoding="utf-8",
    )

    payload = services.run_trainer_manifest_job(
        manifest,
        dataset_root="D:/datasets/custom_drive",
        output_dir=str(tmp_path / "run_events"),
        parameters={},
    )

    record = json.loads(Path(payload["training_run_path"]).read_text(encoding="utf-8"))
    assert record["history"]["loss"] == [0.8, 0.35]
    assert record["history"]["val_loss"] == [0.9, 0.5]
    assert record["metrics"]["loss"] == 0.35
    assert record["metrics"]["val_loss"] == 0.5


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


def test_demo_config_entries_use_standard_johnson_valley_task() -> None:
    rows = services.demo_config_entries()

    assert rows[0]["id"] == services.DEFAULT_DEMO_CONFIG_ID
    assert rows[0]["task_path"].replace("\\", "/").endswith("configs/tasks/beamng_johnson_valley_nav_001.yaml")
    assert rows[0]["world_model_config_id"] == services.DEFAULT_WORLD_MODEL_CONFIG_ID
    assert rows[0]["planner"] == "navigation_mpc"


def test_demo_acceptance_runs_multiple_trials_and_summarizes_metrics(tmp_path, monkeypatch) -> None:
    episode_a = _write_trace_episode(
        tmp_path / "episode_a",
        [(1329.0, -109.0, 2.0, False), (1280.0, -145.0, 5.0, True), (1246.0, -189.0, 1.5, False)],
    )
    episode_b = _write_trace_episode(
        tmp_path / "episode_b",
        [(1329.0, -109.0, 1.0, False), (1290.0, -140.0, 2.5, False), (1247.0, -188.5, 1.0, False)],
    )
    episodes = [episode_a, episode_b]
    captured: list[services.RegionNavigationClosedLoopRequest] = []

    def fake_run_region_navigation(request: services.RegionNavigationClosedLoopRequest) -> dict[str, object]:
        captured.append(request)
        episode = episodes[len(captured) - 1]
        task = load_navigation_region_task(request.task_path)
        evaluation = {
            "episode_path": str(episode),
            "metrics": {
                "collision_count": 0,
                "drive_mode": "manual",
                "steps": 3,
                "agent_diagnostics": {"stuck_recovery": False},
            },
        }
        return {"status": "completed", "evaluation": evaluation, "acceptance": services._navigation_acceptance(evaluation, task)}

    monkeypatch.setattr(services, "run_region_navigation_closed_loop", fake_run_region_navigation)

    report = services.run_demo_acceptance(services.DemoAcceptanceRequest(runs=2, max_steps=30, step_delay_sec=0.0, post_run_hold_sec=0.0))

    assert report["status"] == "accepted"
    assert report["run_count"] == 2
    assert report["all_goal_success"] is True
    assert report["summary"]["collision_count"] == 0
    assert report["summary"]["recovery_triggered"] is True
    assert report["runs"][0]["goal_reached"] is True
    assert report["runs"][0]["trajectory_length_m"] > 0.0
    assert report["runs"][0]["average_speed"] > 0.0
    assert captured[0].task_path.endswith("beamng_johnson_valley_nav_001.yaml")


def _write_trace_episode(path: Path, points: list[tuple[float, float, float, bool]]) -> Path:
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": path.name, "backend": "beamng", "agent": "model_mpc"})
    for index, (x, y, speed, recovery) in enumerate(points):
        recorder.record_step(
            observation=Observation(
                timestamp=float(index),
                vehicle_state=VehicleState(x=x, y=y, z=115.0, speed=speed),
                goal=(1245.995, -189.268),
            ),
            action=Action(throttle=0.2, steer=0.1),
            reward=0.0,
            done=False,
            info={"agent_diagnostics": {"stuck_recovery": recovery}},
        )
    recorder.end_episode({"steps": len(points), "collision_count": 0, "drive_mode": "manual"})
    return recorder.save(path)


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
