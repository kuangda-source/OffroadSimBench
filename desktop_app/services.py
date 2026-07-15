"""Qt-independent service helpers used by the desktop GUI."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import uuid
from html import escape
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.agents import default_agent_registry
from offroad_sim.algorithms import DataPrepRequest, TrainRequest, default_algorithm_registry
from offroad_sim.backends import BeamNGBackend, BeamNGConnectionConfig, default_backend_registry
from offroad_sim.core import Action, VehicleState
from offroad_sim.datasets import (
    DatasetAnalysisOptions,
    analyze_dataset_sequences,
    build_dataset_split,
    create_mock_orfd_dataset,
    default_dataset_registry,
    validate_dataset_split_payload,
)
from offroad_sim.datasets import DatasetFrame, DatasetSequence
from offroad_sim.evaluation import run_episode
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT
from offroad_sim.planning import default_planner_registry
from offroad_sim.tasks import NavigationRegionTask, load_navigation_region_task
from offroad_sim.training import (
    TRAINER_SCHEMA_VERSION,
    TRAINING_JOB_FILENAME,
    ProcessTrainingJob,
    TrainingJobQueue,
    build_trainer_command,
    normalize_trainer_manifest,
    resolve_trainer_environment,
    resolve_trainer_working_directory,
    validate_trainer_parameters,
)
from offroad_sim.utils.yaml_io import load_yaml_file
from offroad_sim.vehicles import load_vehicle_config
from offroad_sim.world_models import MLPDynamicsWorldModel, TinyLearnedWorldModel, default_world_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
DEFAULT_NAVIGATION_TASK_PATH = CONFIG_ROOT / "tasks" / "beamng_johnson_valley_nav_test.yaml"
DEFAULT_LEWM_CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "region_navigation"
    / "johnson_valley_nav_test_train_v2_validated"
    / "model"
    / "lewm_cost_object.ckpt"
)
DEFAULT_ROUTE_FREE_MODEL_PATH = (
    ROOT
    / "outputs"
    / "region_world_model_compare"
    / "johnson_valley_strict_collect_tiny_vs_mlp_1200_20260704"
    / "mlp_dynamics"
    / "training"
    / "model"
)
DEFAULT_ROUTE_FREE_VALIDATION_SOURCE = (
    ROOT
    / "outputs"
    / "region_world_model_eval"
    / "p2_mlp_support_subgoals_strict_1200_20260704"
    / "region_world_model_evaluation_summary.json"
)
WORLD_MODEL_CONFIGS_PATH = CONFIG_ROOT / "world_model_configs.json"
DEFAULT_WORLD_MODEL_CONFIG_ID = "johnson_valley_mlp_model_support_20260704"
DEFAULT_LEWM_WORLD_MODEL_CONFIG_ID = "johnson_valley_lewm_validated"
DEFAULT_DEMO_CONFIG_ID = "johnson_valley_standard_demo"
TRAINING_CONFIGS_PATH = CONFIG_ROOT / "training_configs.json"
SMOKE_TRAINING_DATASET_ROOT = ROOT / "outputs" / "training_studio_smoke" / "datasets" / "mock_orfd"
SMOKE_TINY_MODEL_OUTPUT_DIR = ROOT / "outputs" / "training_studio_smoke" / "models" / "tiny_world_model"
SMOKE_TRAINING_SEQUENCE_ID = "training/seq_0001"
TRAINING_RUN_FILENAME = "training_run.json"
INFERENCE_RUN_FILENAME = "inference_run.json"
DATASET_MANIFEST_FILENAMES = ("dataset_manifest.yaml", "dataset_manifest.yml")
DATASET_MANIFEST_DIRS = (CONFIG_ROOT / "datasets",)
TRAINER_MANIFEST_FILENAMES = ("trainer.yaml", "trainer.yml")
INFERENCE_MANIFEST_FILENAMES = ("inference.yaml", "inference.yml")
TRAINER_MANIFEST_DIRS = (CONFIG_ROOT / "trainers", ROOT / "trainers")
NAN_TEXT = "NaN"
REGION_TRAINING_COLLECTION_FILENAME = "region_training_collection.json"
LIGHTWEIGHT_REGION_WORLD_MODELS = ("tiny_learned", "mlp_dynamics")
UNFINISHED_TEXT = "未完成"


@dataclass(slots=True)
class RunRequest:
    backend: str = "gym_heightmap"
    scenario: str = "forest_trail_001"
    agent: str = "rule_based"
    seed: int = 7
    max_steps: int = 120
    record: bool = True
    record_arrays: bool = False
    world_model_type: str = "simple_kinematic"
    world_model_path: str = ""
    planner: str = ""
    planner_horizon: int = 10
    planner_samples: int = 128
    planner_iterations: int = 4
    dataset_root: str = ""
    sequence_id: str = ""
    adapter: str = ""
    load_assets: bool = False


@dataclass(slots=True)
class PipelineRequest:
    dataset_root: str
    adapter: str = "orfd"
    sequence_id: str = ""
    hdf5_path: str = ""
    model_dir: str = ""
    image_size: int = 64
    planner_horizon: int = 4
    planner_samples: int = 16
    planner_iterations: int = 2
    max_steps: int = 3
    seed: int = 7
    run_beamng: bool = True
    beamng_scenario: str = "beamng_orfd_eval"


@dataclass(slots=True)
class VisibleBeamNGDemoRequest:
    dataset_root: str = ""
    adapter: str = "orfd"
    sequence_id: str = ""
    world_model_type: str = "le_wm"
    world_model_path: str = ""
    planner: str = "navigation_mpc"
    scenario: str = "beamng_visible_autodrive"
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    max_steps: int = 600
    seed: int = 7
    record: bool = True
    pre_run_hold_sec: float = 8.0
    step_delay_sec: float = 0.05
    post_run_hold_sec: float = 0.0
    close_beamng: bool = False
    beamng_gfx: str = "vk"


@dataclass(slots=True)
class BeamNGMapLeWMClosedLoopRequest:
    algorithm: str = "local_lewm_cost"
    scenario: str = "beamng_visible_autodrive"
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    collect_steps: int = 160
    eval_steps: int = 120
    seed: int = 7
    planner: str = "navigation_mpc"
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class RegionNavigationClosedLoopRequest:
    task_path: str
    algorithm: str = "local_lewm_cost"
    algorithm_model_path: str = ""
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    collect_steps: int = 160
    eval_steps: int = 120
    seed: int = 7
    planner: str = "navigation_mpc"
    planner_horizon: int = 4
    planner_samples: int = 16
    planner_iterations: int = 2
    evaluation_agent: str = "model_mpc"
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class RegionSelfSupervisedWorldModelRequest:
    task_path: str
    world_model_type: str = "tiny_learned"
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    collect_steps: int = 240
    collect_rollouts: int = 1
    min_collection_goal_progress_ratio: float = 0.0
    collection_goal_bias_interval: int = 1
    collection_goal_corridor_interval: int = 1
    collection_goal_corridor_lateral_m: float = 2.0
    collection_coverage_grid_size: int = 0
    collection_coverage_target_interval: int = 0
    collection_max_target_steps: int = 80
    collection_strategy: str = "region_explorer"
    collection_route_target_interval: int = 0
    collection_route_lateral_m: float = 0.0
    collection_multi_start: bool = False
    collection_multi_start_lateral_m: float = 0.0
    min_route_coverage_ratio: float = 0.0
    min_goal_zone_coverage: float = 0.0
    max_collection_min_goal_distance_m: float = 0.0
    min_unique_region_cells: int = 0
    eval_steps: int = 1200
    seed: int = 7
    planner: str = "navigation_mpc"
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    planner_goal_weight: float | None = None
    planner_progress_weight: float | None = None
    planner_risk_weight: float | None = None
    planner_heading_weight: float | None = None
    evaluation_agent: str = "world_model_direct"
    evaluation_route_mode: str = "route_free"
    use_experience_corridor: bool = True
    experience_route_min_spacing_m: float = 4.0
    experience_route_max_points: int = 120
    evaluation_allow_reverse_recovery: bool = False
    evaluation_reverse_recovery_after_steps: int = 96
    evaluation_local_subgoal_distance_m: float = 12.0
    evaluation_use_model_support_subgoals: bool = False
    evaluation_use_model_support_field_subgoals: bool = False
    evaluation_use_model_support_graph_subgoals: bool = False
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0
    register_world_model_config: bool = False
    world_model_config_path: str = ""


@dataclass(slots=True)
class RegionTrainingDataCollectionRequest:
    task_path: str
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    collect_steps: int = 1000
    collect_rollouts: int = 3
    seed: int = 7
    min_collection_goal_progress_ratio: float = 0.0
    collection_goal_bias_interval: int = 1
    collection_goal_corridor_interval: int = 1
    collection_goal_corridor_lateral_m: float = 2.0
    collection_coverage_grid_size: int = 4
    collection_coverage_target_interval: int = 1
    collection_max_target_steps: int = 40
    collection_strategy: str = "region_explorer"
    collection_route_target_interval: int = 0
    collection_route_lateral_m: float = 0.0
    collection_multi_start: bool = False
    collection_multi_start_lateral_m: float = 0.0
    min_route_coverage_ratio: float = 0.0
    min_goal_zone_coverage: float = 0.0
    max_collection_min_goal_distance_m: float = 0.0
    min_unique_region_cells: int = 0
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class RegionWorldModelTrainingRequest:
    collection_manifest_path: str
    world_model_type: str = "tiny_learned"
    output_dir: str = ""
    register_world_model_config: bool = True
    world_model_config_path: str = ""


@dataclass(slots=True)
class RegionWorldModelEvaluationRequest:
    task_path: str
    world_model_type: str = "tiny_learned"
    world_model_path: str = ""
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    eval_steps: int = 1000
    seed: int = 7
    planner: str = "navigation_mpc"
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    planner_goal_weight: float | None = None
    planner_progress_weight: float | None = None
    planner_risk_weight: float | None = None
    planner_heading_weight: float | None = None
    evaluation_agent: str = "world_model_direct"
    evaluation_allow_reverse_recovery: bool = False
    evaluation_reverse_recovery_after_steps: int = 96
    evaluation_local_subgoal_distance_m: float = 12.0
    evaluation_use_model_support_subgoals: bool = False
    evaluation_use_model_support_field_subgoals: bool = False
    evaluation_use_model_support_graph_subgoals: bool = False
    use_experience_corridor: bool = False
    experience_route_min_spacing_m: float = 4.0
    experience_route_max_points: int = 120
    include_route_guided_baseline: bool = False
    write_trajectory_plot: bool = True
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class RegionWorldModelComparisonRequest:
    collection_manifest_path: str
    world_model_types: list[str] = field(default_factory=lambda: ["tiny_learned", "mlp_dynamics"])
    task_path: str = ""
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    eval_steps: int = 1000
    seed: int = 7
    planner: str = "navigation_mpc"
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    planner_goal_weight: float | None = None
    planner_progress_weight: float | None = None
    planner_risk_weight: float | None = None
    planner_heading_weight: float | None = None
    evaluation_agent: str = "world_model_direct"
    evaluation_allow_reverse_recovery: bool = False
    evaluation_reverse_recovery_after_steps: int = 96
    evaluation_local_subgoal_distance_m: float = 12.0
    evaluation_use_model_support_subgoals: bool = False
    evaluation_use_model_support_field_subgoals: bool = False
    evaluation_use_model_support_graph_subgoals: bool = False
    use_experience_corridor: bool = False
    experience_route_min_spacing_m: float = 4.0
    experience_route_max_points: int = 120
    include_route_guided_baseline: bool = True
    write_trajectory_plot: bool = True
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class DemoAcceptanceRequest:
    demo_config_id: str = DEFAULT_DEMO_CONFIG_ID
    runs: int = 1
    max_steps: int = 1000
    seed: int = 7
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class ManualNavigationTaskRequest:
    output_path: str
    task_id: str = "manual_region_nav"
    map_id: str = "manual_region"
    level: str = "gridmap_v2"
    region_polygon: list[tuple[float, float]] = field(default_factory=list)
    start_pos: tuple[float, float, float] = (0.0, 0.0, 100.6)
    start_yaw: float = 0.0
    goal_pos: tuple[float, float] = (10.0, 0.0)
    goal_radius: float = 8.0
    expert_route: list[tuple[float, float]] = field(default_factory=list)
    max_steps: int = 300
    max_collision_count: int = 0
    out_of_region_weight: float = 250.0
    boundary_weight: float = 8.0
    boundary_margin_m: float = 3.0
    vehicle_model: str = "pickup"
    collection_drive_mode: str = "ai_line"
    evaluation_drive_mode: str = "manual"
    evaluation_route_mode: str = "expert"
    ai_line_speed: float = 10.0
    steps_per_action: int = 18
    camera_mode: str = "follow"


def config_entries(kind: str, id_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((CONFIG_ROOT / kind).glob("*.yaml")):
        data = load_yaml_file(path)
        rows.append({"id": str(data.get(id_field, path.stem)), "path": str(path), "summary": data})
    return rows


def catalog_snapshot() -> dict[str, list[dict[str, Any]]]:
    return {
        "scenarios": config_entries("scenarios", "scenario_id"),
        "vehicles": config_entries("vehicles", "vehicle_id"),
        "agents": [
            {"name": spec.name, "available": True, "message": "available", "description": spec.description}
            for spec in (default_agent_registry().get(name) for name in default_agent_registry().names())
        ],
        "backends": _registry_rows(default_backend_registry()),
        "world_models": _registry_rows(default_world_model_registry()),
        "planners": _registry_rows(default_planner_registry()),
        "algorithms": _algorithm_rows(),
        "navigation_tasks": navigation_task_entries(),
        "demo_configs": demo_config_entries(),
        "model_checkpoints": model_checkpoint_entries(),
        "world_model_configs": world_model_config_entries(),
        "demo_ready_world_model_configs": demo_ready_world_model_config_entries(),
        "dataset_manifests": dataset_manifest_entries(),
        "training_configs": training_config_entries(),
        "training_presets": training_preset_entries(),
        "training_runs": training_run_entries(),
        "training_jobs": training_job_entries(),
        "training_artifacts": training_artifact_entries(),
        "inference_runs": inference_run_entries(),
        "episodes": episode_summaries(),
    }


def dataset_manifest_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Discover installed manifest datasets."""

    paths: list[Path] = []
    if root is None:
        for directory in DATASET_MANIFEST_DIRS:
            paths.extend(_dataset_manifest_paths(directory))
    else:
        paths.extend(_dataset_manifest_paths(Path(root)))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            rows.append(load_dataset_manifest(resolved))
        except Exception:
            continue
    return sorted(rows, key=lambda row: str(row.get("label") or row.get("id") or ""))


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        candidates = _dataset_manifest_paths(manifest_path)
        if not candidates:
            raise FileNotFoundError(f"Dataset manifest not found in {manifest_path}")
        manifest_path = candidates[0].resolve()
    data = load_yaml_file(manifest_path)
    adapter = str(data.get("adapter") or "manifest_dataset").strip()
    if adapter != "manifest_dataset":
        raise ValueError(f"Dataset manifest adapter must be manifest_dataset: {manifest_path}")
    dataset_id = _safe_name(str(data.get("dataset_id") or data.get("id") or manifest_path.parent.name))
    if not dataset_id:
        raise ValueError(f"Dataset manifest has no dataset_id: {manifest_path}")
    sequences = data.get("sequences") if isinstance(data.get("sequences"), list) else []
    sequence_ids = [
        str(row.get("id") or row.get("sequence_id") or "")
        for row in sequences
        if isinstance(row, dict) and str(row.get("id") or row.get("sequence_id") or "")
    ]
    return {
        "id": dataset_id,
        "label": str(data.get("display_name") or data.get("label") or dataset_id),
        "adapter": adapter,
        "dataset_root": str(manifest_path.parent.resolve()),
        "dataset_type": str(data.get("dataset_type") or "manifest_dataset"),
        "manifest_path": str(manifest_path),
        "sequences": sequence_ids,
        "imported_from": str(data.get("imported_from") or ""),
    }


def import_dataset_manifest(source_path: str | Path, destination_root: str | Path | None = None) -> dict[str, Any]:
    """Install an external dataset manifest into the project dataset catalog."""

    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {source}")
    source_row = load_dataset_manifest(source)
    data = load_yaml_file(source)
    data["adapter"] = "manifest_dataset"
    data["dataset_id"] = source_row["id"]
    data["imported_from"] = str(source)
    sequences = data.get("sequences") if isinstance(data.get("sequences"), list) else []
    rewritten_sequences: list[Any] = []
    for raw_sequence in sequences:
        if not isinstance(raw_sequence, dict):
            rewritten_sequences.append(raw_sequence)
            continue
        row = dict(raw_sequence)
        root_value = str(row.get("root") or ".")
        root_path = Path(root_value)
        if not root_path.is_absolute():
            row["root"] = str((source.parent / root_path).resolve())
        rewritten_sequences.append(row)
    data["sequences"] = rewritten_sequences
    destination = Path(destination_root or CONFIG_ROOT / "datasets") / source_row["id"]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "dataset_manifest.yaml"
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to import dataset manifests.") from exc
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_dataset_manifest(target)


def save_dataset_manifest(
    *,
    dataset_id: str,
    display_name: str = "",
    dataset_root: str,
    sequences: list[dict[str, Any]],
    destination_root: str | Path | None = None,
    dataset_type: str = "manifest_dataset",
) -> dict[str, Any]:
    """Create a generic manifest_dataset entry from a user-described directory.

    This is the GUI boundary for "bring any driving dataset": the user still
    declares sequences and asset patterns, while the rest of the platform only
    sees the registered manifest adapter.
    """

    source_root = Path(dataset_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")
    manifest_id = _safe_name(dataset_id).lower()
    if not manifest_id:
        raise ValueError("Dataset id is required.")
    if not sequences:
        raise ValueError("At least one dataset sequence is required.")

    normalized_sequences: list[dict[str, Any]] = []
    for index, raw_sequence in enumerate(sequences, start=1):
        sequence_id = str(raw_sequence.get("id") or raw_sequence.get("sequence_id") or f"sequence_{index:04d}").strip()
        sequence_root = Path(str(raw_sequence.get("root") or "."))
        if not sequence_root.is_absolute():
            sequence_root = (source_root / sequence_root).resolve()
        row: dict[str, Any] = {
            "id": sequence_id,
            "root": str(sequence_root.resolve()),
        }
        for key in ("pose_csv", "timestamp_csv", "actions_csv", "metadata"):
            value = raw_sequence.get(key)
            if value:
                row[key] = value
        assets = raw_sequence.get("assets") if isinstance(raw_sequence.get("assets"), dict) else {}
        if assets:
            row["assets"] = {str(name): str(pattern) for name, pattern in assets.items() if str(name)}
        normalized_sequences.append(row)

    data = {
        "adapter": "manifest_dataset",
        "dataset_id": manifest_id,
        "display_name": display_name or dataset_id,
        "dataset_type": dataset_type or "manifest_dataset",
        "source_root": str(source_root),
        "sequences": normalized_sequences,
    }
    destination = Path(destination_root or DATASET_MANIFEST_DIRS[0]) / manifest_id
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "dataset_manifest.yaml"
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to save dataset manifests.") from exc
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_dataset_manifest(target)


def suggest_dataset_manifest_sequences(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Suggest manifest_dataset sequence rows for common driving-dataset layouts."""

    root = Path(dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    candidates: list[Path] = []
    if _sequence_pose_csv(root) is not None:
        candidates.append(root)
    for child in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
        if _sequence_pose_csv(child) is not None:
            candidates.append(child)
    if not candidates:
        for child in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
            for grandchild in sorted((path for path in child.iterdir() if path.is_dir()), key=lambda path: path.name.lower()):
                if _sequence_pose_csv(grandchild) is not None:
                    candidates.append(grandchild)

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for sequence_dir in candidates:
        resolved = sequence_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        pose_csv = _sequence_pose_csv(resolved)
        if pose_csv is None:
            continue
        relative_root = _relative_path_for_manifest(resolved, root)
        row: dict[str, Any] = {
            "id": _safe_name(resolved.name or root.name) or f"sequence_{len(rows) + 1:04d}",
            "root": relative_root,
            "pose_csv": pose_csv.name,
        }
        actions_csv = _first_existing_file(resolved, ("actions.csv", "controls.csv", "commands.csv"))
        if actions_csv is not None:
            row["actions_csv"] = actions_csv.name
        assets = _suggest_sequence_assets(resolved)
        if assets:
            row["assets"] = assets
        rows.append(row)

    if not rows:
        raise ValueError("No manifest-compatible sequences found. Expected poses.csv plus common asset folders such as images/, rgb/, depth/, masks/, or lidar/.")
    return rows


def _sequence_pose_csv(sequence_root: Path) -> Path | None:
    return _first_existing_file(
        sequence_root,
        (
            "poses.csv",
            "pose.csv",
            "odometry.csv",
            "trajectory.csv",
            "state.csv",
            "states.csv",
        ),
    )


def _first_existing_file(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _suggest_sequence_assets(sequence_root: Path) -> dict[str, str]:
    candidates: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("front_rgb", ("front_rgb", "images", "image", "rgb", "camera", "camera_front"), (".png", ".jpg", ".jpeg", ".npy")),
        ("depth", ("depth", "depths", "dense_depth", "dense-depth"), (".npy", ".png", ".exr")),
        ("label", ("masks", "mask", "labels", "label", "segmentation", "seg"), (".png", ".npy")),
        ("lidar", ("lidar", "lidar_points", "pointcloud", "point_cloud", "velodyne"), (".bin", ".npy", ".pcd")),
        ("local_bev", ("local_bev", "bev"), (".npy", ".png")),
        ("terrain_map", ("terrain_map", "terrain"), (".npy", ".png")),
    )
    assets: dict[str, str] = {}
    for asset_name, directory_names, extensions in candidates:
        pattern = _first_asset_pattern(sequence_root, directory_names, extensions)
        if pattern:
            assets[asset_name] = pattern
    return assets


def _first_asset_pattern(sequence_root: Path, directory_names: tuple[str, ...], extensions: tuple[str, ...]) -> str:
    for directory_name in directory_names:
        directory = sequence_root / directory_name
        if not directory.is_dir():
            continue
        for extension in extensions:
            if any(path.is_file() for path in directory.glob(f"*{extension}")):
                return f"{directory_name}/*{extension}"
    return ""


def _relative_path_for_manifest(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return str(path.resolve())
    text = relative.as_posix()
    return "." if text == "." else text


def training_preset_entries(trainer_root: str | Path | None = None) -> list[dict[str, Any]]:
    rows = [
        {
            "id": "stablewm_hdf5",
            "label": "数据集导出为 StableWM HDF5",
            "kind": "export",
            "available": True,
            "status": "available",
            "description": "将图像序列和动作导出为 StableWM/LE-WM 训练使用的 HDF5 格式。",
        },
        {
            "id": "lewm_cost_model",
            "label": "训练 LE-WM 代价模型",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "从 StableWM HDF5 训练本地轻量 LE-WM 兼容代价 checkpoint。",
        },
        {
            "id": "tiny_world_model",
            "label": "训练 Tiny 世界模型",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "训练内置轻量动力学模型，用于快速验证数据和训练流程。",
        },
        {
            "id": "beamng_region_training_data",
            "label": "采集 BeamNG 区域训练数据",
            "kind": "collection",
            "available": True,
            "status": "available",
            "description": "采集可复用的 BeamNG 区域 episode，供仿真世界模型训练。",
        },
        {
            "id": "region_world_model_training",
            "label": "训练 BeamNG 区域世界模型",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "使用已采集的 BeamNG 区域 episode 训练内置轻量世界模型。",
        },
        {
            "id": "lewm_full_self_supervised",
            "label": "LE-WM 完整自监督训练",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "完整视觉潜变量 LE-WM 训练栈的预留适配器。",
        },
        {
            "id": "tdmpc2_adapter",
            "label": "TD-MPC2 适配器",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "TD-MPC2 模型控制实验的预留适配器。",
        },
        {
            "id": "dreamerv3_adapter",
            "label": "DreamerV3 适配器",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "DreamerV3 世界模型训练的预留适配器。",
        },
    ]
    manifest_ids = {row["id"] for row in rows}
    for row in trainer_manifest_entries(trainer_root):
        if row["id"] in manifest_ids:
            continue
        rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "kind": "training",
                "available": True,
                "status": "available",
                "description": row.get("description", "Run an external trainer described by trainer.yaml."),
                "manifest_path": row["manifest_path"],
                "schema_version": row.get("schema_version", TRAINER_SCHEMA_VERSION),
                "launch": dict(row.get("launch", {})),
                "parameters": dict(row.get("parameters", {})),
                "input": dict(row.get("input", {})),
                "outputs": dict(row.get("outputs", {})),
                "inference": dict(row.get("inference", {})),
            }
        )
    return rows


def trainer_manifest_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Discover external trainer manifests.

    A trainer manifest lets users plug in an arbitrary local algorithm script
    without changing the GUI or services code.
    """

    paths: list[Path] = []
    if root is None:
        for directory in TRAINER_MANIFEST_DIRS:
            paths.extend(_trainer_manifest_paths(directory))
    else:
        paths.extend(_trainer_manifest_paths(Path(root)))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            rows.append(load_trainer_manifest(resolved))
        except Exception:
            continue
    return sorted(rows, key=lambda row: str(row.get("label") or row.get("id") or ""))


def load_trainer_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    data = load_yaml_file(manifest_path)
    inference_manifest_path = _attach_external_inference_manifest(data, manifest_path)
    normalized = normalize_trainer_manifest(data, manifest_path=manifest_path)
    trainer_id = _safe_name(str(normalized["trainer_id"]))
    launch = dict(normalized["launch"])
    return {
        "id": trainer_id,
        "label": str(normalized["display_name"]),
        "trainer_id": trainer_id,
        "schema_version": normalized["schema_version"],
        "runtime": "python" if str(launch["kind"]).startswith("python_") else "executable",
        "entrypoint": str(launch.get("entrypoint") or ""),
        "module": str(launch.get("module") or ""),
        "launch": launch,
        "description": str(normalized["description"]),
        "arguments": list(normalized["arguments"]),
        "parameters": dict(normalized["parameters"]),
        "input": dict(normalized["input"]),
        "outputs": dict(normalized["outputs"]),
        "inference": dict(normalized["inference"]),
        "inference_manifest_path": str(inference_manifest_path) if inference_manifest_path else "",
        "manifest_path": str(manifest_path),
        "manifest_dir": str(manifest_path.parent),
    }


def _snapshot_training_inputs(
    output_dir: Path,
    manifest_path: str | Path,
    split_path: str = "",
) -> dict[str, str]:
    """Persist portable, hash-addressed copies of inputs used by one run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(manifest_path).resolve()
    manifest_data = load_yaml_file(source_manifest)
    external_inference = _attach_external_inference_manifest(manifest_data, source_manifest)
    manifest_data = normalize_trainer_manifest(manifest_data, manifest_path=source_manifest)
    _rebase_manifest_launch_paths(manifest_data, source_manifest.parent)
    if external_inference and isinstance(manifest_data.get("inference"), dict):
        inference = manifest_data["inference"]
        launch = inference.get("launch") if isinstance(inference.get("launch"), dict) else {}
        _rebase_launch_paths(launch, external_inference.parent)
        inference["launch"] = launch
        manifest_data.pop("inference_manifest", None)
    manifest_snapshot = output_dir / "trainer_manifest.snapshot.yaml"
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to snapshot trainer manifests.") from exc
    manifest_snapshot.write_text(
        yaml.safe_dump(manifest_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    provenance = {
        "trainer_manifest_source_path": str(source_manifest),
        "trainer_manifest_snapshot_path": str(manifest_snapshot.resolve()),
        "trainer_manifest_sha256": _file_sha256(manifest_snapshot),
    }
    for prefix, launch in (
        ("trainer", manifest_data.get("launch")),
        (
            "inference",
            manifest_data.get("inference", {}).get("launch")
            if isinstance(manifest_data.get("inference"), dict)
            else None,
        ),
    ):
        if not isinstance(launch, dict):
            continue
        entrypoint_value = str(launch.get("entrypoint") or "").strip()
        if not entrypoint_value:
            continue
        entrypoint = Path(entrypoint_value)
        if entrypoint.is_file():
            provenance[f"{prefix}_entrypoint_path"] = str(entrypoint.resolve())
            provenance[f"{prefix}_entrypoint_sha256"] = _file_sha256(entrypoint)
    if split_path:
        source_split = Path(split_path).resolve()
        payload = json.loads(source_split.read_text(encoding="utf-8"))
        validate_dataset_split_payload(payload)
        split_snapshot = output_dir / "dataset_split.snapshot.json"
        split_snapshot.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        provenance.update(
            split_source_path=str(source_split),
            split_snapshot_path=str(split_snapshot.resolve()),
            split_sha256=_file_sha256(split_snapshot),
        )
    return provenance


def _verify_recorded_training_snapshot(path: str | Path, kind: str) -> None:
    """Reject edited run snapshots and trainer code before a historical rerun."""

    candidate = Path(path).resolve()
    record_path = candidate.parent / TRAINING_RUN_FILENAME
    if not record_path.is_file():
        return
    try:
        record = _read_json(record_path)
    except (OSError, json.JSONDecodeError):
        return
    if kind == "manifest":
        path_key = "trainer_manifest_snapshot_path"
        hash_key = "trainer_manifest_sha256"
    elif kind == "split":
        path_key = "split_snapshot_path"
        hash_key = "split_sha256"
    else:
        raise ValueError(f"Unknown training snapshot kind: {kind}")
    recorded_path = str(record.get(path_key) or "").strip()
    expected_hash = str(record.get(hash_key) or "").strip()
    if not recorded_path or Path(recorded_path).resolve() != candidate:
        return
    if not candidate.is_file():
        raise FileNotFoundError(f"Recorded {kind} snapshot is missing: {candidate}")
    actual_hash = _file_sha256(candidate)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"Recorded {kind} snapshot failed SHA-256 verification: {candidate}"
        )
    if kind == "manifest":
        for prefix in ("trainer", "inference"):
            entrypoint_value = str(record.get(f"{prefix}_entrypoint_path") or "").strip()
            entrypoint_hash = str(record.get(f"{prefix}_entrypoint_sha256") or "").strip()
            if not entrypoint_value or not entrypoint_hash:
                continue
            entrypoint = Path(entrypoint_value).resolve()
            if not entrypoint.is_file() or _file_sha256(entrypoint) != entrypoint_hash:
                raise ValueError(
                    f"Recorded {prefix} entrypoint failed SHA-256 verification: {entrypoint}"
                )


def _rebase_manifest_launch_paths(data: dict[str, Any], base_dir: Path) -> None:
    launch = data.get("launch") if isinstance(data.get("launch"), dict) else {}
    if launch:
        _rebase_launch_paths(launch, base_dir)
        data["launch"] = launch
    elif data.get("entrypoint"):
        entrypoint = Path(str(data["entrypoint"]))
        if not entrypoint.is_absolute():
            data["entrypoint"] = str((base_dir / entrypoint).resolve())
    inference = data.get("inference") if isinstance(data.get("inference"), dict) else {}
    inference_launch = inference.get("launch") if isinstance(inference.get("launch"), dict) else {}
    if inference_launch:
        _rebase_launch_paths(inference_launch, base_dir)
        inference["launch"] = inference_launch


def _rebase_launch_paths(launch: dict[str, Any], base_dir: Path) -> None:
    entrypoint_value = str(launch.get("entrypoint") or "").strip()
    kind = str(launch.get("kind") or "python_script")
    entrypoint_is_path = any(separator in entrypoint_value for separator in ("/", "\\")) or entrypoint_value.startswith(".")
    if entrypoint_value and (kind == "python_script" or entrypoint_is_path):
        entrypoint = Path(entrypoint_value)
        if not entrypoint.is_absolute():
            launch["entrypoint"] = str((base_dir / entrypoint).resolve())
    working_value = str(launch.get("working_directory") or "").strip()
    if working_value:
        working_directory = Path(working_value)
        if not working_directory.is_absolute():
            launch["working_directory"] = str((base_dir / working_directory).resolve())
    else:
        launch["working_directory"] = str(base_dir.resolve())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_trainer_manifest(source_path: str | Path, destination_root: str | Path | None = None) -> dict[str, Any]:
    """Install an external trainer manifest into the project trainer catalog."""

    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Trainer manifest not found: {source}")
    source_row = load_trainer_manifest(source)
    data = load_yaml_file(source)
    if source_row.get("inference_manifest_path"):
        data.pop("inference_manifest", None)
        data["inference"] = dict(source_row.get("inference") or {})
    launch = data.get("launch") if isinstance(data.get("launch"), dict) else None
    if launch is not None:
        entrypoint_value = str(launch.get("entrypoint") or "").strip()
        entrypoint_is_path = any(separator in entrypoint_value for separator in ("/", "\\")) or entrypoint_value.startswith(".")
        launch_kind = str(launch.get("kind") or "python_script")
        if entrypoint_value and (launch_kind == "python_script" or entrypoint_is_path) and not Path(entrypoint_value).is_absolute():
            launch["entrypoint"] = str((source.parent / entrypoint_value).resolve())
        working_directory = str(launch.get("working_directory") or "").strip()
        if working_directory and not Path(working_directory).is_absolute():
            launch["working_directory"] = str((source.parent / working_directory).resolve())
    else:
        entrypoint = Path(str(data.get("entrypoint") or ""))
        if not entrypoint.is_absolute():
            data["entrypoint"] = str((source.parent / entrypoint).resolve())
    inference = data.get("inference") if isinstance(data.get("inference"), dict) else {}
    inference_launch = inference.get("launch") if isinstance(inference.get("launch"), dict) else {}
    inference_entrypoint = str(inference_launch.get("entrypoint") or "").strip()
    inference_is_path = any(separator in inference_entrypoint for separator in ("/", "\\")) or inference_entrypoint.startswith(".")
    inference_kind = str(inference_launch.get("kind") or "python_script")
    if inference_entrypoint and (inference_kind == "python_script" or inference_is_path) and not Path(inference_entrypoint).is_absolute():
        inference_launch["entrypoint"] = str((source.parent / inference_entrypoint).resolve())
    inference_working_directory = str(inference_launch.get("working_directory") or "").strip()
    if inference_working_directory and not Path(inference_working_directory).is_absolute():
        inference_launch["working_directory"] = str((source.parent / inference_working_directory).resolve())
    data["imported_from"] = str(source)
    destination = Path(destination_root or CONFIG_ROOT / "trainers")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{_safe_name(source_row['id'])}.yaml"
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to import trainer manifests.") from exc
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_trainer_manifest(target)


def _attach_external_inference_manifest(data: dict[str, Any], trainer_path: Path) -> Path | None:
    if isinstance(data.get("inference"), dict) and data["inference"]:
        return None
    declared = str(data.get("inference_manifest") or "").strip()
    if declared:
        candidate = (trainer_path.parent / declared).resolve()
    else:
        candidate = next(
            (
                trainer_path.with_name(name)
                for name in INFERENCE_MANIFEST_FILENAMES
                if trainer_path.with_name(name).is_file()
            ),
            trainer_path.with_name(INFERENCE_MANIFEST_FILENAMES[0]),
        )
    if not candidate.is_file():
        if declared:
            raise FileNotFoundError(f"Inference manifest not found: {candidate}")
        return None
    payload = load_yaml_file(candidate)
    raw = payload.get("inference") if isinstance(payload.get("inference"), dict) else payload
    if not isinstance(raw, dict):
        raise ValueError(f"Inference manifest must contain a mapping: {candidate}")
    inference = {
        str(key): value
        for key, value in raw.items()
        if key not in {"schema_version", "inference_id", "display_name", "description"}
    }
    launch = dict(inference.get("launch") if isinstance(inference.get("launch"), dict) else {})
    kind = str(launch.get("kind") or "python_script")
    entrypoint = str(launch.get("entrypoint") or "").strip()
    entrypoint_is_path = any(separator in entrypoint for separator in ("/", "\\")) or entrypoint.startswith(".")
    if entrypoint and (kind == "python_script" or entrypoint_is_path) and not Path(entrypoint).is_absolute():
        launch["entrypoint"] = str((candidate.parent / entrypoint).resolve())
    working_directory = str(launch.get("working_directory") or "").strip()
    if working_directory and not Path(working_directory).is_absolute():
        launch["working_directory"] = str((candidate.parent / working_directory).resolve())
    inference["launch"] = launch
    data["inference"] = inference
    return candidate


def save_trainer_manifest(
    *,
    trainer_id: str = "",
    label: str = "",
    entrypoint: str,
    runtime: str = "python",
    launch_kind: str = "",
    module: str = "",
    conda_env: str = "",
    working_directory: str = "",
    environment: dict[str, str] | None = None,
    arguments: list[Any] | None = None,
    parameters: dict[str, Any] | None = None,
    input_spec: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    description: str = "",
    destination_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a trainer manifest from a local algorithm entrypoint."""

    kind = str(launch_kind or ("python_script" if runtime == "python" else "executable")).lower()
    entrypoint_path = Path(entrypoint).resolve() if entrypoint else None
    if kind != "python_module" and (entrypoint_path is None or not entrypoint_path.exists()):
        raise FileNotFoundError(f"Trainer entrypoint not found: {entrypoint_path or entrypoint}")
    fallback_name = module.rsplit(".", 1)[-1] if kind == "python_module" else str(entrypoint_path.stem)
    manifest_id = _safe_name(str(trainer_id or label or fallback_name)).lower()
    display_name = str(label or trainer_id or fallback_name)
    data: dict[str, Any] = {
        "schema_version": TRAINER_SCHEMA_VERSION,
        "trainer_id": manifest_id,
        "display_name": display_name,
        "runtime": str(runtime or "python"),
        "entrypoint": str(entrypoint_path) if entrypoint_path is not None else "",
        "launch": {
            "kind": kind,
            "entrypoint": str(entrypoint_path) if entrypoint_path is not None else "",
            "module": module,
            "conda_env": conda_env,
            "working_directory": working_directory,
            "environment": dict(environment or {}),
        },
        "arguments": list(arguments or ["{dataset_root}", "--output", "{output_dir}"]),
        "parameters": dict(parameters or {}),
        "input": dict(input_spec or {}),
        "outputs": dict(outputs or {"artifact_type": "artifact"}),
    }
    if description:
        data["description"] = description
    destination = Path(destination_root or CONFIG_ROOT / "trainers")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{manifest_id}.yaml"
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to save trainer manifests.") from exc
    target.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_trainer_manifest(target)


def run_trainer_manifest_job(
    manifest_path: str | Path,
    *,
    dataset_root: str,
    output_dir: str,
    parameters: dict[str, Any] | None = None,
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
) -> dict[str, Any]:
    _verify_recorded_training_snapshot(manifest_path, "manifest")
    if split_path:
        _verify_recorded_training_snapshot(split_path, "split")
    manifest = load_trainer_manifest(manifest_path)
    target_dir = Path(output_dir or ROOT / "outputs" / "training_runs" / f"{manifest['id']}_{_timestamp()}")
    target_dir.mkdir(parents=True, exist_ok=True)
    provenance = _snapshot_training_inputs(target_dir, manifest_path, split_path)
    execution_split_path = str(provenance.get("split_snapshot_path") or split_path)
    resolved_parameters = _trainer_parameters(manifest.get("parameters", {}), parameters or {})
    command = _trainer_command(
        manifest,
        dataset_root,
        target_dir,
        resolved_parameters,
        adapter,
        sequence_id,
        execution_split_path,
    )
    manifest_dir = Path(str(manifest["manifest_dir"]))
    working_directory = resolve_trainer_working_directory(manifest, manifest_dir)
    environment = resolve_trainer_environment(manifest, manifest_dir)
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path = target_dir / "stdout.log"
    stderr_path = target_dir / "stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        write_training_run_record(
            target_dir,
            preset_id=manifest["id"],
            status="failed",
            dataset_root=dataset_root,
            adapter=adapter,
            sequence_id=sequence_id,
            split_path=split_path,
            artifact_path=str(target_dir.resolve()),
            artifact_type=str(manifest.get("outputs", {}).get("artifact_type") or "artifact"),
            parameters=resolved_parameters,
            summary={
                "command": command,
                "working_directory": str(working_directory),
                "stderr": completed.stderr.strip(),
            },
            logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
            provenance=provenance,
        )
        raise RuntimeError((completed.stderr or completed.stdout or "trainer command failed").strip())

    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    payload = _trainer_result_payload(manifest, target_dir, completed.stdout, command)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    history = payload.get("history") if isinstance(payload.get("history"), dict) else {}
    history_steps = payload.get("history_steps") if isinstance(payload.get("history_steps"), dict) else {}
    artifact_path = str(
        payload.get("checkpoint_path")
        or payload.get("model_path")
        or payload.get("artifact_path")
        or payload.get("output_dir")
        or outputs.get("artifact_path")
        or target_dir
    )
    artifact_path = str(_resolve_trainer_output_path(target_dir, artifact_path))
    artifact_type = str(payload.get("artifact_type") or outputs.get("artifact_type") or "artifact")
    try:
        _require_training_artifact(artifact_path)
    except FileNotFoundError as exc:
        write_training_run_record(
            target_dir,
            preset_id=manifest["id"],
            status="failed",
            dataset_root=dataset_root,
            adapter=adapter,
            sequence_id=sequence_id,
            split_path=split_path,
            artifact_path=artifact_path,
            artifact_type=artifact_type,
            metrics=metrics,
            history=history,
            history_steps=history_steps,
            parameters=resolved_parameters,
            summary={
                "trainer_manifest_path": str(Path(manifest_path).resolve()),
                "command": command,
                "working_directory": str(working_directory),
                "error": str(exc),
            },
            logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
            provenance=provenance,
        )
        raise
    record = write_training_run_record(
        target_dir,
        preset_id=manifest["id"],
        status="completed",
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=sequence_id,
        split_path=split_path,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        metrics=metrics,
        history=history,
        history_steps=history_steps,
        parameters=resolved_parameters,
        summary={
            "trainer_manifest_path": str(Path(manifest_path).resolve()),
            "command": command,
            "working_directory": str(working_directory),
        },
        logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
        provenance=provenance,
    )
    payload["output_dir"] = str(target_dir.resolve())
    payload["metrics"] = metrics
    payload["history"] = history
    payload["history_steps"] = history_steps
    payload["artifact_path"] = artifact_path
    payload["artifact_type"] = artifact_type
    payload["training_run_path"] = record["path"]
    payload["command"] = command
    return payload


def queue_trainer_manifest_job(
    queue: TrainingJobQueue,
    manifest_path: str | Path,
    *,
    dataset_root: str,
    output_dir: str,
    parameters: dict[str, Any] | None = None,
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
) -> ProcessTrainingJob:
    """Queue a trainer manifest and finalize its standard training record."""

    _verify_recorded_training_snapshot(manifest_path, "manifest")
    if split_path:
        _verify_recorded_training_snapshot(split_path, "split")
    manifest = load_trainer_manifest(manifest_path)
    target_dir = Path(output_dir or ROOT / "outputs" / "training_runs" / f"{manifest['id']}_{_timestamp()}")
    target_dir.mkdir(parents=True, exist_ok=True)
    provenance = _snapshot_training_inputs(target_dir, manifest_path, split_path)
    execution_split_path = str(provenance.get("split_snapshot_path") or split_path)
    resolved_parameters = _trainer_parameters(manifest.get("parameters", {}), parameters or {})
    command = _trainer_command(
        manifest,
        dataset_root,
        target_dir,
        resolved_parameters,
        adapter,
        sequence_id,
        execution_split_path,
    )
    manifest_dir = Path(str(manifest["manifest_dir"]))
    working_directory = resolve_trainer_working_directory(manifest, manifest_dir)
    environment = resolve_trainer_environment(manifest, manifest_dir)
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    stdout_path = target_dir / "stdout.log"
    stderr_path = target_dir / "stderr.log"
    write_training_run_record(
        target_dir,
        preset_id=manifest["id"],
        status="queued",
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=sequence_id,
        split_path=split_path,
        artifact_path=str(target_dir.resolve()),
        artifact_type=str(outputs.get("artifact_type") or "artifact"),
        parameters=resolved_parameters,
        summary={
            "trainer_manifest_path": str(Path(manifest_path).resolve()),
            "command": command,
            "working_directory": str(working_directory),
        },
        logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
        provenance=provenance,
    )

    def finalize(job: ProcessTrainingJob) -> dict[str, Any]:
        snapshot = job.snapshot()
        resource_history = snapshot.get("resource_history") if isinstance(snapshot.get("resource_history"), dict) else {}
        normalized_resources = {
            f"resource.{key}": list(values)
            for key, values in resource_history.items()
            if isinstance(values, list) and values
        }
        resource_steps = {
            key: list(resource_history.get("elapsed_sec", range(len(values))))[: len(values)]
            for key, values in normalized_resources.items()
        }
        return_code = snapshot.get("return_code")
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.is_file() else ""
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
        if snapshot.get("status") == "canceling":
            status = "canceled"
        elif return_code != 0:
            status = "failed"
        else:
            status = "completed"

        if status != "completed":
            metric_diagnostics = training_metric_diagnostics(
                {
                    "status": status,
                    "history": normalized_resources,
                    "summary": {
                        "events_path": str(_trainer_output_file(target_dir, outputs.get("events_file"), "events.jsonl"))
                    },
                }
            )
            record = write_training_run_record(
                target_dir,
                preset_id=manifest["id"],
                status=status,
                dataset_root=dataset_root,
                adapter=adapter,
                sequence_id=sequence_id,
                split_path=split_path,
                artifact_path=str(target_dir.resolve()),
                artifact_type=str(outputs.get("artifact_type") or "artifact"),
                history=normalized_resources,
                history_steps=resource_steps,
                parameters=resolved_parameters,
                summary={
                    "trainer_manifest_path": str(Path(manifest_path).resolve()),
                    "command": command,
                    "working_directory": str(working_directory),
                    "return_code": return_code,
                    "error": stderr.strip() or stdout[-1000:].strip(),
                    "resource_sampling_available": bool(normalized_resources),
                    "events_path": str(_trainer_output_file(target_dir, outputs.get("events_file"), "events.jsonl")),
                    "metric_diagnostics": metric_diagnostics,
                },
                logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
                provenance=provenance,
            )
            return {"status": status, "training_run_path": record["path"]}

        payload = _trainer_result_payload(manifest, target_dir, stdout, command)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        history = payload.get("history") if isinstance(payload.get("history"), dict) else {}
        history_steps = payload.get("history_steps") if isinstance(payload.get("history_steps"), dict) else {}
        history.update(normalized_resources)
        history_steps.update(resource_steps)
        for key, values in normalized_resources.items():
            if values:
                metrics.setdefault(key, values[-1])
        events_path = str(_trainer_output_file(target_dir, outputs.get("events_file"), "events.jsonl"))
        metric_diagnostics = training_metric_diagnostics(
            {
                "status": "completed",
                "metrics": metrics,
                "history": history,
                "history_steps": history_steps,
                "summary": {"events_path": events_path},
            }
        )
        artifact_path = str(
            payload.get("checkpoint_path")
            or payload.get("model_path")
            or payload.get("artifact_path")
            or payload.get("output_dir")
            or outputs.get("artifact_path")
            or target_dir
        )
        artifact_path = str(_resolve_trainer_output_path(target_dir, artifact_path))
        artifact_type = str(payload.get("artifact_type") or outputs.get("artifact_type") or "artifact")
        try:
            _require_training_artifact(artifact_path)
        except FileNotFoundError as exc:
            record = write_training_run_record(
                target_dir,
                preset_id=manifest["id"],
                status="failed",
                dataset_root=dataset_root,
                adapter=adapter,
                sequence_id=sequence_id,
                split_path=split_path,
                artifact_path=artifact_path,
                artifact_type=artifact_type,
                metrics=metrics,
                history=history,
                history_steps=history_steps,
                parameters=resolved_parameters,
                summary={
                    "trainer_manifest_path": str(Path(manifest_path).resolve()),
                    "command": command,
                    "working_directory": str(working_directory),
                    "training_job_path": str(job.state_path),
                    "error": str(exc),
                    "events_path": events_path,
                },
                logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
                provenance=provenance,
            )
            return {"status": "failed", "error": str(exc), "training_run_path": record["path"]}
        record = write_training_run_record(
            target_dir,
            preset_id=manifest["id"],
            status="completed",
            dataset_root=dataset_root,
            adapter=adapter,
            sequence_id=sequence_id,
            split_path=split_path,
            artifact_path=artifact_path,
            artifact_type=artifact_type,
            metrics=metrics,
            history=history,
            history_steps=history_steps,
            parameters=resolved_parameters,
            summary={
                "trainer_manifest_path": str(Path(manifest_path).resolve()),
                "command": command,
                "working_directory": str(working_directory),
                "training_job_path": str(job.state_path),
                "resource_sampling_available": bool(normalized_resources),
                "events_path": events_path,
                "metric_diagnostics": metric_diagnostics,
            },
            logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
            provenance=provenance,
        )
        payload.update(
            status="completed",
            output_dir=str(target_dir.resolve()),
            metrics=metrics,
            history=history,
            history_steps=history_steps,
            artifact_path=artifact_path,
            artifact_type=artifact_type,
            training_run_path=record["path"],
            metric_diagnostics=metric_diagnostics,
            command=command,
        )
        return payload

    return queue.submit(
        command=command,
        working_directory=working_directory,
        environment=environment,
        output_dir=target_dir,
        metadata={
            "trainer_id": manifest["id"],
            "trainer_label": manifest["label"],
            "dataset_root": dataset_root,
            "adapter": adapter,
            "sequence_id": sequence_id,
            "split_path": split_path,
            "trainer_manifest_path": str(Path(manifest_path).resolve()),
            "events_path": str(_trainer_output_file(target_dir, outputs.get("events_file"), "events.jsonl")),
        },
        finalizer=finalize,
    )


def queue_training_config_job(
    queue: TrainingJobQueue,
    training_config: str | dict[str, Any],
    *,
    config_path: str | Path | None = None,
    trainer_root: str | Path | None = None,
) -> ProcessTrainingJob:
    """Validate and queue a reusable manifest-backed training config."""

    report = validate_training_config_setup(training_config, config_path=config_path, trainer_root=trainer_root)
    if not report.get("ready"):
        raise ValueError("; ".join(str(item) for item in report.get("issues", [])) or "Training config is not ready.")
    row = report["training_config"]
    preset = report["training_preset"]
    manifest_path = str(preset.get("manifest_path") or "").strip()
    if not manifest_path:
        raise ValueError("Only manifest-backed trainers can be queued as subprocess jobs.")
    return queue_trainer_manifest_job(
        queue,
        manifest_path,
        dataset_root=str(row.get("dataset_root") or ""),
        output_dir=str(report.get("output_path") or row.get("output_path") or ""),
        parameters=report.get("parameters") if isinstance(report.get("parameters"), dict) else {},
        adapter=str(row.get("adapter") or ""),
        sequence_id=str(row.get("sequence_id") or ""),
        split_path=str(row.get("split_path") or ""),
    )


def validate_inference_setup(
    manifest_path: str | Path,
    *,
    artifact_path: str,
    dataset_root: str,
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
    parameters: dict[str, Any] | None = None,
    output_dir: str = "",
) -> dict[str, Any]:
    """Validate a manifest-declared inference run without starting it."""

    issues: list[str] = []
    manifest = load_trainer_manifest(manifest_path)
    inference = manifest.get("inference") if isinstance(manifest.get("inference"), dict) else {}
    if not inference:
        issues.append("Trainer manifest does not declare an inference entrypoint.")
    artifact = Path(artifact_path) if artifact_path else None
    if artifact is None:
        issues.append("Model artifact path is required.")
    elif not artifact.exists():
        issues.append(f"Model artifact not found: {artifact}")
    dataset = Path(dataset_root) if dataset_root else None
    if dataset is None:
        issues.append("Dataset root is required for inference.")
    elif not dataset.exists():
        issues.append(f"Dataset root not found: {dataset}")

    schema = inference.get("parameters") if isinstance(inference.get("parameters"), dict) else {}
    resolved_parameters: dict[str, Any] = {}
    try:
        resolved_parameters = _trainer_parameters(schema, parameters or {})
    except ValueError as exc:
        issues.append(str(exc))
    compatibility: dict[str, Any] = {"compatible": True, "issues": []}
    if inference:
        try:
            _validate_trainer_launch({**inference, "manifest_dir": str(manifest["manifest_dir"])})
        except Exception as exc:
            issues.append(f"Inference launch is invalid: {exc}")
    if inference and dataset is not None and dataset.exists():
        compatibility = _trainer_dataset_compatibility(
            {"input": dict(inference.get("input") or {})},
            dataset_root=dataset_root,
            adapter=adapter,
            sequence_id=sequence_id,
            split_path=split_path,
        )
        issues.extend(str(item) for item in compatibility.get("issues", []))
    target_dir = Path(
        output_dir
        or ROOT / "outputs" / "inference_runs" / f"{manifest['id']}_{_timestamp()}"
    )
    command: list[str] = []
    working_directory = ""
    if inference and artifact is not None and not issues:
        try:
            command = _inference_command(
                inference,
                manifest_dir=Path(str(manifest["manifest_dir"])),
                artifact_path=str(artifact),
                dataset_root=dataset_root,
                output_dir=target_dir,
                parameters=resolved_parameters,
                adapter=adapter,
                sequence_id=sequence_id,
                split_path=split_path,
            )
            working_directory = str(
                resolve_trainer_working_directory(inference, Path(str(manifest["manifest_dir"])))
            )
        except Exception as exc:
            issues.append(f"Inference launch is invalid: {exc}")
    return {
        "ready": not issues,
        "status": "ready" if not issues else "invalid",
        "issues": issues,
        "manifest": manifest,
        "inference": inference,
        "artifact_path": str(artifact.resolve()) if artifact is not None and artifact.exists() else artifact_path,
        "dataset_root": dataset_root,
        "adapter": adapter,
        "sequence_id": sequence_id,
        "split_path": split_path,
        "parameters": resolved_parameters,
        "compatibility": compatibility,
        "output_dir": str(target_dir.resolve()),
        "command_preview": command,
        "working_directory": working_directory,
    }


def inference_parameter_defaults(manifest_path: str | Path) -> dict[str, Any]:
    """Return validated default parameters for a manifest inference entrypoint."""

    manifest = load_trainer_manifest(manifest_path)
    inference = manifest.get("inference") if isinstance(manifest.get("inference"), dict) else {}
    if not inference:
        return {}
    schema = inference.get("parameters") if isinstance(inference.get("parameters"), dict) else {}
    return _trainer_parameters(schema, {})


def run_inference_manifest_job(
    manifest_path: str | Path,
    *,
    artifact_path: str,
    dataset_root: str,
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
    parameters: dict[str, Any] | None = None,
    output_dir: str = "",
) -> dict[str, Any]:
    """Run generic checkpoint inference and persist predictions and metrics."""

    report = validate_inference_setup(
        manifest_path,
        artifact_path=artifact_path,
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=sequence_id,
        split_path=split_path,
        parameters=parameters,
        output_dir=output_dir,
    )
    if not report["ready"]:
        raise ValueError("; ".join(str(item) for item in report["issues"]))
    manifest = report["manifest"]
    inference = report["inference"]
    target_dir = Path(report["output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)
    command = list(report["command_preview"])
    manifest_dir = Path(str(manifest["manifest_dir"]))
    working_directory = resolve_trainer_working_directory(inference, manifest_dir)
    environment = resolve_trainer_environment(inference, manifest_dir)
    completed = subprocess.run(
        command,
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path = target_dir / "stdout.log"
    stderr_path = target_dir / "stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    outputs = inference.get("outputs") if isinstance(inference.get("outputs"), dict) else {}

    def persist_failure(error: str) -> dict[str, Any]:
        failed = {
            "run_id": target_dir.name,
            "status": "failed",
            "trainer_id": manifest["id"],
            "trainer_manifest_path": str(Path(manifest_path).resolve()),
            "artifact_path": report["artifact_path"],
            "dataset_root": str(Path(dataset_root).resolve()),
            "adapter": adapter,
            "sequence_id": sequence_id,
            "split_path": split_path,
            "parameters": report["parameters"],
            "metrics": {},
            "history": {},
            "predictions": [],
            "prediction_count": 0,
            "previews": {},
            "error": error,
            "command": command,
            "logs": {"stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve())},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (target_dir / INFERENCE_RUN_FILENAME).write_text(
            json.dumps(failed, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return failed

    if completed.returncode != 0:
        failed = persist_failure((completed.stderr or completed.stdout or "Inference command failed").strip())
        raise RuntimeError(failed["error"])
    try:
        payload = _trainer_result_payload({"outputs": outputs}, target_dir, completed.stdout or "", command)
        predictions = payload.get("predictions") if isinstance(payload.get("predictions"), list) else []
        predictions_path = _trainer_output_file(target_dir, outputs.get("predictions_file"), "predictions.json")
        if not predictions and predictions_path.is_file():
            raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            predictions = (
                raw_predictions
                if isinstance(raw_predictions, list)
                else list(raw_predictions.get("predictions", []))
            )
        previews = _normalize_inference_previews(payload.get("previews"), target_dir)
        configured_preview = str(outputs.get("preview_file") or "").strip()
        if configured_preview:
            preview_path = _resolve_trainer_output_path(target_dir, configured_preview)
            if preview_path.is_file():
                previews.setdefault("primary", str(preview_path))
        record = {
            "run_id": target_dir.name,
            "status": "completed",
            "trainer_id": manifest["id"],
            "trainer_manifest_path": str(Path(manifest_path).resolve()),
            "artifact_path": report["artifact_path"],
            "dataset_root": str(Path(dataset_root).resolve()),
            "adapter": adapter,
            "sequence_id": sequence_id,
            "split_path": split_path,
            "parameters": report["parameters"],
            "metrics": dict(payload.get("metrics") or {}),
            "history": dict(payload.get("history") or {}),
            "predictions": predictions,
            "prediction_count": len(predictions),
            "previews": previews,
            "command": command,
            "logs": {"stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve())},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        record_path = target_dir / INFERENCE_RUN_FILENAME
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        record["path"] = str(record_path.resolve())
        record["output_dir"] = str(target_dir.resolve())
        return record
    except Exception as exc:
        failed = persist_failure(f"Inference post-processing failed: {exc}")
        raise RuntimeError(failed["error"]) from exc


def run_training_config_job(
    training_config: str | dict[str, Any],
    *,
    config_path: str | Path | None = None,
    trainer_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a reusable training config through the neutral trainer boundary."""

    report = validate_training_config_setup(training_config, config_path=config_path, trainer_root=trainer_root)
    if not report.get("ready"):
        issues = report.get("issues") if isinstance(report.get("issues"), list) else []
        raise ValueError("; ".join(str(item) for item in issues) or "Training config is not ready.")
    row = report["training_config"]
    preset = report["training_preset"]
    preset_id = str(row.get("training_preset_id") or "")
    parameters = report.get("parameters") if isinstance(report.get("parameters"), dict) else {}
    output_path = str(report.get("output_path") or row.get("output_path") or "")
    dataset_root = str(row.get("dataset_root") or "")
    adapter = str(row.get("adapter") or "")
    sequence_id = str(row.get("sequence_id") or "")
    split_path = str(row.get("split_path") or "")

    manifest_path = str(preset.get("manifest_path") or "").strip()
    if manifest_path:
        payload = run_trainer_manifest_job(
            manifest_path,
            dataset_root=dataset_root,
            output_dir=output_path,
            parameters=parameters,
            adapter=adapter,
            sequence_id=sequence_id,
            split_path=split_path,
        )
    elif preset_id == "stablewm_hdf5":
        payload = export_lewm_hdf5(
            dataset_root,
            output_path,
            adapter=adapter,
            sequence_id=sequence_id,
            image_size=int(parameters.get("image_size", 64)),
        )
    elif preset_id == "tiny_world_model":
        payload = train_tiny_world_model(
            dataset_root,
            output_path,
            adapter=adapter,
            sequence_id=sequence_id,
            ridge=float(parameters.get("ridge", 1e-4)),
        )
    elif preset_id == "lewm_cost_model":
        input_hdf5 = str(parameters.get("input_hdf5") or dataset_root)
        payload = train_lewm_cost_model(input_hdf5, output_path)
    else:
        raise ValueError(f"Training preset is not runnable: {preset_id or NAN_TEXT}")

    payload["training_config"] = row
    return payload


def validate_training_config_setup(
    training_config: str | dict[str, Any],
    *,
    config_path: str | Path | None = None,
    trainer_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a reusable training config without launching the trainer.

    The GUI uses this as a neutral adapter boundary: built-in trainers, imported
    trainer manifests, and future model algorithms all reduce to the same
    dataset/preset/parameter/output contract before any long-running work starts.
    """

    row = _resolve_training_config(training_config, config_path=config_path)
    _materialize_training_config_assets(row)
    preset_id = str(row.get("training_preset_id") or "")
    presets = {str(item.get("id") or ""): dict(item) for item in training_preset_entries(trainer_root)}
    preset = presets.get(preset_id, {})
    issues: list[str] = []

    dataset_root = str(row.get("dataset_root") or "").strip()
    dataset_path = Path(dataset_root) if dataset_root else None
    dataset_exists = bool(dataset_path and dataset_path.exists())
    if not dataset_root:
        issues.append("Dataset root is required.")
    elif not dataset_exists:
        issues.append(f"Dataset root not found: {dataset_root}")

    if not preset:
        issues.append(f"Training preset not found: {preset_id or NAN_TEXT}")
    elif preset.get("available") is False:
        issues.append(f"Training preset is not available: {preset.get('label') or preset_id}")

    schema = preset.get("parameters") if isinstance(preset.get("parameters"), dict) else {}
    parameters: dict[str, Any] = {}
    try:
        parameters = _trainer_parameters(schema, row.get("parameters") if isinstance(row.get("parameters"), dict) else {})
    except ValueError as exc:
        issues.append(str(exc))

    output_path = str(row.get("output_path") or "").strip()
    if not output_path:
        output_path = str(ROOT / "outputs" / "training_runs" / f"{row['id']}_{_timestamp()}")

    command_preview: list[str] = []
    manifest: dict[str, Any] = {}
    manifest_path = str(preset.get("manifest_path") or "").strip()
    compatibility: dict[str, Any] = {
        "compatible": True,
        "adapter": str(row.get("adapter") or ""),
        "available_modalities": [],
        "required_modalities": [],
        "issues": [],
    }
    if manifest_path:
        try:
            manifest = load_trainer_manifest(manifest_path)
            _validate_trainer_launch(manifest)
            if dataset_exists:
                compatibility = _trainer_dataset_compatibility(
                    manifest,
                    dataset_root=dataset_root,
                    adapter=str(row.get("adapter") or ""),
                    sequence_id=str(row.get("sequence_id") or ""),
                    split_path=str(row.get("split_path") or ""),
                )
                issues.extend(str(item) for item in compatibility["issues"])
            if parameters or not schema:
                command_preview = _trainer_command(
                    manifest,
                    dataset_root,
                    Path(output_path),
                    parameters,
                    str(row.get("adapter") or ""),
                    str(row.get("sequence_id") or ""),
                    str(row.get("split_path") or ""),
                )
        except Exception as exc:
            issues.append(f"Trainer manifest is invalid: {exc}")

    status = "ready" if not issues else "invalid"
    if preset and preset.get("available") is False:
        status = "unfinished"
    return {
        "ready": status == "ready",
        "status": status,
        "issues": issues,
        "training_config": row,
        "training_preset": preset,
        "trainer_manifest": manifest,
        "dataset": {
            "root": dataset_root,
            "exists": dataset_exists,
            "adapter": str(row.get("adapter") or ""),
            "sequence_id": str(row.get("sequence_id") or ""),
            "split_path": str(row.get("split_path") or ""),
        },
        "compatibility": compatibility,
        "parameters": parameters,
        "parameter_schema": dict(schema),
        "output_path": output_path,
        "command_preview": command_preview,
    }


def write_training_run_record(
    run_dir: str | Path,
    *,
    preset_id: str,
    status: str,
    dataset_root: str = "",
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
    artifact_path: str = "",
    artifact_type: str = "",
    metrics: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
    history_steps: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    logs: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    record_path = target_dir / TRAINING_RUN_FILENAME
    preset = next((row for row in training_preset_entries() if row["id"] == preset_id), {})
    normalized_history, normalized_history_steps = _normalize_metric_series(
        history or {},
        history_steps or {},
    )
    payload: dict[str, Any] = {
        "run_id": _safe_name(target_dir.name or preset_id),
        "preset_id": preset_id,
        "preset_label": str(preset.get("label") or preset_id),
        "status": status,
        "dataset_root": dataset_root,
        "adapter": adapter,
        "sequence_id": sequence_id,
        "split_path": str(Path(split_path).resolve()) if split_path else "",
        "artifact_path": str(Path(artifact_path).resolve()) if artifact_path else "",
        "artifact_type": artifact_type,
        "metrics": dict(metrics or {}),
        "history": normalized_history,
        "history_steps": normalized_history_steps,
        "parameters": dict(parameters or {}),
        "summary": dict(summary or {}),
        "logs": _normalize_log_paths(logs or {}),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload.update(dict(provenance or {}))
    record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    payload["path"] = str(record_path.resolve())
    payload["relative_path"] = _relative_to_root(record_path)
    return payload


def training_run_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    output_root = Path(root or ROOT / "outputs")
    if not output_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in output_root.glob(f"**/{TRAINING_RUN_FILENAME}"):
        try:
            data = _read_json(path)
            stat = path.stat()
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        row = dict(data)
        row["path"] = str(path.resolve())
        row["relative_path"] = _relative_to_root(path)
        row["mtime"] = stat.st_mtime
        row["label"] = str(row.get("preset_label") or row.get("preset_id") or path.parent.name)
        rows.append(row)
    return sorted(rows, key=lambda row: (-float(row.get("mtime", 0.0)), str(row.get("label", ""))))


def training_job_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Discover persisted jobs; active states from old app sessions are interrupted."""

    output_root = Path(root or ROOT / "outputs")
    if not output_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in output_root.glob(f"**/{TRAINING_JOB_FILENAME}"):
        try:
            data = _read_json(path)
            stat = path.stat()
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        row = dict(data)
        if row.get("status") in {"queued", "running", "canceling"}:
            row["status"] = "interrupted"
            row["message"] = "Previous desktop session ended before this job completed."
        row["path"] = str(path.resolve())
        row["relative_path"] = _relative_to_root(path)
        row["mtime"] = stat.st_mtime
        rows.append(row)
    return sorted(rows, key=lambda row: -float(row.get("mtime", 0.0)))


def training_artifact_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return training artifacts with source-run and inference capability metadata."""

    rows: list[dict[str, Any]] = []
    for run in training_run_entries(root):
        artifact_value = _training_record_artifact_path(run)
        if not artifact_value:
            continue
        artifact = Path(artifact_value)
        exists = artifact.exists()
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        manifest_path = str(summary.get("trainer_manifest_path") or "").strip()
        inference_available = False
        if manifest_path and Path(manifest_path).is_file():
            try:
                loaded_manifest = load_trainer_manifest(manifest_path)
                inference = loaded_manifest.get("inference") if isinstance(loaded_manifest.get("inference"), dict) else {}
                if inference:
                    _validate_trainer_launch(
                        {**inference, "manifest_dir": str(loaded_manifest["manifest_dir"])}
                    )
                    inference_available = True
            except Exception:
                inference_available = False
        rows.append(
            {
                "id": str(run.get("run_id") or artifact.stem),
                "label": str(run.get("preset_label") or run.get("preset_id") or artifact.stem),
                "artifact_path": str(artifact.resolve()),
                "artifact_type": str(run.get("artifact_type") or "artifact"),
                "exists": exists,
                "size_bytes": _artifact_size_bytes(artifact) if exists else 0,
                "status": str(run.get("status") or ""),
                "training_run_path": str(run.get("path") or ""),
                "trainer_manifest_path": manifest_path,
                "inference_available": inference_available,
                "dataset_root": str(run.get("dataset_root") or ""),
                "adapter": str(run.get("adapter") or ""),
                "sequence_id": str(run.get("sequence_id") or ""),
                "split_path": str(run.get("split_path") or ""),
                "metrics": dict(run.get("metrics") if isinstance(run.get("metrics"), dict) else {}),
                "parameters": dict(run.get("parameters") if isinstance(run.get("parameters"), dict) else {}),
                "epoch": _training_artifact_epoch(run),
                "favorite": bool(run.get("favorite")),
                "best": bool(run.get("best_marks")),
                "best_marks": dict(run.get("best_marks") if isinstance(run.get("best_marks"), dict) else {}),
                "created_at": run.get("created_at"),
                "mtime": run.get("mtime", 0.0),
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("created_at")),
            str(row.get("created_at") or ""),
            float(row.get("mtime", 0.0)),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows):
        row["latest"] = index == 0
    return rows


def set_training_artifact_favorite(training_run_path: str | Path, favorite: bool) -> dict[str, Any]:
    path = Path(training_run_path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Training run is invalid: {path}")
    payload["favorite"] = bool(favorite)
    payload["favorite_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    payload["path"] = str(path.resolve())
    return payload


def delete_training_artifact(
    training_run_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Delete one unreferenced artifact while preserving its training record."""

    run_path = Path(training_run_path).resolve()
    record = _read_json(run_path)
    if not isinstance(record, dict):
        raise ValueError(f"Training run is invalid: {run_path}")
    artifact_value = _training_record_artifact_path(record)
    if not artifact_value:
        raise ValueError("Training run does not declare an artifact path.")
    artifact = Path(artifact_value).resolve()
    allowed_root = Path(output_root or ROOT / "outputs").resolve()
    try:
        artifact.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Artifact is outside the managed output root: {artifact}") from exc
    if artifact in {allowed_root, run_path.parent} or (artifact.is_dir() and (artifact / TRAINING_RUN_FILENAME).is_file()):
        raise ValueError("Refusing to delete an entire training-run directory as an artifact.")
    references = _configuration_references_for_paths([artifact, run_path])
    if references:
        raise ValueError("Artifact is referenced by configuration: " + ", ".join(references))
    if not artifact.exists():
        raise FileNotFoundError(f"Training artifact not found: {artifact}")
    if artifact.is_dir():
        shutil.rmtree(artifact)
    else:
        artifact.unlink()
    record["artifact_deleted_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["favorite"] = False
    run_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {
        "training_run_path": str(run_path),
        "artifact_path": str(artifact),
        "deleted": True,
    }


def _training_artifact_epoch(run: dict[str, Any]) -> float | None:
    metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
    for key in ("epoch", "best_epoch", "final_epoch"):
        value = _float_or_nan(metrics.get(key))
        if math.isfinite(value):
            return float(value)
    history = training_metric_history(run)
    for key in ("epoch", "global_step", "step"):
        if history.get(key):
            return float(history[key][-1])
    return None


def inference_run_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    output_root = Path(root or ROOT / "outputs")
    if not output_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in output_root.glob(f"**/{INFERENCE_RUN_FILENAME}"):
        try:
            payload = _read_json(path)
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["path"] = str(path.resolve())
        row["relative_path"] = _relative_to_root(path)
        row["mtime"] = mtime
        rows.append(row)
    return sorted(rows, key=lambda row: -float(row.get("mtime", 0.0)))


def training_metric_history(record: dict[str, Any]) -> dict[str, list[float]]:
    raw_history = record.get("history") if isinstance(record.get("history"), dict) else {}
    raw_steps = record.get("history_steps") if isinstance(record.get("history_steps"), dict) else {}
    history, _ = _normalize_metric_series(raw_history, raw_steps)
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in _numeric_metric_items(metrics):
            history.setdefault(key, [value])
    return history


def training_metric_steps(record: dict[str, Any]) -> dict[str, list[float]]:
    raw_history = record.get("history") if isinstance(record.get("history"), dict) else {}
    raw_steps = record.get("history_steps") if isinstance(record.get("history_steps"), dict) else {}
    history, steps = _normalize_metric_series(raw_history, raw_steps)
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in _numeric_metric_items(metrics):
            if key not in history:
                steps[key] = [0.0]
    return steps


def live_training_metric_record(job: dict[str, Any]) -> dict[str, Any]:
    """Build a transient training record from events and sampled resources."""

    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    output_dir = Path(str(job.get("output_dir") or ROOT / "outputs"))
    events_value = str(metadata.get("events_path") or output_dir / "events.jsonl")
    history, history_steps = _read_metric_events_with_steps(Path(events_value))
    resource_history = job.get("resource_history") if isinstance(job.get("resource_history"), dict) else {}
    elapsed = resource_history.get("elapsed_sec")
    elapsed_values = list(elapsed) if isinstance(elapsed, list) else []
    for key, values in resource_history.items():
        if isinstance(values, list) and values:
            metric_values: list[float] = []
            metric_steps: list[float] = []
            for index, raw_value in enumerate(values):
                value = _float_or_nan(raw_value)
                if not math.isfinite(value):
                    continue
                raw_step = elapsed_values[index] if index < len(elapsed_values) else index
                step = _float_or_nan(raw_step)
                metric_values.append(float(value))
                metric_steps.append(float(step) if math.isfinite(step) else float(index))
            if metric_values:
                history[f"resource.{key}"] = metric_values
                history_steps[f"resource.{key}"] = metric_steps
    record = {
        "run_id": job.get("job_id"),
        "status": job.get("status"),
        "history": history,
        "history_steps": history_steps,
        "summary": {"events_path": events_value},
        "mtime": time.time(),
    }
    record["metric_diagnostics"] = training_metric_diagnostics(record)
    return record


def training_metric_diagnostics(
    record: dict[str, Any],
    *,
    stale_window: int = 20,
    explosion_ratio: float = 20.0,
    no_update_seconds: float = 120.0,
) -> dict[str, Any]:
    """Detect non-finite, exploding, stalled, and inactive training metrics."""

    warnings: list[dict[str, Any]] = []
    warning_keys: set[tuple[str, str]] = set()
    raw_series = _raw_training_metric_series(record)
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    events_path = Path(str(summary.get("events_path") or "")) if summary.get("events_path") else None
    if events_path is not None and events_path.is_file():
        for key, values in _read_raw_metric_events(events_path).items():
            raw_series.setdefault(key, []).extend(values)

    def add(severity: str, code: str, metric: str, message: str, **details: Any) -> None:
        dedupe_key = (code, metric)
        if dedupe_key in warning_keys:
            return
        warning_keys.add(dedupe_key)
        warnings.append(
            {
                "severity": severity,
                "code": code,
                "metric": metric,
                "message": message,
                **details,
            }
        )

    for metric, values in raw_series.items():
        if any(not math.isfinite(value) for value in values):
            add("critical", "non_finite", metric, f"{metric} contains NaN or infinity.")

    history = training_metric_history(record)
    history_steps = training_metric_steps(record)
    for metric, values in history.items():
        lowered = metric.lower()
        if "loss" in lowered and len(values) >= 2:
            for index, value in enumerate(values[1:], start=1):
                previous = [abs(item) for item in values[:index] if math.isfinite(item)]
                baseline = min(previous) if previous else 0.0
                threshold = max(1.0, baseline * float(explosion_ratio))
                if abs(value) > threshold:
                    add(
                        "critical",
                        "loss_explosion",
                        metric,
                        f"{metric} jumped to {value:.6g} at point {index + 1}.",
                        point_index=index,
                        value=value,
                        threshold=threshold,
                    )
                    break
        if (
            len(values) >= stale_window
            and not lowered.startswith("resource.")
            and lowered not in {"step", "epoch", "iteration", "global_step"}
        ):
            tail = values[-stale_window:]
            tolerance = max(1e-12, abs(sum(tail) / len(tail)) * 1e-5)
            if max(tail) - min(tail) <= tolerance:
                add(
                    "warning",
                    "metric_stalled",
                    metric,
                    f"{metric} has not changed across the last {stale_window} points.",
                    point_index=len(values) - 1,
                    value=values[-1],
                )

    status = str(record.get("status") or "").lower()
    if status in {"queued", "running", "canceling"} and events_path is not None:
        try:
            age_seconds = max(0.0, time.time() - events_path.stat().st_mtime)
        except OSError:
            age_seconds = math.inf
        if age_seconds >= no_update_seconds:
            add(
                "warning",
                "no_recent_update",
                "events",
                f"No metric event has been written for {age_seconds:.0f} seconds.",
                age_seconds=age_seconds,
            )

    severity_rank = {"healthy": 0, "warning": 1, "critical": 2}
    diagnostic_status = "healthy"
    for warning in warnings:
        severity = str(warning.get("severity") or "warning")
        if severity_rank.get(severity, 1) > severity_rank[diagnostic_status]:
            diagnostic_status = severity
    return {
        "status": diagnostic_status,
        "warning_count": len(warnings),
        "warnings": warnings,
        "metric_count": len(history),
        "point_count": sum(len(values) for values in history.values()),
        "resource_metrics": sorted(key for key in history if key.startswith("resource.")),
    }


def export_training_metrics(record: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Persist normalized metrics, diagnostics, and a rectangular CSV table."""

    record_path = Path(str(record.get("path") or "")) if record.get("path") else None
    if output_dir is not None:
        target_dir = Path(output_dir)
    elif record_path is not None:
        target_dir = record_path.parent / "metric_exports"
    else:
        target_dir = ROOT / "outputs" / "training_reports" / f"metrics_{_timestamp()}"
    target_dir.mkdir(parents=True, exist_ok=True)
    history = training_metric_history(record)
    history_steps = training_metric_steps(record)
    diagnostics = training_metric_diagnostics(record)
    json_path = target_dir / "training_metrics.json"
    csv_path = target_dir / "training_metrics.csv"
    payload = {
        "schema_version": 1,
        "run_id": record.get("run_id"),
        "preset_id": record.get("preset_id"),
        "status": record.get("status"),
        "metrics": record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
        "history": history,
        "history_steps": history_steps,
        "diagnostics": diagnostics,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    metric_names = sorted(history)
    all_steps = sorted({step for name in metric_names for step in history_steps.get(name, [])})
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", *metric_names])
        value_maps = {
            name: dict(zip(history_steps.get(name, []), history[name], strict=False)) for name in metric_names
        }
        for step in all_steps:
            writer.writerow(
                [
                    step,
                    *[value_maps[name].get(step, "") for name in metric_names],
                ]
            )
    return {
        "output_dir": str(target_dir.resolve()),
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve()),
        "metric_count": len(metric_names),
        "point_count": len(all_steps),
        "diagnostics": diagnostics,
    }


def filter_training_runs(
    runs: list[dict[str, Any]],
    *,
    query: str = "",
    preset_id: str = "",
    dataset_root: str = "",
    status: str = "",
    date_from: str = "",
    date_to: str = "",
) -> list[dict[str, Any]]:
    """Filter experiment records without assuming a model-specific schema."""

    query_text = query.strip().lower()
    filtered: list[dict[str, Any]] = []
    for run in runs:
        created_date = str(run.get("created_at") or "")[:10]
        haystack = " ".join(
            str(run.get(key) or "")
            for key in ("run_id", "preset_id", "preset_label", "dataset_root", "adapter", "artifact_path")
        ).lower()
        if query_text and query_text not in haystack:
            continue
        if preset_id and str(run.get("preset_id") or "") != preset_id:
            continue
        if dataset_root and str(run.get("dataset_root") or "") != dataset_root:
            continue
        if status and str(run.get("status") or "") != status:
            continue
        if date_from and (not created_date or created_date < date_from):
            continue
        if date_to and (not created_date or created_date > date_to):
            continue
        filtered.append(dict(run))
    return filtered


def experiment_metric_names(runs: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for run in runs:
        names.update(training_metric_history(run))
    return sorted(names)


def compare_training_runs(
    runs: list[dict[str, Any]],
    *,
    metric: str = "",
    direction: str = "auto",
) -> dict[str, Any]:
    """Rank comparable runs and return parameter, metric, and curve evidence."""

    if len(runs) < 2:
        raise ValueError("Select at least two training runs for comparison.")
    available_metrics = experiment_metric_names(runs)
    selected_metric = metric if metric in available_metrics else _default_comparison_metric(available_metrics)
    if not selected_metric:
        raise ValueError("Selected runs do not contain comparable numeric metrics.")
    resolved_direction = _metric_direction(selected_metric) if direction == "auto" else direction
    if resolved_direction not in {"min", "max"}:
        raise ValueError("Comparison direction must be auto, min, or max.")

    rows: list[dict[str, Any]] = []
    curves: dict[str, list[float]] = {}
    curve_steps: dict[str, list[float]] = {}
    for index, run in enumerate(runs):
        history = training_metric_history(run)
        steps = training_metric_steps(run)
        values = history.get(selected_metric, [])
        if not values:
            continue
        run_id = str(run.get("run_id") or f"run_{index + 1}")
        label = str(run.get("preset_label") or run.get("preset_id") or run_id)
        curve_key = f"{label} [{run_id}]"
        curves[curve_key] = list(values)
        curve_steps[curve_key] = list(steps.get(selected_metric, range(len(values))))
        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "status": str(run.get("status") or ""),
                "created_at": run.get("created_at"),
                "dataset_root": str(run.get("dataset_root") or ""),
                "preset_id": str(run.get("preset_id") or ""),
                "artifact_path": str(run.get("artifact_path") or ""),
                "metric": selected_metric,
                "value": float(values[-1]),
                "parameters": dict(run.get("parameters") if isinstance(run.get("parameters"), dict) else {}),
                "final_metrics": {
                    key: series[-1] for key, series in history.items() if series
                },
                "path": str(run.get("path") or ""),
            }
        )
    if len(rows) < 2:
        raise ValueError(f"At least two selected runs must contain metric: {selected_metric}")
    rows.sort(key=lambda row: float(row["value"]), reverse=resolved_direction == "max")
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["best"] = rank == 1
    parameter_names = sorted({key for row in rows for key in row["parameters"]})
    return {
        "metric": selected_metric,
        "direction": resolved_direction,
        "available_metrics": available_metrics,
        "rows": rows,
        "best_run_id": rows[0]["run_id"],
        "best_value": rows[0]["value"],
        "parameter_names": parameter_names,
        "curves": curves,
        "curve_steps": curve_steps,
        "run_count": len(rows),
    }


def mark_best_training_run(run_path: str | Path, *, metric: str, direction: str, value: float) -> dict[str, Any]:
    path = Path(run_path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Training run is invalid: {path}")
    marks = payload.get("best_marks") if isinstance(payload.get("best_marks"), dict) else {}
    marks[str(metric)] = {
        "direction": direction,
        "value": float(value),
        "marked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload["best_marks"] = marks
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    payload["path"] = str(path.resolve())
    return payload


def clone_training_config_from_run(
    run: dict[str, Any],
    *,
    label: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    run_id = str(run.get("run_id") or run.get("preset_id") or "experiment")
    clone_label = label or f"{run.get('preset_label') or run.get('preset_id') or run_id} copy"
    return save_training_config(
        config_id=f"{_safe_name(run_id)}_copy_{_timestamp()}",
        label=clone_label,
        training_preset_id=str(run.get("preset_id") or "tiny_world_model"),
        dataset_root=str(run.get("dataset_root") or ""),
        adapter=str(run.get("adapter") or ""),
        sequence_id=str(run.get("sequence_id") or ""),
        split_path=str(run.get("split_path") or ""),
        output_path="",
        parameters=dict(run.get("parameters") if isinstance(run.get("parameters"), dict) else {}),
        path=path,
    )


def export_experiment_report(
    comparison: dict[str, Any],
    output_dir: str | Path,
    *,
    title: str = "Training Experiment Comparison",
) -> dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison.get("rows") if isinstance(comparison.get("rows"), list) else []
    parameter_names = comparison.get("parameter_names") if isinstance(comparison.get("parameter_names"), list) else []
    headers = ["Rank", "Run", "Preset", str(comparison.get("metric") or "Metric"), *parameter_names]
    table_rows = []
    for row in rows:
        parameters = row.get("parameters") if isinstance(row.get("parameters"), dict) else {}
        table_rows.append(
            [
                str(row.get("rank") or ""),
                str(row.get("run_id") or ""),
                str(row.get("label") or ""),
                display_value(row.get("value")),
                *[display_value(parameters.get(name)) for name in parameter_names],
            ]
        )
    markdown_lines = [
        f"# {_markdown_text(title, table=False)}",
        "",
        f"- Metric: {_markdown_text(comparison.get('metric') or NAN_TEXT, table=False)}",
        f"- Direction: {_markdown_text(comparison.get('direction') or NAN_TEXT, table=False)}",
        f"- Best run: {_markdown_text(comparison.get('best_run_id') or NAN_TEXT, table=False)}",
        f"- Best value: {_markdown_text(display_value(comparison.get('best_value')), table=False)}",
        "",
        "| " + " | ".join(_markdown_text(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(_markdown_text(value) for value in row) + " |" for row in table_rows)
    markdown_path = target_dir / "experiment_report.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    html_headers = "".join(f"<th>{escape(header)}</th>" for header in headers)
    html_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in table_rows
    )
    html_path = target_dir / "experiment_report.html"
    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>"
        + escape(title)
        + "</title><style>body{font-family:Segoe UI,Arial;margin:32px;color:#1d1d1f}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #d2d2d7;padding:8px;text-align:left}"
        "th{background:#f5f5f7}</style></head><body>"
        + f"<h1>{escape(title)}</h1><p>Metric: <b>{escape(str(comparison.get('metric') or NAN_TEXT))}</b> "
        + f"({escape(str(comparison.get('direction') or NAN_TEXT))}); best run: "
        + f"<b>{escape(str(comparison.get('best_run_id') or NAN_TEXT))}</b></p>"
        + f"<table><thead><tr>{html_headers}</tr></thead><tbody>{html_rows}</tbody></table></body></html>",
        encoding="utf-8",
    )
    return {
        "output_dir": str(target_dir.resolve()),
        "markdown_path": str(markdown_path.resolve()),
        "html_path": str(html_path.resolve()),
        "run_count": len(table_rows),
    }


def _markdown_text(value: Any, *, table: bool = True) -> str:
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("`", "\\`").replace("|", "\\|")
    return text.replace("\n", "<br>") if table else text.replace("\n", " ")


def cleanup_training_runs(
    run_paths: list[str | Path],
    *,
    output_root: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove only unreferenced failed or invalid run directories under output_root."""

    allowed_root = Path(output_root or ROOT / "outputs").resolve()
    removed: list[str] = []
    refused: list[dict[str, Any]] = []
    candidates: list[str] = []
    for value in run_paths:
        path = Path(value).resolve()
        run_dir = path.parent if path.name == TRAINING_RUN_FILENAME else path
        record_path = run_dir / TRAINING_RUN_FILENAME
        try:
            run_dir.relative_to(allowed_root)
        except ValueError:
            refused.append({"path": str(run_dir), "reason": "outside output root"})
            continue
        if run_dir == allowed_root or not record_path.is_file():
            refused.append({"path": str(run_dir), "reason": "missing or unsafe training_run.json"})
            continue
        record = _read_json(record_path)
        status = str(record.get("status") or "").lower() if isinstance(record, dict) else "invalid"
        artifact_value = _training_record_artifact_path(record) if isinstance(record, dict) else ""
        artifact_exists = bool(artifact_value and Path(artifact_value).exists())
        removable = status in {"failed", "canceled", "interrupted", "invalid"} or not artifact_exists
        if not removable:
            refused.append({"path": str(run_dir), "reason": "completed run has a valid artifact"})
            continue
        references = _training_run_references(record_path, artifact_value)
        if references:
            refused.append({"path": str(run_dir), "reason": "referenced by configuration", "references": references})
            continue
        candidates.append(str(run_dir))
        if not dry_run:
            shutil.rmtree(run_dir)
            removed.append(str(run_dir))
    return {
        "dry_run": dry_run,
        "candidates": candidates,
        "removed": removed,
        "refused": refused,
    }


def _default_comparison_metric(names: list[str]) -> str:
    priority = (
        "validation_loss",
        "val_loss",
        "validation_rmse",
        "validation_mse",
        "loss",
        "final_loss",
        "accuracy",
        "iou",
        "reward",
    )
    return next((name for name in priority if name in names), names[0] if names else "")


def _metric_direction(metric: str) -> str:
    lowered = metric.lower()
    minimize_tokens = ("loss", "error", "rmse", "mse", "mae", "distance", "latency", "risk", "collision")
    return "min" if any(token in lowered for token in minimize_tokens) else "max"


def _training_run_references(run_path: Path, artifact_path: str) -> list[str]:
    paths = [run_path.resolve()]
    if artifact_path:
        paths.append(Path(artifact_path).resolve())
    return _configuration_references_for_paths(paths)


def _configuration_references_for_paths(paths: list[Path]) -> list[str]:
    resolved_paths = {path.resolve() for path in paths}
    resolved_values = {str(path) for path in resolved_paths}
    needles = {
        candidate
        for value in resolved_values
        for candidate in (value, value.replace("\\", "/"), json.dumps(value, ensure_ascii=False)[1:-1])
    }
    references: list[str] = []
    if not CONFIG_ROOT.exists():
        return references
    for path in CONFIG_ROOT.rglob("*"):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matched = any(needle and needle in text for needle in needles)
        try:
            payload = _load_mapping_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
        if not matched:
            for value in _nested_string_values(payload):
                raw = value.strip()
                if not raw:
                    continue
                candidate = Path(raw)
                candidates = (
                    [candidate.resolve()]
                    if candidate.is_absolute()
                    else [(path.parent / candidate).resolve(), (ROOT / candidate).resolve()]
                )
                if any(
                    _path_references_target(candidate_path, target)
                    for candidate_path in candidates
                    for target in resolved_paths
                ):
                    matched = True
                    break
        if matched:
            references.append(str(path.resolve()))
    return sorted(references)


def _nested_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _nested_string_values(nested)]
    if isinstance(value, (list, tuple)):
        return [item for nested in value for item in _nested_string_values(nested)]
    return []


def _path_references_target(candidate: Path, target: Path) -> bool:
    if candidate == target:
        return True
    if target.is_dir():
        try:
            candidate.relative_to(target)
        except ValueError:
            return False
        return True
    return False


def _raw_training_metric_series(record: dict[str, Any]) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for source_name in ("history", "metrics"):
        source = record.get(source_name)
        if not isinstance(source, dict):
            continue
        for key, values in _raw_metric_series(source).items():
            rows.setdefault(key, []).extend(values)
    return rows


def _raw_metric_series(payload: dict[str, Any], prefix: str = "") -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    for key, raw in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(raw, dict):
            nested = _raw_metric_series(raw, name)
            for nested_name, values in nested.items():
                rows.setdefault(nested_name, []).extend(values)
            continue
        values = raw if isinstance(raw, (list, tuple)) else [raw]
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool):
                continue
            try:
                converted.append(float(value))
            except (TypeError, ValueError):
                continue
        if converted:
            rows[name] = converted
    return rows


def _read_raw_metric_events(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        for key, values in _raw_metric_series(event).items():
            rows.setdefault(key, []).extend(values)
    return rows


def _trainer_result_payload(manifest: dict[str, Any], output_dir: Path, stdout: str, command: list[str]) -> dict[str, Any]:
    """Collect a trainer result from stdout JSON and optional sidecar files."""

    payload: dict[str, Any] = {}
    if stdout.strip():
        try:
            payload = _json_from_command_output(stdout, command)
        except RuntimeError:
            payload = {}
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else _numeric_payload(payload)
    history = payload.get("history") if isinstance(payload.get("history"), dict) else {}
    history_steps = payload.get("history_steps") if isinstance(payload.get("history_steps"), dict) else {}

    metrics_from_file = _read_optional_json_mapping(_trainer_output_file(output_dir, outputs.get("metrics_file"), "metrics.json"))
    if metrics_from_file:
        metrics.update(metrics_from_file)

    history_from_file = _read_optional_json_mapping(_trainer_output_file(output_dir, outputs.get("history_file"), "history.json"))
    if history_from_file:
        history.update(history_from_file)

    history_steps_from_file = _read_optional_json_mapping(
        _trainer_output_file(output_dir, outputs.get("history_steps_file"), "history_steps.json")
    )
    if history_steps_from_file:
        history_steps.update(history_steps_from_file)

    event_history, event_steps = _read_metric_events_with_steps(
        _trainer_output_file(output_dir, outputs.get("events_file"), "events.jsonl")
    )
    if event_history:
        for key, values in event_history.items():
            existing = history.get(key)
            if not isinstance(existing, (list, tuple)) or not existing:
                history[key] = list(values)
            if len(history.get(key, [])) == len(values):
                history_steps[key] = list(event_steps.get(key, []))

    normalized_history, normalized_steps = _normalize_metric_series(history, history_steps)
    if not normalized_history:
        normalized_history = {key: [value] for key, value in _numeric_metric_items(metrics)}
    for key, values in normalized_history.items():
        if key.lower() in {"step", "epoch", "iteration", "global_step"}:
            continue
        if values and key not in metrics:
            metrics[key] = values[-1]

    payload["metrics"] = dict(metrics)
    payload["history"] = normalized_history
    payload["history_steps"] = normalized_steps
    return payload


def _trainer_output_file(output_dir: Path, configured: Any, default_name: str) -> Path:
    raw = str(configured or default_name).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return output_dir / path


def _resolve_trainer_output_path(output_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (output_dir / path).resolve()


def _require_training_artifact(artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path).resolve()
    if not artifact.exists():
        raise FileNotFoundError(f"Trainer declared an artifact that does not exist: {artifact}")
    return artifact


def _read_optional_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _read_metric_events(path: Path) -> dict[str, list[float]]:
    history, _ = _read_metric_events_with_steps(path)
    return history


def _read_metric_events_with_steps(path: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    if not path.exists():
        return {}, {}
    history: dict[str, list[float]] = {}
    history_steps: dict[str, list[float]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}, {}
    for event_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_step = _event_step_value(event, event_index)
        for key, value in _numeric_metric_items(event):
            history.setdefault(key, []).append(value)
            history_steps.setdefault(key, []).append(event_step)
    return history, history_steps


def _event_step_value(event: dict[str, Any], fallback: int) -> float:
    for key in ("global_step", "step", "iteration", "epoch"):
        value = _float_or_nan(event.get(key))
        if math.isfinite(value):
            return float(value)
    return float(fallback)


def _normalize_metric_history(raw: dict[str, Any]) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for key, value in raw.items():
        metric_name = str(key)
        if isinstance(value, (list, tuple)):
            values = [_float_or_nan(item) for item in value]
        else:
            values = [_float_or_nan(value)]
        finite_values = [float(item) for item in values if math.isfinite(item)]
        if finite_values:
            history[metric_name] = finite_values
    return history


def _normalize_metric_series(
    raw_history: dict[str, Any],
    raw_steps: dict[str, Any],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Normalize metric value/step pairs without shifting points after NaN filtering."""

    history: dict[str, list[float]] = {}
    steps: dict[str, list[float]] = {}
    for raw_name, raw_values in raw_history.items():
        name = str(raw_name)
        values = list(raw_values) if isinstance(raw_values, (list, tuple)) else [raw_values]
        raw_metric_steps = raw_steps.get(raw_name, raw_steps.get(name))
        candidates = list(raw_metric_steps) if isinstance(raw_metric_steps, (list, tuple)) else []
        normalized_values: list[float] = []
        normalized_steps: list[float] = []
        for index, raw_value in enumerate(values):
            value = _float_or_nan(raw_value)
            if not math.isfinite(value):
                continue
            step = _float_or_nan(candidates[index]) if index < len(candidates) else float(index)
            normalized_values.append(float(value))
            normalized_steps.append(float(step) if math.isfinite(step) else float(index))
        if normalized_values:
            history[name] = normalized_values
            steps[name] = normalized_steps
    return history, steps


def _normalize_metric_steps(
    raw_steps: dict[str, Any],
    history: dict[str, list[float]],
) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for metric, values in history.items():
        raw = raw_steps.get(metric)
        candidates = list(raw) if isinstance(raw, (list, tuple)) else []
        steps: list[float] = []
        for index in range(len(values)):
            value = _float_or_nan(candidates[index]) if index < len(candidates) else float(index)
            steps.append(float(value) if math.isfinite(value) else float(index))
        normalized[str(metric)] = steps
    return normalized


def _normalize_log_paths(raw: dict[str, Any]) -> dict[str, str]:
    logs: dict[str, str] = {}
    for key, value in raw.items():
        text = str(value or "").strip()
        if not text:
            continue
        logs[str(key)] = str(Path(text).resolve())
    return logs


def _numeric_metric_items(metrics: dict[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_numeric_metric_items(value, name))
            continue
        if isinstance(value, bool):
            continue
        number = _float_or_nan(value)
        if math.isfinite(number):
            rows.append((name, float(number)))
    return rows


def _training_run_record_for_artifact(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path)
    candidates = [path.with_suffix("") / TRAINING_RUN_FILENAME, path.parent / TRAINING_RUN_FILENAME]
    for candidate in candidates:
        try:
            data = _read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data:
            row = dict(data)
            row["path"] = str(candidate.resolve())
            row["relative_path"] = _relative_to_root(candidate)
            return row
    return {}


def navigation_task_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    task_root = Path(root or CONFIG_ROOT / "tasks")
    if not task_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(task_root.glob("*.yaml")):
        try:
            data = load_yaml_file(path)
        except Exception:
            continue
        if str(data.get("task_type", "")) != "navigation_region_v1":
            continue
        task_id = str(data.get("task_id") or path.stem)
        level = str(data.get("level") or data.get("map_id") or "")
        label = f"{task_id} ({level})" if level else task_id
        rows.append(
            {
                "id": task_id,
                "label": label,
                "path": str(path.resolve()),
                "relative_path": _relative_to_root(path),
                "summary": data,
            }
        )
    return sorted(rows, key=lambda row: (0 if _same_path(row["path"], DEFAULT_NAVIGATION_TASK_PATH) else 1, str(row["id"])))


def demo_config_entries() -> list[dict[str, Any]]:
    """Return runnable GUI demo presets.

    A demo config is intentionally higher level than task/model/planner fields:
    it is the one thing a new user selects before pressing Start.
    """

    return [
        {
            "id": DEFAULT_DEMO_CONFIG_ID,
            "label": "Johnson Valley standard demo",
            "description": "BeamNG Johnson Valley route-free model-control demo with the validated MLP dynamics model.",
            "task_path": str(DEFAULT_NAVIGATION_TASK_PATH.resolve()),
            "task_relative_path": _relative_to_root(DEFAULT_NAVIGATION_TASK_PATH),
            "world_model_config_id": DEFAULT_WORLD_MODEL_CONFIG_ID,
            "planner": "navigation_mpc",
            "evaluation_agent": "world_model_direct",
            "beamng_gfx": "vk",
        }
    ]


def resolve_demo_config(config_id: str = "") -> dict[str, Any]:
    requested = str(config_id or DEFAULT_DEMO_CONFIG_ID)
    for row in demo_config_entries():
        if str(row.get("id") or "") == requested:
            return dict(row)
    raise ValueError(f"Unknown demo config: {requested}")


def model_checkpoint_entries(root: str | Path | None = None) -> list[dict[str, Any]]:
    output_root = Path(root or ROOT / "outputs")
    if not output_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in output_root.glob("**/*_object.ckpt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        label = _checkpoint_display_label(path, output_root)
        rows.append(
            {
                "id": path.stem,
                "label": label,
                "path": str(path.resolve()),
                "relative_path": _relative_to_root(path),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if _same_path(row["path"], DEFAULT_LEWM_CHECKPOINT_PATH) else 1,
            -float(row.get("mtime", 0.0)),
            str(row["path"]),
        ),
    )


def world_model_config_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    config_path = Path(path or WORLD_MODEL_CONFIGS_PATH)
    rows: dict[str, dict[str, Any]] = {
        DEFAULT_WORLD_MODEL_CONFIG_ID: _world_model_config_row(
            {
                "id": DEFAULT_WORLD_MODEL_CONFIG_ID,
                "label": "Johnson Valley MLP support-route validated",
                "algorithm": "world_model_direct",
                "world_model": "mlp_dynamics",
                "model_path": str(DEFAULT_ROUTE_FREE_MODEL_PATH),
                "source_training_run_path": str(DEFAULT_ROUTE_FREE_MODEL_PATH.parent / "training_run.json"),
                "validation": {
                    "demo_ready": True,
                    "validation_source": str(DEFAULT_ROUTE_FREE_VALIDATION_SOURCE),
                    "goal_success": True,
                    "goal_reached": True,
                    "final_goal_reached": True,
                    "min_goal_distance": 11.897536452288634,
                    "final_goal_distance": 11.897536452288634,
                    "goal_radius": 12.0,
                    "model_controlled": True,
                    "route_free": True,
                    "route_free_direct": False,
                    "model_support_subgoals": True,
                    "model_support_field_subgoals": False,
                    "model_support_graph_subgoals": True,
                    "evaluation_route_mode": "route_free",
                    "route_waypoint_count": 0,
                    "collision_count": 0,
                    "max_collision_count": 0,
                    "distance_traveled": 193.88469705904217,
                    "stuck_recovery_count": 0,
                    "reverse_count": 0,
                    "support_route_count": 6,
                    "support_point_count": 2038,
                    "evaluation_local_subgoal_distance_m": 12.0,
                    "evaluation_allow_reverse_recovery": False,
                    "route_guided_goal_success": True,
                    "route_guided_final_goal_distance": 11.599028573749857,
                },
            }
        ),
        DEFAULT_LEWM_WORLD_MODEL_CONFIG_ID: _world_model_config_row(
            {
                "id": DEFAULT_LEWM_WORLD_MODEL_CONFIG_ID,
                "label": "Johnson Valley LE-WM validated",
                "algorithm": "stablewm_lewm",
                "world_model": "le_wm",
                "model_path": str(DEFAULT_LEWM_CHECKPOINT_PATH),
                "validation": {
                    "demo_ready": True,
                    "validation_source": "johnson_valley_standard_demo",
                    "goal_success": True,
                },
            }
        )
    }
    if config_path.exists():
        payload = _read_json(config_path)
        raw_rows: Any = payload.get("configs", payload) if isinstance(payload, dict) else payload
        if isinstance(raw_rows, list):
            for raw in raw_rows:
                if not isinstance(raw, dict):
                    continue
                row = _world_model_config_row(raw)
                rows[row["id"]] = row
    return list(rows.values())


def demo_ready_world_model_config_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    return [row for row in world_model_config_entries(path) if bool(row.get("demo_ready"))]


def _world_model_config_by_id(config_id: str) -> dict[str, Any]:
    requested = str(config_id or DEFAULT_WORLD_MODEL_CONFIG_ID)
    for row in world_model_config_entries():
        if str(row.get("id") or "") == requested:
            return dict(row)
    raise ValueError(f"Unknown world model config: {requested}")


def save_world_model_config(
    *,
    config_id: str,
    label: str,
    algorithm: str,
    world_model: str,
    model_path: str,
    source_training_run_path: str = "",
    validation: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    if not model_path.strip():
        raise ValueError("Model path is required.")
    row = _world_model_config_row(
        {
            "id": config_id or label,
            "label": label or config_id,
            "algorithm": algorithm or "stablewm_lewm",
            "world_model": world_model or "le_wm",
            "model_path": model_path,
            "source_training_run_path": source_training_run_path,
            "validation": validation or {},
        }
    )
    config_path = Path(path or WORLD_MODEL_CONFIGS_PATH)
    rows = [item for item in world_model_config_entries(config_path) if item["id"] != row["id"]]
    rows.append(row)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"configs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def import_world_model_config(
    source_path: str | Path,
    *,
    label: str = "",
    algorithm: str = "",
    world_model: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"World model path not found: {source}")
    inferred_world_model = _infer_world_model_type(source)
    resolved_world_model = world_model or inferred_world_model or "le_wm"
    resolved_algorithm = algorithm or ("world_model_direct" if resolved_world_model == "tiny_learned" else "stablewm_lewm")
    resolved_label = label or (source.parent.name if source.is_file() and source.name == "model.json" else source.stem if source.is_file() else source.name)
    return save_world_model_config(
        config_id=resolved_label,
        label=resolved_label,
        algorithm=resolved_algorithm,
        world_model=resolved_world_model,
        model_path=str(source),
        path=path,
    )


def register_training_run_artifact_as_world_model_config(
    training_run_path: str | Path,
    *,
    label: str = "",
    config_id: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Promote a completed training run artifact into a selectable model config."""

    record_path = Path(training_run_path)
    if record_path.is_dir():
        record_path = record_path / TRAINING_RUN_FILENAME
    if not record_path.exists():
        raise FileNotFoundError(f"Training run record not found: {record_path}")
    record = _read_json(record_path)
    if not isinstance(record, dict):
        raise ValueError(f"Invalid training run record: {record_path}")
    if str(record.get("status") or "").lower() not in {"completed", "ok", "success"}:
        raise ValueError("Only successful training runs can be registered as model configs.")

    artifact_path = _training_record_artifact_path(record)
    if not artifact_path:
        raise ValueError("Training run has no model artifact path.")
    artifact_type = str(record.get("artifact_type") or "").lower()
    if artifact_type == "hdf5":
        raise ValueError("HDF5 exports are dataset artifacts, not runnable world-model configs.")

    source = Path(artifact_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Training artifact not found: {source}")
    world_model = _infer_world_model_type(source)
    if not world_model:
        raise ValueError(f"Cannot infer runnable world-model type from artifact: {source}")
    algorithm = "world_model_direct" if world_model == "tiny_learned" else "stablewm_lewm"
    default_label = str(label or record.get("preset_label") or record.get("run_id") or source.stem or "Training Model")
    validation = _training_record_validation(record)
    row = save_world_model_config(
        config_id=config_id or default_label,
        label=default_label,
        algorithm=algorithm,
        world_model=world_model,
        model_path=str(source),
        source_training_run_path=str(record_path.resolve()),
        validation=validation,
        path=path,
    )
    _attach_world_model_config_to_training_run(str(record_path.resolve()), row)
    return row


def _register_region_self_supervised_world_model_config(
    *,
    request: RegionSelfSupervisedWorldModelRequest,
    task_id: str,
    model_dir: str,
    model_type: str,
    training_run_path: str,
    acceptance: dict[str, Any],
    quality_gate: dict[str, Any],
    model_metadata: dict[str, Any],
    experience_corridor_used: bool,
) -> dict[str, Any]:
    if not request.register_world_model_config:
        return {}
    if model_type != "tiny_learned" or not model_dir:
        return {}
    if not bool(acceptance.get("goal_success")):
        return {}
    evaluation_route_free = str(request.evaluation_route_mode or "route_free").strip().lower() in {"route_free", "none", "direct"}
    route_free_direct = bool(evaluation_route_free and not experience_corridor_used)
    validation = {
        "demo_ready": route_free_direct,
        "goal_success": bool(acceptance.get("goal_success")),
        "goal_reached": bool(acceptance.get("goal_reached")),
        "final_goal_reached": bool(acceptance.get("final_goal_reached")),
        "min_goal_distance": acceptance.get("min_goal_distance"),
        "final_goal_distance": acceptance.get("final_goal_distance"),
        "goal_radius": acceptance.get("goal_radius"),
        "model_controlled": bool(acceptance.get("model_controlled", True)),
        "route_free": bool(evaluation_route_free),
        "route_free_direct": route_free_direct,
        "experience_corridor": bool(experience_corridor_used),
        "evaluation_route_mode": "route_free",
        "route_waypoint_count": int(acceptance.get("route_waypoint_count", 0) or 0),
        "collision_count": acceptance.get("collision_count"),
        "max_collision_count": acceptance.get("max_collision_count"),
        "distance_traveled": acceptance.get("distance_traveled"),
        "stuck_recovery_count": acceptance.get("stuck_recovery_count"),
        "reverse_count": acceptance.get("reverse_count"),
        "quality_gate_passed": bool(quality_gate.get("passed")),
        "collection_progress_ratio": quality_gate.get("progress_ratio"),
        **_world_model_training_quality_metrics(model_metadata),
    }
    config = save_world_model_config(
        config_id=f"{task_id}_self_supervised_world_model",
        label=f"{task_id} self-supervised world model",
        algorithm="world_model_direct",
        world_model=model_type,
        model_path=model_dir,
        source_training_run_path=training_run_path,
        validation=validation,
        path=request.world_model_config_path or None,
    )
    _attach_world_model_config_to_training_run(training_run_path, config)
    return config


def _world_model_training_quality_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    quality_keys = (
        "train_rmse",
        "train_mse",
        "sequence_count",
        "sample_count",
        "transition_count",
        "train_sample_count",
        "validation_sample_count",
        "validation_fraction",
        "validation_mse",
        "validation_rmse",
        "segment_rmse",
        "segment_sample_count",
        "recorded_action_sample_count",
        "experience_route_point_count",
    )
    return {key: metadata.get(key) for key in quality_keys if key in metadata}


def _attach_world_model_config_to_training_run(training_run_path: str, config: dict[str, Any]) -> None:
    path = Path(training_run_path)
    try:
        record = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(record, dict) or not record:
        return
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    summary = dict(summary)
    summary["world_model_config"] = {
        "id": str(config.get("id") or ""),
        "label": str(config.get("label") or ""),
        "algorithm": str(config.get("algorithm") or ""),
        "world_model": str(config.get("world_model") or ""),
        "model_path": str(config.get("model_path") or ""),
        "validation": dict(config.get("validation") if isinstance(config.get("validation"), dict) else {}),
    }
    record["summary"] = summary
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _training_record_artifact_path(record: dict[str, Any]) -> str:
    for key in ("artifact_path", "model_path", "checkpoint_path", "model_dir", "output_dir"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    for key in ("artifact_path", "model_path", "checkpoint_path", "model_dir", "output_dir"):
        value = str(summary.get(key) or "").strip()
        if value:
            return value
    return ""


def _training_record_validation(record: dict[str, Any]) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    for source in (summary, metrics):
        for key in (
            "goal_success",
            "goal_reached",
            "collision_count",
            "final_goal_distance",
            "min_goal_distance",
            "average_speed",
            "loss",
            "rmse",
        ):
            if key in source and key not in validation:
                validation[key] = source[key]
    return validation


def training_config_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    config_path = Path(path or TRAINING_CONFIGS_PATH)
    rows: dict[str, dict[str, Any]] = {}
    source_paths = [TRAINING_CONFIGS_PATH]
    if config_path.resolve() != TRAINING_CONFIGS_PATH.resolve():
        source_paths.append(config_path)
    for source_path in source_paths:
        if not source_path.exists():
            continue
        payload = _read_json(source_path)
        raw_rows: Any = payload.get("configs", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            row = _training_config_row(raw)
            rows[row["id"]] = row
    return sorted(rows.values(), key=lambda row: str(row.get("label") or row.get("id") or ""))


def save_training_config(
    *,
    config_id: str = "",
    label: str = "",
    training_preset_id: str = "",
    dataset_root: str = "",
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
    output_path: str = "",
    parameters: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    row = _training_config_row(
        {
            "id": config_id or label,
            "label": label or config_id,
            "training_preset_id": training_preset_id or "tiny_world_model",
            "dataset_root": dataset_root,
            "adapter": adapter,
            "sequence_id": sequence_id,
            "split_path": split_path,
            "output_path": output_path,
            "parameters": parameters or {},
        }
    )
    config_path = Path(path or TRAINING_CONFIGS_PATH)
    rows = [item for item in training_config_entries(config_path) if item["id"] != row["id"]]
    rows.append(row)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"configs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def save_script_training_config(
    *,
    label: str,
    trainer_entrypoint: str,
    dataset_root: str,
    adapter: str = "",
    sequence_id: str = "",
    split_path: str = "",
    output_path: str = "",
    parameters: dict[str, Any] | None = None,
    parameter_schema: dict[str, Any] | None = None,
    arguments: list[str] | None = None,
    trainer_id: str = "",
    runtime: str = "python",
    input_spec: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    description: str = "",
    trainer_destination_root: str | Path | None = None,
    training_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a reusable trainer and config from a local training script.

    This is the simple GUI boundary for "bring a dataset, a script path, and
    parameters": users can avoid hand-writing trainer.yaml unless they need a
    custom command template.
    """

    values = dict(parameters or {})
    schema = dict(parameter_schema or _infer_trainer_parameter_schema(values))
    command_args = list(arguments or _default_trainer_arguments(schema))
    trainer_label = label or Path(trainer_entrypoint).stem.replace("_", " ").title()
    trainer = save_trainer_manifest(
        trainer_id=trainer_id or trainer_label,
        label=trainer_label,
        entrypoint=trainer_entrypoint,
        runtime=runtime,
        arguments=command_args,
        parameters=schema,
        input_spec=input_spec or {"dataset_format": "any_registered_adapter"},
        outputs=outputs or {"artifact_type": "checkpoint"},
        description=description,
        destination_root=trainer_destination_root,
    )
    config = save_training_config(
        config_id=label or trainer_label,
        label=label or trainer_label,
        training_preset_id=str(trainer["id"]),
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=sequence_id,
        split_path=split_path,
        output_path=output_path,
        parameters=values,
        path=training_config_path,
    )
    return {"trainer": trainer, "training_config": config}


def import_training_config(
    source_path: str | Path,
    *,
    path: str | Path | None = None,
    dataset_destination_root: str | Path | None = None,
    trainer_destination_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install a reusable training config bundle into the GUI catalog.

    The config may point at a dataset manifest and/or a trainer manifest using
    paths relative to the config file. Those referenced resources are imported
    first, then the saved training config points at the installed resources.
    """

    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Training config not found: {source}")
    data = _load_mapping_file(source)
    raw = dict(data.get("training_config") if isinstance(data.get("training_config"), dict) else data)

    dataset_root = str(raw.get("dataset_root") or "").strip()
    adapter = str(raw.get("adapter") or "").strip()
    dataset_manifest = str(raw.get("dataset_manifest") or "").strip()
    if dataset_manifest:
        dataset_row = import_dataset_manifest(
            _resolve_relative_path(source.parent, dataset_manifest),
            destination_root=dataset_destination_root or DATASET_MANIFEST_DIRS[0],
        )
        dataset_root = str(dataset_row.get("dataset_root") or dataset_root)
        adapter = str(dataset_row.get("adapter") or adapter or "manifest_dataset")

    preset_id = str(raw.get("training_preset_id") or raw.get("preset_id") or "").strip()
    trainer_manifest = str(raw.get("trainer_manifest") or raw.get("algorithm_manifest") or "").strip()
    if trainer_manifest:
        trainer_row = import_trainer_manifest(
            _resolve_relative_path(source.parent, trainer_manifest),
            destination_root=trainer_destination_root or TRAINER_MANIFEST_DIRS[0],
        )
        preset_id = str(trainer_row.get("id") or preset_id)
    inline_trainer = raw.get("trainer") if isinstance(raw.get("trainer"), dict) else {}
    if inline_trainer and not trainer_manifest:
        trainer_data = dict(inline_trainer)
        entrypoint = str(trainer_data.get("entrypoint") or "").strip()
        if not entrypoint:
            raise ValueError(f"Inline trainer has no entrypoint: {source}")
        trainer_row = save_trainer_manifest(
            trainer_id=str(trainer_data.get("trainer_id") or trainer_data.get("id") or source.stem),
            label=str(trainer_data.get("display_name") or trainer_data.get("label") or trainer_data.get("trainer_id") or source.stem),
            entrypoint=str(_resolve_relative_path(source.parent, entrypoint)),
            runtime=str(trainer_data.get("runtime") or "python"),
            arguments=list(trainer_data.get("arguments") or ["{dataset_root}", "--output", "{output_dir}"]),
            parameters=dict(trainer_data.get("parameters") if isinstance(trainer_data.get("parameters"), dict) else {}),
            input_spec=dict(trainer_data.get("input") if isinstance(trainer_data.get("input"), dict) else {}),
            outputs=dict(trainer_data.get("outputs") if isinstance(trainer_data.get("outputs"), dict) else {"artifact_type": "artifact"}),
            description=str(trainer_data.get("description") or ""),
            destination_root=trainer_destination_root or TRAINER_MANIFEST_DIRS[0],
        )
        preset_id = str(trainer_row.get("id") or preset_id)

    row = save_training_config(
        config_id=str(raw.get("id") or raw.get("config_id") or raw.get("label") or source.stem),
        label=str(raw.get("label") or raw.get("display_name") or source.stem),
        training_preset_id=preset_id or "tiny_world_model",
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=str(raw.get("sequence_id") or ""),
        output_path=str(raw.get("output_path") or raw.get("model_path") or raw.get("hdf5_path") or ""),
        parameters=dict(raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}),
        path=path,
    )
    return row


def beamng_status() -> dict[str, Any]:
    status = default_backend_registry().status("beamng")
    return asdict(status) if is_dataclass(status) else dict(status)


def inspect_dataset(dataset_root: str, adapter: str = "", sequence_id: str = "") -> dict[str, Any]:
    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    sequences = resolved_adapter.list_sequences(dataset_root)
    if not sequences:
        raise ValueError("No dataset sequences found.")
    selected = sequence_id or sequences[0]
    sequence = resolved_adapter.load_sequence(dataset_root, selected)
    analysis = analyze_dataset_sequences(
        [sequence],
        dataset_root=dataset_root,
        options=DatasetAnalysisOptions(max_asset_checks=100),
    )
    asset_counts: dict[str, int] = {}
    for frame in sequence.frames:
        for name in frame.available_assets():
            asset_counts[name] = asset_counts.get(name, 0) + 1
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "adapter": resolved_adapter.name,
        "sequences": sequences,
        "selected_sequence": selected,
        "dataset_id": sequence.dataset_id,
        "dataset_type": sequence.dataset_type,
        "frame_count": len(sequence.frames),
        "asset_counts": asset_counts,
        "metadata": sequence.metadata,
        "details": {
            "analysis_scope": "selected_sequence",
            "dataset_sequence_count": len(sequences),
            "selected_sequence_id": selected,
            "selected_sequence_frame_count": len(sequence.frames),
            "modalities": analysis["modalities"],
            "resolutions": analysis["resolutions"],
            "time_start": analysis["time_start"],
            "time_end": analysis["time_end"],
            "duration_sec": analysis["duration_sec"],
            "referenced_disk_usage_bytes": analysis["referenced_disk_usage_bytes"],
            "dataset_disk_usage_bytes": analysis["dataset_disk_usage_bytes"],
            "dataset_file_count": analysis["dataset_file_count"],
            "disk_usage_truncated": analysis["disk_usage_truncated"],
        },
        "quality": analysis,
    }


def analyze_dataset_quality(
    dataset_root: str,
    adapter: str = "",
    sequence_ids: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    max_asset_checks: int = 0,
) -> dict[str, Any]:
    """Run a full dataset quality scan and persist JSON/Markdown reports."""

    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    available = resolved_adapter.list_sequences(dataset_root)
    selected = sequence_ids or available
    unknown = [sequence_id for sequence_id in selected if sequence_id not in available]
    if unknown:
        raise ValueError(f"Unknown dataset sequences: {', '.join(unknown)}")
    sequences = [resolved_adapter.load_sequence(dataset_root, sequence_id) for sequence_id in selected]
    analysis = analyze_dataset_sequences(
        sequences,
        dataset_root=dataset_root,
        options=DatasetAnalysisOptions(max_asset_checks=max_asset_checks),
    )
    report_dir = Path(
        output_dir
        or ROOT / "outputs" / "dataset_reports" / f"{_safe_name(Path(dataset_root).name)}_{_timestamp()}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "dataset_quality_report.json"
    markdown_path = report_dir / "dataset_quality_report.md"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_dataset_quality_markdown(analysis), encoding="utf-8")
    return {
        "status": analysis["status"],
        "training_ready": analysis["training_ready"],
        "analysis": analysis,
        "report_json_path": str(json_path.resolve()),
        "report_markdown_path": str(markdown_path.resolve()),
    }


def create_dataset_split_definition(
    dataset_root: str,
    adapter: str = "",
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create and save a deterministic train/validation/test split."""

    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    sequence_ids = resolved_adapter.list_sequences(dataset_root)
    sequences = [resolved_adapter.load_sequence(dataset_root, sequence_id) for sequence_id in sequence_ids]
    payload = build_dataset_split(
        sequences,
        dataset_root=dataset_root,
        adapter=resolved_adapter.name,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    target = Path(
        output_path
        or ROOT / "outputs" / "dataset_splits" / f"{_safe_name(Path(dataset_root).name)}_split.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["path"] = str(target.resolve())
    return payload


def preview_dataset_frame(
    dataset_root: str,
    adapter: str = "",
    sequence_id: str = "",
    *,
    frame_index: int = 0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    sequences = resolved_adapter.list_sequences(dataset_root)
    if not sequences:
        raise ValueError("No dataset sequences found.")
    selected = sequence_id or sequences[0]
    sequence = resolved_adapter.load_sequence(dataset_root, selected)
    if not sequence.frames:
        raise ValueError("Selected sequence has no frames.")
    index = min(max(0, int(frame_index)), len(sequence.frames) - 1)
    frame = sequence.frames[index]
    out = Path(output_dir or ROOT / "outputs" / "gui_previews" / _safe_name(selected))
    out.mkdir(parents=True, exist_ok=True)

    previews: dict[str, str] = {}
    for name, asset_path in frame.available_assets().items():
        if name not in {"front_rgb", "depth", "label", "lidar_points", "local_bev", "terrain_map"}:
            continue
        preview_path = _write_preview_image(asset_path, out / f"{index:06d}_{name}.png", modality=name)
        if preview_path is not None:
            previews[name] = str(preview_path.resolve())

    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "adapter": resolved_adapter.name,
        "sequence_id": selected,
        "frame_index": index,
        "frame_count": len(sequence.frames),
        "frame_id": frame.frame_id,
        "assets": frame.available_assets(),
        "previews": previews,
        "metadata": frame.metadata,
    }


def _dataset_quality_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Dataset Quality Report",
        "",
        f"- Status: {analysis.get('status', NAN_TEXT)}",
        f"- Training ready: {analysis.get('training_ready', False)}",
        f"- Sequences: {analysis.get('sequence_count', 0)}",
        f"- Samples: {analysis.get('sample_count', 0)}",
        f"- Modalities: {', '.join(analysis.get('modalities', [])) or NAN_TEXT}",
        f"- Available modalities: {', '.join(analysis.get('available_modalities', [])) or NAN_TEXT}",
        f"- Asset check mode: {analysis.get('asset_check_mode', NAN_TEXT)} ({analysis.get('checked_asset_count', 0)}/{analysis.get('available_asset_count', 0)})",
        f"- Referenced disk usage: {analysis.get('referenced_disk_usage_bytes', 0)} bytes",
        f"- Dataset disk usage: {analysis.get('dataset_disk_usage_bytes', 0)} bytes",
        f"- Errors: {analysis.get('error_count', 0)}",
        f"- Warnings: {analysis.get('warning_count', 0)}",
        "",
        "## Sequences",
        "",
        "| Sequence | Frames | Modalities | Frame gaps | Timestamp issues |",
        "|---|---:|---|---:|---:|",
    ]
    for row in analysis.get("sequences", []):
        lines.append(
            "| {sequence_id} | {frame_count} | {modalities} | {frame_id_gap_count} | {timestamp_issue_count} |".format(
                sequence_id=row.get("sequence_id", ""),
                frame_count=row.get("frame_count", 0),
                modalities=", ".join(row.get("modalities", [])),
                frame_id_gap_count=row.get("frame_id_gap_count", 0),
                timestamp_issue_count=row.get("timestamp_issue_count", 0),
            )
        )
    lines.extend(["", "## Issues", ""])
    issues = analysis.get("issues", [])
    if not issues:
        lines.append("No issues found.")
    else:
        for issue in issues:
            lines.append(
                f"- [{str(issue.get('severity', 'warning')).upper()}] {issue.get('code', '')}: {issue.get('message', '')}"
            )
    return "\n".join(lines) + "\n"


def run_episode_from_request(request: RunRequest) -> dict[str, Any]:
    result = run_episode(
        backend_name=request.backend,
        scenario=scenario_for_request(request),
        agent_name=request.agent,
        seed=request.seed,
        max_steps=request.max_steps,
        record=request.record,
        output_root=DEFAULT_OUTPUT_ROOT,
        record_arrays=request.record_arrays,
        backend_options=backend_options(request),
        agent_options=agent_options(request),
    )
    return result.to_dict()


def train_tiny_world_model(
    dataset_root: str,
    output_dir: str,
    *,
    adapter: str = "",
    sequence_id: str = "",
    ridge: float = 1e-4,
) -> dict[str, Any]:
    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    sequence_ids = [sequence_id] if sequence_id else resolved_adapter.list_sequences(dataset_root)[:1]
    sequences = [resolved_adapter.load_sequence(dataset_root, item) for item in sequence_ids]
    model = TinyLearnedWorldModel.fit(sequences, ridge=ridge)
    model.metadata.update(
        {
            "dataset_root": str(Path(dataset_root).resolve()),
            "adapter": resolved_adapter.name,
            "sequence_ids": sequence_ids,
        }
    )
    metadata_path = model.save(output_dir)
    record = write_training_run_record(
        output_dir,
        preset_id="tiny_world_model",
        status="completed",
        dataset_root=str(Path(dataset_root).resolve()),
        adapter=resolved_adapter.name,
        sequence_id=", ".join(sequence_ids),
        artifact_path=str(Path(output_dir).resolve()),
        artifact_type="world_model",
        metrics=model.metadata,
        history={
            "train_rmse": [model.metadata.get("train_rmse")],
            "train_mse": [model.metadata.get("train_mse")],
            "sample_count": [model.metadata.get("sample_count")],
        },
        parameters={"ridge": ridge},
        summary={"model_type": model.model_type, "model_path": str(metadata_path)},
    )
    return {
        "model_type": model.model_type,
        "model_path": str(metadata_path),
        "output_dir": str(Path(output_dir).resolve()),
        "metrics": model.metadata,
        "training_run_path": record["path"],
    }


def _normalize_lightweight_region_world_model_type(world_model_type: str) -> str:
    normalized = str(world_model_type or "").strip().lower().replace("-", "_")
    aliases = {"tiny": "tiny_learned", "mlp": "mlp_dynamics"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in LIGHTWEIGHT_REGION_WORLD_MODELS:
        supported = ", ".join(LIGHTWEIGHT_REGION_WORLD_MODELS)
        raise ValueError(f"BeamNG region training supports world_model_type in: {supported}.")
    return normalized


def _fit_lightweight_region_world_model(
    world_model_type: str,
    sequences: list[DatasetSequence],
) -> TinyLearnedWorldModel | MLPDynamicsWorldModel:
    normalized = _normalize_lightweight_region_world_model_type(world_model_type)
    if normalized == "tiny_learned":
        return TinyLearnedWorldModel.fit(sequences)
    return MLPDynamicsWorldModel.fit(sequences)


def export_lewm_hdf5(
    dataset_root: str,
    output_hdf5: str,
    *,
    adapter: str = "",
    sequence_id: str = "",
    image_size: int = 64,
) -> dict[str, Any]:
    if not dataset_root:
        raise ValueError("Dataset root is required.")
    if not output_hdf5:
        raise ValueError("Output HDF5 path is required.")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_lewm_hdf5.py"),
        dataset_root,
        output_hdf5,
        "--image-size",
        str(image_size),
    ]
    if adapter:
        command.extend(["--adapter", adapter])
    if sequence_id:
        command.extend(["--sequence-id", sequence_id])
    payload = _run_json_command(command)
    artifact_path = str(payload.get("output_hdf5") or output_hdf5)
    record = write_training_run_record(
        Path(artifact_path).with_suffix(""),
        preset_id="stablewm_hdf5",
        status="completed",
        dataset_root=str(Path(dataset_root).resolve()),
        adapter=adapter,
        sequence_id=sequence_id,
        artifact_path=str(Path(artifact_path).resolve()),
        artifact_type="hdf5",
        metrics=dict(payload),
        history={"total_frames": [payload.get("total_frames") or payload.get("frame_count")]},
        parameters={"image_size": image_size},
    )
    payload["training_run_path"] = record["path"]
    return payload


def export_episodes_hdf5(episode_root: str, output_hdf5: str, *, actions_from_state: bool = False) -> dict[str, Any]:
    if not episode_root:
        raise ValueError("Episode root is required.")
    if not output_hdf5:
        raise ValueError("Output HDF5 path is required.")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_episodes_hdf5.py"),
        episode_root,
        output_hdf5,
    ]
    if actions_from_state:
        command.append("--actions-from-state")
    return _run_json_command(command)


def train_lewm_cost_model(input_hdf5: str, output_dir: str) -> dict[str, Any]:
    if not input_hdf5:
        raise ValueError("Input HDF5 path is required.")
    if not output_dir:
        raise ValueError("Output model directory is required.")
    payload = _run_json_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_lewm_cost_model.py"),
            input_hdf5,
            "--output",
            output_dir,
        ]
    )
    artifact_path = str(payload.get("checkpoint_path") or payload.get("model_path") or output_dir)
    source_record = _training_run_record_for_artifact(input_hdf5)
    summary: dict[str, Any] = {}
    if source_record:
        summary["source_training_run_path"] = source_record.get("path", "")
    record = write_training_run_record(
        payload.get("output_dir") or output_dir,
        preset_id="lewm_cost_model",
        status="completed",
        dataset_root=str(source_record.get("dataset_root") or "") if source_record else "",
        adapter=str(source_record.get("adapter") or "") if source_record else "",
        sequence_id=str(source_record.get("sequence_id") or "") if source_record else "",
        artifact_path=str(Path(artifact_path).resolve()),
        artifact_type="checkpoint",
        metrics=dict(payload),
        history=payload.get("history") if isinstance(payload.get("history"), dict) else {},
        parameters={"input_hdf5": input_hdf5},
        summary=summary,
    )
    payload["training_run_path"] = record["path"]
    return payload


def make_algorithm_adapter(name: str) -> Any:
    return default_algorithm_registry().create(name)


def run_orfd_lewm_pipeline(request: PipelineRequest) -> dict[str, Any]:
    if not request.dataset_root:
        raise ValueError("Dataset root is required.")
    selected_info = inspect_dataset(request.dataset_root, request.adapter, request.sequence_id)
    sequence_id = str(selected_info["selected_sequence"])
    stamp = time.strftime("%Y%m%dT%H%M%S")
    hdf5_path = request.hdf5_path or str(ROOT / "outputs" / "stablewm" / f"gui_orfd_{_safe_name(sequence_id)}_{stamp}.h5")
    model_dir = request.model_dir or str(ROOT / "outputs" / "models" / f"gui_lewm_{_safe_name(sequence_id)}_{stamp}")

    export = export_lewm_hdf5(
        request.dataset_root,
        hdf5_path,
        adapter=request.adapter,
        sequence_id=sequence_id,
        image_size=request.image_size,
    )
    training = train_lewm_cost_model(str(export["output_hdf5"]), model_dir)
    replay_request = RunRequest(
        backend="dataset_replay",
        scenario=f"dataset_{Path(request.dataset_root).name}",
        agent="world_model",
        seed=request.seed,
        max_steps=request.max_steps,
        record=True,
        world_model_type="le_wm",
        world_model_path=model_dir,
        planner="le_wm_cem",
        planner_horizon=request.planner_horizon,
        planner_samples=request.planner_samples,
        planner_iterations=request.planner_iterations,
        dataset_root=request.dataset_root,
        sequence_id=sequence_id,
        adapter=request.adapter,
        load_assets=False,
    )
    replay = run_episode_from_request(replay_request)

    beamng: dict[str, Any] | None = None
    if request.run_beamng:
        beamng_request = RunRequest(
            backend="beamng",
            scenario=request.beamng_scenario,
            agent="world_model",
            seed=request.seed,
            max_steps=max(1, request.max_steps),
            record=True,
            world_model_type="le_wm",
            world_model_path=model_dir,
            planner="le_wm_cem",
            planner_horizon=request.planner_horizon,
            planner_samples=request.planner_samples,
            planner_iterations=request.planner_iterations,
        )
        beamng = run_episode_from_request(beamng_request)

    return {
        "status": "completed",
        "dataset": selected_info,
        "hdf5": export,
        "training": training,
        "dataset_replay": replay,
        "beamng": beamng,
        "output_dir": model_dir,
        "model_dir": model_dir,
        "hdf5_path": str(export["output_hdf5"]),
    }


def build_visible_beamng_demo_request(request: VisibleBeamNGDemoRequest) -> RunRequest:
    return RunRequest(
        backend="beamng",
        scenario=request.scenario,
        agent="route_world_model",
        seed=request.seed,
        max_steps=request.max_steps,
        record=request.record,
        world_model_type=request.world_model_type,
        world_model_path=request.world_model_path,
        planner=request.planner,
        planner_horizon=4,
        planner_samples=16,
        planner_iterations=2,
        dataset_root=request.dataset_root,
        sequence_id=request.sequence_id,
        adapter=request.adapter,
    )


def run_visible_beamng_demo(request: VisibleBeamNGDemoRequest) -> dict[str, Any]:
    run_request = build_visible_beamng_demo_request(request)
    result = run_episode(
        backend_name=run_request.backend,
        scenario=scenario_for_request(run_request),
        agent_name=run_request.agent,
        seed=run_request.seed,
        max_steps=run_request.max_steps,
        record=run_request.record,
        output_root=DEFAULT_OUTPUT_ROOT,
        record_arrays=run_request.record_arrays,
        backend_options={**backend_options(run_request), "connection": BeamNGConnectionConfig(gfx=request.beamng_gfx or None)},
        agent_options=agent_options(run_request),
        vehicle=request.vehicle,
        pre_run_hold_sec=request.pre_run_hold_sec,
        step_delay_sec=request.step_delay_sec,
        post_run_hold_sec=request.post_run_hold_sec,
        close_backend=request.close_beamng,
    )
    payload = result.to_dict()
    payload["visible_demo"] = {
        "dataset_root": request.dataset_root or None,
        "adapter": request.adapter or None,
        "sequence_id": request.sequence_id or None,
        "world_model_type": request.world_model_type,
        "world_model_path": request.world_model_path or None,
        "planner": request.planner or None,
        "scenario": request.scenario,
        "vehicle": request.vehicle,
        "beamng_gfx": request.beamng_gfx,
    }
    return payload


def run_beamng_map_lewm_closed_loop(request: BeamNGMapLeWMClosedLoopRequest) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "beamng_map_lewm" / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection = run_visible_beamng_demo(
        VisibleBeamNGDemoRequest(
            world_model_type="simple_kinematic",
            world_model_path="",
            planner="",
            scenario=request.scenario,
            vehicle=request.vehicle,
            max_steps=max(1, int(request.collect_steps)),
            seed=request.seed,
            record=True,
            pre_run_hold_sec=request.pre_run_hold_sec,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=0.0,
            close_beamng=True,
            beamng_gfx=request.beamng_gfx,
        )
    )
    episode_path = collection.get("episode_path")
    if not episode_path:
        raise RuntimeError("BeamNG collection did not produce an episode path.")

    algorithm = make_algorithm_adapter(request.algorithm)
    hdf5_path = output_dir / "beamng_map_lewm.h5"
    prep = algorithm.prepare_data(DataPrepRequest(episode_root=str(episode_path), output_path=str(hdf5_path), actions_from_state=True))
    hdf5 = {"output_hdf5": prep.output_path, **prep.metadata}

    model_dir = output_dir / "model"
    trained = algorithm.train(TrainRequest(input_path=str(hdf5["output_hdf5"]), output_dir=str(model_dir)))
    training = {"output_dir": trained.output_dir, "checkpoint_path": trained.checkpoint_path, **trained.metadata}
    model_path = str(training.get("output_dir") or model_dir)

    evaluation = run_visible_beamng_demo(
        VisibleBeamNGDemoRequest(
            world_model_type="le_wm",
            world_model_path=model_path,
            planner=request.planner,
            scenario=request.scenario,
            vehicle=request.vehicle,
            max_steps=max(1, int(request.eval_steps)),
            seed=request.seed,
            record=True,
            pre_run_hold_sec=0.0,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=request.post_run_hold_sec,
            close_beamng=request.close_beamng,
            beamng_gfx=request.beamng_gfx,
        )
    )

    payload: dict[str, Any] = {
        "status": "completed",
        "algorithm": algorithm.algorithm_id,
        "output_dir": str(output_dir.resolve()),
        "hdf5_path": str(hdf5.get("output_hdf5", hdf5_path)),
        "model_dir": model_path,
        "collection": collection,
        "hdf5": hdf5,
        "training": training,
        "evaluation": evaluation,
        "acceptance": {
            "collection_horizontal_distance": metric_value(collection.get("metrics", {}), "horizontal_distance_traveled"),
            "evaluation_horizontal_distance": metric_value(evaluation.get("metrics", {}), "horizontal_distance_traveled"),
            "world_model_type": "le_wm",
            "planner": request.planner,
        },
    }
    summary_path = output_dir / "closed_loop_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def run_region_navigation_closed_loop(request: RegionNavigationClosedLoopRequest) -> dict[str, Any]:
    task = load_navigation_region_task(request.task_path)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "region_navigation" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_scenario = task.to_beamng_scenario(mode="collection")
    evaluation_scenario = task.to_beamng_scenario(mode="evaluation")
    hdf5_path = output_dir / f"{_safe_name(task.task_id)}.h5"
    algorithm = make_algorithm_adapter(request.algorithm)
    existing_model_path = str(request.algorithm_model_path or "").strip()

    if existing_model_path:
        collection = {"status": "skipped", "reason": "algorithm_model_path provided"}
        hdf5 = {"status": "skipped", "reason": "algorithm_model_path provided"}
        training = {"status": "skipped", "output_dir": existing_model_path, "checkpoint_path": existing_model_path}
        model_path = existing_model_path
    else:
        collection = _run_region_beamng_episode_with_reconnect_retry(
            scenario=collection_scenario,
            vehicle=request.vehicle,
            max_steps=min(max(1, int(request.collect_steps)), task.max_steps),
            seed=request.seed,
            world_model_type="simple_kinematic",
            world_model_path="",
            planner="",
            planner_horizon=request.planner_horizon,
            planner_samples=request.planner_samples,
            planner_iterations=request.planner_iterations,
            record=True,
            beamng_gfx=request.beamng_gfx,
            pre_run_hold_sec=request.pre_run_hold_sec,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=0.0,
            close_beamng=True,
        )
        episode_path = collection.get("episode_path")
        if not episode_path:
            raise RuntimeError("Region navigation collection did not produce an episode path.")

        prep = algorithm.prepare_data(DataPrepRequest(episode_root=str(episode_path), output_path=str(hdf5_path), actions_from_state=True))
        hdf5 = {"output_hdf5": prep.output_path, **prep.metadata}

        model_dir = output_dir / "model"
        trained = algorithm.train(TrainRequest(input_path=str(hdf5["output_hdf5"]), output_dir=str(model_dir)))
        training = {"output_dir": trained.output_dir, "checkpoint_path": trained.checkpoint_path, **trained.metadata}
        model_path = str(training.get("output_dir") or model_dir)

    evaluation = _run_region_beamng_episode_with_reconnect_retry(
        scenario=evaluation_scenario,
        vehicle=request.vehicle,
        max_steps=min(max(1, int(request.eval_steps)), task.max_steps),
        seed=request.seed,
        agent_name=request.evaluation_agent,
        world_model_type="simple_kinematic" if request.evaluation_agent == "model_mpc" else "le_wm",
        world_model_path="" if request.evaluation_agent == "model_mpc" else model_path,
        algorithm_name=algorithm.algorithm_id if request.evaluation_agent == "model_mpc" else "",
        algorithm_model_path=model_path if request.evaluation_agent == "model_mpc" else "",
        planner="navigation_mpc" if request.evaluation_agent == "model_mpc" else request.planner,
        planner_horizon=request.planner_horizon,
        planner_samples=request.planner_samples,
        planner_iterations=request.planner_iterations,
        record=True,
        beamng_gfx=request.beamng_gfx,
        pre_run_hold_sec=0.0,
        step_delay_sec=request.step_delay_sec,
        post_run_hold_sec=request.post_run_hold_sec,
        close_beamng=request.close_beamng,
    )
    acceptance = _navigation_acceptance(evaluation, task)
    region_navigation = evaluation.get("region_navigation", {}) if isinstance(evaluation.get("region_navigation"), dict) else {}
    payload: dict[str, Any] = {
        "status": "completed",
        "algorithm": algorithm.algorithm_id,
        "task": task.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "hdf5_path": str(hdf5.get("output_hdf5", hdf5_path)),
        "model_dir": model_path,
        "collection": collection,
        "hdf5": hdf5,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "region_navigation": region_navigation,
    }
    summary_path = output_dir / "region_navigation_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def run_demo_acceptance(request: DemoAcceptanceRequest) -> dict[str, Any]:
    demo = resolve_demo_config(request.demo_config_id)
    runs = min(3, max(1, int(request.runs)))
    world_model_config = _world_model_config_by_id(str(demo.get("world_model_config_id") or DEFAULT_WORLD_MODEL_CONFIG_ID))
    if not bool(world_model_config.get("demo_ready")):
        raise ValueError(f"World model config is not demo-ready: {world_model_config.get('id')}")
    algorithm = str(world_model_config.get("algorithm") or "stablewm_lewm")
    world_model = str(world_model_config.get("world_model") or "le_wm")
    model_path = str(world_model_config.get("model_path") or "")
    validation = world_model_config.get("validation") if isinstance(world_model_config.get("validation"), dict) else {}
    task_path = str(demo.get("task_path") or DEFAULT_NAVIGATION_TASK_PATH)
    planner = str(demo.get("planner") or "navigation_mpc")
    run_rows: list[dict[str, Any]] = []

    for index in range(runs):
        seed = int(request.seed) + index
        if algorithm == "world_model_direct" or world_model == "tiny_learned":
            payload = run_region_world_model_evaluation(
                RegionWorldModelEvaluationRequest(
                    task_path=task_path,
                    world_model_type=world_model,
                    world_model_path=model_path,
                    eval_steps=max(1, int(request.max_steps)),
                    seed=seed,
                    planner=planner,
                    planner_horizon=request.planner_horizon,
                    planner_samples=request.planner_samples,
                    planner_iterations=request.planner_iterations,
                    include_route_guided_baseline=True,
                    planner_goal_weight=_validation_float_or_none(validation, "planner_goal_weight"),
                    planner_progress_weight=_validation_float_or_none(validation, "planner_progress_weight"),
                    planner_risk_weight=_validation_float_or_none(validation, "planner_risk_weight"),
                    planner_heading_weight=_validation_float_or_none(validation, "planner_heading_weight"),
                    evaluation_allow_reverse_recovery=bool(validation.get("evaluation_allow_reverse_recovery")),
                    evaluation_reverse_recovery_after_steps=_coerce_int(
                        validation.get("evaluation_reverse_recovery_after_steps"),
                        default=96,
                    ),
                    evaluation_local_subgoal_distance_m=_validation_float_or_default(
                        validation,
                        "evaluation_local_subgoal_distance_m",
                        12.0,
                    ),
                    use_experience_corridor=bool(validation.get("experience_corridor")),
                    evaluation_use_model_support_subgoals=bool(validation.get("model_support_subgoals")),
                    evaluation_use_model_support_field_subgoals=bool(validation.get("model_support_field_subgoals")),
                    evaluation_use_model_support_graph_subgoals=bool(validation.get("model_support_graph_subgoals")),
                    beamng_gfx=request.beamng_gfx or str(demo.get("beamng_gfx") or "vk"),
                    close_beamng=request.close_beamng,
                    step_delay_sec=request.step_delay_sec,
                    pre_run_hold_sec=request.pre_run_hold_sec,
                    post_run_hold_sec=request.post_run_hold_sec,
                )
            )
        else:
            payload = run_region_navigation_closed_loop(
                RegionNavigationClosedLoopRequest(
                    task_path=task_path,
                    algorithm=algorithm,
                    algorithm_model_path=model_path,
                    collect_steps=max(1, int(request.max_steps)),
                    eval_steps=max(1, int(request.max_steps)),
                    seed=seed,
                    planner=planner,
                    planner_horizon=request.planner_horizon,
                    planner_samples=request.planner_samples,
                    planner_iterations=request.planner_iterations,
                    evaluation_agent=str(demo.get("evaluation_agent") or "model_mpc"),
                    beamng_gfx=request.beamng_gfx or str(demo.get("beamng_gfx") or "vk"),
                    close_beamng=request.close_beamng,
                    step_delay_sec=request.step_delay_sec,
                    pre_run_hold_sec=request.pre_run_hold_sec,
                    post_run_hold_sec=request.post_run_hold_sec,
                )
            )
        run_rows.append(_demo_acceptance_run_summary(index + 1, seed, payload))

    accepted = bool(run_rows) and all(bool(row.get("goal_success")) for row in run_rows)
    summary = _demo_acceptance_summary(run_rows)
    return {
        "status": "accepted" if accepted else "failed",
        "accepted": accepted,
        "run_count": len(run_rows),
        "all_goal_success": accepted,
        "demo_config": demo,
        "world_model_config": world_model_config,
        "summary": summary,
        "runs": run_rows,
    }


def collect_region_training_data(request: RegionTrainingDataCollectionRequest) -> dict[str, Any]:
    task = load_navigation_region_task(request.task_path)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "beamng_region_training_data" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_strategy = str(request.collection_strategy or "region_explorer").strip().lower()
    collection_scenario = _collection_region_scenario(task, strategy=collection_strategy)
    route_aware = collection_strategy in {"route_aware", "route-aware", "curriculum", "route_curriculum"}
    agent_options = {
        "goal_bias_interval": max(0, int(request.collection_goal_bias_interval)),
        "goal_corridor_interval": max(0, int(request.collection_goal_corridor_interval)),
        "goal_corridor_lateral_m": max(0.0, float(request.collection_goal_corridor_lateral_m)),
        "coverage_grid_size": max(0, int(request.collection_coverage_grid_size)),
        "coverage_target_interval": max(0, int(request.collection_coverage_target_interval)),
        "max_target_steps": max(1, int(request.collection_max_target_steps)),
    }
    if route_aware:
        agent_options.update(
            {
                "route_target_interval": max(1, int(request.collection_route_target_interval or 1)),
                "route_lateral_m": max(0.0, float(request.collection_route_lateral_m)),
            }
        )

    collections: list[dict[str, Any]] = []
    collection_acceptances: list[dict[str, Any]] = []
    collection_distances: list[float] = []
    episode_paths: list[str] = []
    rollout_count = max(1, int(request.collect_rollouts))
    for rollout_index in range(rollout_count):
        rollout_scenario = _collection_rollout_scenario(
            task,
            collection_scenario,
            strategy=collection_strategy,
            rollout_index=rollout_index,
            rollout_count=rollout_count,
            multi_start=bool(request.collection_multi_start),
            lateral_m=float(request.collection_multi_start_lateral_m),
            seed=int(request.seed) + rollout_index,
        )
        collection = _run_region_beamng_episode_with_reconnect_retry(
            scenario=rollout_scenario,
            vehicle=request.vehicle,
            max_steps=min(max(1, int(request.collect_steps)), task.max_steps),
            seed=int(request.seed) + rollout_index,
            agent_name="region_explorer",
            agent_options=agent_options,
            world_model_type="simple_kinematic",
            world_model_path="",
            planner="",
            planner_horizon=1,
            planner_samples=4,
            planner_iterations=1,
            record=True,
            beamng_gfx=request.beamng_gfx,
            pre_run_hold_sec=request.pre_run_hold_sec if rollout_index == 0 else 0.0,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=request.post_run_hold_sec,
            close_beamng=request.close_beamng,
        )
        episode_path = collection.get("episode_path")
        if not episode_path:
            raise RuntimeError("BeamNG region data collection did not produce an episode path.")
        collections.append(collection)
        episode_paths.append(str(Path(episode_path).resolve()))
        collection_acceptances.append(_navigation_acceptance(collection, task))
        collection_metrics = collection.get("metrics", {}) if isinstance(collection.get("metrics"), dict) else {}
        collection_distance = _float_or_nan(collection_metrics.get("horizontal_distance_traveled"))
        if math.isfinite(collection_distance):
            collection_distances.append(collection_distance)

    best_acceptance = min(collection_acceptances, key=lambda row: _finite_or_inf(row.get("min_goal_distance")))
    collection_distance_total = float(sum(collection_distances)) if collection_distances else math.nan
    coverage = _collection_coverage_metrics(
        task,
        episode_paths,
        grid_size=max(2, int(request.collection_coverage_grid_size)),
    )
    route_metrics = _collection_route_metrics(task, episode_paths)
    quality_gate = _collection_quality_gate(
        task,
        best_acceptance,
        min_progress_ratio=request.min_collection_goal_progress_ratio,
        route_coverage_ratio=route_metrics["route_coverage_ratio"],
        min_route_coverage_ratio=request.min_route_coverage_ratio,
        goal_zone_coverage=route_metrics["goal_zone_coverage"],
        min_goal_zone_coverage=request.min_goal_zone_coverage,
        max_collection_min_goal_distance_m=request.max_collection_min_goal_distance_m,
        unique_region_cells=coverage["cell_count"],
        min_unique_region_cells=request.min_unique_region_cells,
    )
    metrics = {
        "collection_rollout_count": rollout_count,
        "collection_distance_traveled": collection_distance_total,
        "collection_goal_reached": bool(best_acceptance.get("goal_reached")),
        "collection_min_goal_distance": best_acceptance.get("min_goal_distance"),
        "collection_final_goal_distance": best_acceptance.get("final_goal_distance"),
        "collection_collision_count": best_acceptance.get("collision_count"),
        "collection_progress_ratio": quality_gate.get("progress_ratio"),
        "required_collection_progress_ratio": quality_gate.get("required_progress_ratio"),
        "collection_coverage_cell_count": coverage["cell_count"],
        "collection_coverage_total_cells": coverage["total_cells"],
        "collection_coverage_ratio": coverage["ratio"],
        "unique_region_cells": coverage["cell_count"],
        "route_coverage_ratio": route_metrics["route_coverage_ratio"],
        "route_covered_waypoint_count": route_metrics["route_covered_waypoint_count"],
        "route_waypoint_count": route_metrics["route_waypoint_count"],
        "goal_zone_coverage": route_metrics["goal_zone_coverage"],
    }
    collection_status = "completed" if bool(quality_gate.get("passed")) else "collection_insufficient"
    payload: dict[str, Any] = {
        "status": collection_status,
        "task": task.to_dict(),
        "task_path": str(Path(request.task_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "collections": collections,
        "episode_paths": episode_paths,
        "collection_acceptance": best_acceptance,
        "collection_acceptances": collection_acceptances,
        "quality_gate": quality_gate,
        "metrics": metrics,
        "parameters": {
            "collect_steps": min(max(1, int(request.collect_steps)), task.max_steps),
            "collect_rollouts": rollout_count,
            "seed": int(request.seed),
            "collection_goal_bias_interval": max(0, int(request.collection_goal_bias_interval)),
            "collection_goal_corridor_interval": max(0, int(request.collection_goal_corridor_interval)),
            "collection_goal_corridor_lateral_m": max(0.0, float(request.collection_goal_corridor_lateral_m)),
            "collection_coverage_grid_size": max(0, int(request.collection_coverage_grid_size)),
            "collection_coverage_target_interval": max(0, int(request.collection_coverage_target_interval)),
            "collection_max_target_steps": max(1, int(request.collection_max_target_steps)),
            "collection_strategy": collection_strategy,
            "collection_route_target_interval": agent_options.get("route_target_interval", 0),
            "collection_route_lateral_m": agent_options.get("route_lateral_m", 0.0),
            "collection_multi_start": bool(request.collection_multi_start),
            "collection_multi_start_lateral_m": max(0.0, float(request.collection_multi_start_lateral_m)),
            "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
            "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
            "max_collection_min_goal_distance_m": max(0.0, float(request.max_collection_min_goal_distance_m)),
            "min_unique_region_cells": max(0, int(request.min_unique_region_cells)),
        },
    }
    manifest_path = output_dir / REGION_TRAINING_COLLECTION_FILENAME
    payload["collection_manifest_path"] = str(manifest_path.resolve())
    training_run = write_training_run_record(
        output_dir,
        preset_id="beamng_region_training_data",
        status=collection_status,
        dataset_root=episode_paths[0],
        adapter="beamng_episode",
        sequence_id=task.task_id,
        artifact_path=str(manifest_path.resolve()),
        artifact_type="beamng_collection",
        metrics=metrics,
        history={
            "collection_min_goal_distance": [best_acceptance.get("min_goal_distance")],
            "collection_progress_ratio": [quality_gate.get("progress_ratio")],
            "collection_distance_traveled": [collection_distance_total],
            "route_coverage_ratio": [route_metrics["route_coverage_ratio"]],
            "goal_zone_coverage": [route_metrics["goal_zone_coverage"]],
        },
        parameters=payload["parameters"],
        summary={
            "task_path": str(Path(request.task_path).resolve()),
            "quality_gate": quality_gate,
            "coverage": coverage,
            "route_metrics": route_metrics,
            "collection_acceptance": best_acceptance,
            "collection_acceptances": collection_acceptances,
            "episode_paths": episode_paths,
        },
    )
    payload["training_run_path"] = training_run["path"]
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return payload


def train_region_world_model_from_collection(request: RegionWorldModelTrainingRequest) -> dict[str, Any]:
    world_model_type = _normalize_lightweight_region_world_model_type(request.world_model_type)
    manifest_path = Path(request.collection_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Collection manifest not found: {manifest_path}")
    collection = _read_json(manifest_path)
    if not isinstance(collection, dict) or str(collection.get("status") or "") != "completed":
        raise ValueError("Collection manifest must be a completed BeamNG region collection.")
    quality_gate = collection.get("quality_gate") if isinstance(collection.get("quality_gate"), dict) else {}
    if quality_gate and not bool(quality_gate.get("passed", True)):
        reason = str(quality_gate.get("reason") or "collection_insufficient")
        raise ValueError(f"Collection quality gate failed: {reason}")
    task = _navigation_task_from_collection_manifest(collection, manifest_path)
    episode_paths = [str(Path(path).resolve()) for path in collection.get("episode_paths", []) if str(path or "").strip()]
    if not episode_paths:
        raise ValueError("Collection manifest has no episode paths.")
    sequences = [_episode_trace_to_dataset_sequence(path, task) for path in episode_paths]

    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "beamng_region_world_models" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _fit_lightweight_region_world_model(world_model_type, sequences)
    model.metadata.update(
        {
            "training_source": "beamng_region_collection",
            "task_id": task.task_id,
            "collection_manifest_path": str(manifest_path.resolve()),
            "episode_paths": episode_paths,
            "collection_rollout_count": len(episode_paths),
        }
    )
    model_dir = output_dir / "model"
    metadata_path = model.save(model_dir)
    training = {
        "status": "completed",
        "model_type": model.model_type,
        "model_path": str(model_dir.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metrics": model.metadata,
    }
    collection_metrics = collection.get("metrics") if isinstance(collection.get("metrics"), dict) else {}
    model_quality_metrics = _world_model_training_quality_metrics(model.metadata)
    metrics = {
        **model_quality_metrics,
        "collection_rollout_count": len(episode_paths),
        "collection_distance_traveled": collection_metrics.get("collection_distance_traveled"),
        "collection_min_goal_distance": collection_metrics.get("collection_min_goal_distance"),
        "collection_collision_count": collection_metrics.get("collection_collision_count"),
    }
    training_run = write_training_run_record(
        output_dir,
        preset_id="region_world_model_training",
        status="completed",
        dataset_root=str(manifest_path.resolve()),
        adapter="beamng_episode_collection",
        sequence_id=task.task_id,
        artifact_path=str(model_dir.resolve()),
        artifact_type="world_model",
        metrics=metrics,
        history={
            "train_rmse": [model.metadata.get("train_rmse")],
            "train_mse": [model.metadata.get("train_mse")],
            "validation_rmse": [model.metadata.get("validation_rmse")],
            "validation_mse": [model.metadata.get("validation_mse")],
            "collection_min_goal_distance": [collection_metrics.get("collection_min_goal_distance")],
        },
        parameters={
            "world_model_type": world_model_type,
            "collection_manifest_path": str(manifest_path.resolve()),
        },
        summary={
            "task": task.to_dict(),
            "collection_manifest_path": str(manifest_path.resolve()),
            "collection_metrics": collection_metrics,
            "model_quality_metrics": model_quality_metrics,
        },
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "task": task.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "collection_manifest_path": str(manifest_path.resolve()),
        "episode_paths": episode_paths,
        "training": training,
        "training_run_path": training_run["path"],
    }
    if request.register_world_model_config:
        validation = {
            **model_quality_metrics,
            "collection_rollout_count": len(episode_paths),
            "collection_min_goal_distance": collection_metrics.get("collection_min_goal_distance"),
            "collection_collision_count": collection_metrics.get("collection_collision_count"),
            "route_free": True,
            "evaluation_route_mode": "route_free",
            "route_waypoint_count": 0,
            "model_support_subgoals": True,
            "model_support_field_subgoals": False,
            "model_support_graph_subgoals": True,
            "evaluation_local_subgoal_distance_m": 12.0,
            "evaluation_allow_reverse_recovery": False,
        }
        config = save_world_model_config(
            config_id=f"{task.task_id}_beamng_trained_world_model",
            label=f"{task.task_id} BeamNG trained world model",
            algorithm="world_model_direct",
            world_model=model.model_type,
            model_path=str(model_dir.resolve()),
            source_training_run_path=str(training_run["path"]),
            validation=validation,
            path=request.world_model_config_path or None,
        )
        _attach_world_model_config_to_training_run(str(training_run["path"]), config)
        payload["world_model_config"] = config
    summary_path = output_dir / "region_world_model_training_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def run_region_self_supervised_world_model(request: RegionSelfSupervisedWorldModelRequest) -> dict[str, Any]:
    task = load_navigation_region_task(request.task_path)
    world_model_type = _normalize_lightweight_region_world_model_type(request.world_model_type)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "region_self_supervised" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_strategy = str(request.collection_strategy or "region_explorer").strip().lower()
    collection_scenario = _collection_region_scenario(task, strategy=collection_strategy)
    route_aware = collection_strategy in {"route_aware", "route-aware", "curriculum", "route_curriculum"}
    collection_agent_options = {
        "goal_bias_interval": max(0, int(request.collection_goal_bias_interval)),
        "goal_corridor_interval": max(0, int(request.collection_goal_corridor_interval)),
        "goal_corridor_lateral_m": max(0.0, float(request.collection_goal_corridor_lateral_m)),
        "coverage_grid_size": max(0, int(request.collection_coverage_grid_size)),
        "coverage_target_interval": max(0, int(request.collection_coverage_target_interval)),
        "max_target_steps": max(1, int(request.collection_max_target_steps)),
    }
    if route_aware:
        collection_agent_options.update(
            {
                "route_target_interval": max(1, int(request.collection_route_target_interval or 1)),
                "route_lateral_m": max(0.0, float(request.collection_route_lateral_m)),
            }
        )
    evaluation_route_free = str(request.evaluation_route_mode or "route_free").strip().lower() in {"route_free", "none", "direct"}
    evaluation_scenario = (
        _route_free_region_scenario(task.to_beamng_scenario(mode="evaluation"))
        if evaluation_route_free
        else task.to_beamng_scenario(mode="evaluation")
    )

    collections: list[dict[str, Any]] = []
    collection_acceptances: list[dict[str, Any]] = []
    collection_distances: list[float] = []
    episode_paths: list[str] = []
    sequences: list[DatasetSequence] = []
    short_episode_paths: list[str] = []
    rollout_count = max(1, int(request.collect_rollouts))
    for rollout_index in range(rollout_count):
        rollout_scenario = _collection_rollout_scenario(
            task,
            collection_scenario,
            strategy=collection_strategy,
            rollout_index=rollout_index,
            rollout_count=rollout_count,
            multi_start=bool(request.collection_multi_start),
            lateral_m=float(request.collection_multi_start_lateral_m),
            seed=int(request.seed) + rollout_index,
        )
        collection = _run_region_beamng_episode_with_reconnect_retry(
            scenario=rollout_scenario,
            vehicle=request.vehicle,
            max_steps=min(max(1, int(request.collect_steps)), task.max_steps),
            seed=int(request.seed) + rollout_index,
            agent_name="region_explorer",
            agent_options=collection_agent_options,
            world_model_type="simple_kinematic",
            world_model_path="",
            planner="",
            planner_horizon=request.planner_horizon,
            planner_samples=request.planner_samples,
            planner_iterations=request.planner_iterations,
            record=True,
            beamng_gfx=request.beamng_gfx,
            pre_run_hold_sec=request.pre_run_hold_sec if rollout_index == 0 else 0.0,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=0.0,
            close_beamng=True,
        )
        episode_path = collection.get("episode_path")
        if not episode_path:
            raise RuntimeError("Region self-supervised collection did not produce an episode path.")
        collections.append(collection)
        episode_paths.append(str(Path(episode_path).resolve()))
        collection_acceptances.append(_navigation_acceptance(collection, task))
        collection_metrics = collection.get("metrics", {}) if isinstance(collection.get("metrics"), dict) else {}
        collection_distance = _float_or_nan(collection_metrics.get("horizontal_distance_traveled"))
        if math.isfinite(collection_distance):
            collection_distances.append(collection_distance)
        try:
            sequences.append(_episode_trace_to_dataset_sequence(episode_path, task))
        except ValueError:
            short_episode_paths.append(str(Path(episode_path).resolve()))
    collection_acceptance = min(collection_acceptances, key=lambda row: _finite_or_inf(row.get("min_goal_distance")))
    collection_distance = float(sum(collection_distances)) if collection_distances else math.nan
    coverage = _collection_coverage_metrics(
        task,
        episode_paths,
        grid_size=max(2, int(request.collection_coverage_grid_size)),
    )
    route_metrics = _collection_route_metrics(task, episode_paths)
    use_experience_corridor = bool(request.use_experience_corridor) and evaluation_route_free
    experience_route = (
        _experience_route_from_episode_traces(
            task,
            episode_paths,
            min_spacing_m=max(0.25, float(request.experience_route_min_spacing_m)),
            max_points=max(2, int(request.experience_route_max_points)),
        )
        if use_experience_corridor
        else []
    )
    experience_route_point_count = len(experience_route)
    quality_gate = _collection_quality_gate(
        task,
        collection_acceptance,
        min_progress_ratio=request.min_collection_goal_progress_ratio,
        route_coverage_ratio=route_metrics["route_coverage_ratio"],
        min_route_coverage_ratio=request.min_route_coverage_ratio,
        goal_zone_coverage=route_metrics["goal_zone_coverage"],
        min_goal_zone_coverage=request.min_goal_zone_coverage,
        max_collection_min_goal_distance_m=request.max_collection_min_goal_distance_m,
        unique_region_cells=coverage["cell_count"],
        min_unique_region_cells=request.min_unique_region_cells,
    )
    if short_episode_paths:
        quality_gate = {
            **quality_gate,
            "passed": False,
            "reason": "collection_episode_too_short",
            "short_episode_paths": short_episode_paths,
        }
    trajectory_plot_path = _write_region_self_supervised_trajectory_plot(task, output_dir, episode_paths=episode_paths)
    if not quality_gate["passed"]:
        diagnostics = _region_self_supervised_diagnostics(
            quality_gate=quality_gate,
            collection_acceptance=collection_acceptance,
            acceptance={},
            training={},
            region_navigation={
                "collection_agent": "region_explorer",
                "evaluation_agent": request.evaluation_agent,
                "route_free": evaluation_route_free,
                "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
                "experience_corridor": bool(experience_route),
                "experience_route_point_count": experience_route_point_count,
            },
        )
        training_run = write_training_run_record(
            output_dir,
            preset_id="region_self_supervised_world_model",
            status="collection_insufficient",
            dataset_root=episode_paths[0],
            adapter="beamng_episode",
            sequence_id=task.task_id,
            artifact_path="",
            artifact_type="world_model",
            metrics={
                "collection_goal_reached": bool(collection_acceptance.get("goal_reached")),
                "collection_min_goal_distance": collection_acceptance.get("min_goal_distance"),
                "collection_final_goal_distance": collection_acceptance.get("final_goal_distance"),
                "collection_distance_traveled": collection_distance,
                "collection_rollout_count": rollout_count,
                "collection_collision_count": collection_acceptance.get("collision_count"),
                "collection_progress_ratio": quality_gate.get("progress_ratio"),
                "required_collection_progress_ratio": quality_gate.get("required_progress_ratio"),
                "collection_coverage_cell_count": coverage["cell_count"],
                "collection_coverage_total_cells": coverage["total_cells"],
                "collection_coverage_ratio": coverage["ratio"],
                "unique_region_cells": coverage["cell_count"],
                "route_coverage_ratio": route_metrics["route_coverage_ratio"],
                "goal_zone_coverage": route_metrics["goal_zone_coverage"],
                "experience_route_point_count": experience_route_point_count,
            },
            history={
                "collection_min_goal_distance": [collection_acceptance.get("min_goal_distance")],
                "collection_progress_ratio": [quality_gate.get("progress_ratio")],
                "collection_coverage_ratio": [coverage["ratio"]],
                "route_coverage_ratio": [route_metrics["route_coverage_ratio"]],
                "goal_zone_coverage": [route_metrics["goal_zone_coverage"]],
            },
            parameters={
                "world_model_type": world_model_type,
                "planner": request.planner,
                "planner_horizon": request.planner_horizon,
                "planner_samples": request.planner_samples,
                "planner_iterations": request.planner_iterations,
                "evaluation_agent": request.evaluation_agent,
                "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
                "collect_rollouts": rollout_count,
                "min_collection_goal_progress_ratio": float(request.min_collection_goal_progress_ratio),
                "collection_goal_bias_interval": max(0, int(request.collection_goal_bias_interval)),
                "collection_goal_corridor_interval": max(0, int(request.collection_goal_corridor_interval)),
                "collection_goal_corridor_lateral_m": max(0.0, float(request.collection_goal_corridor_lateral_m)),
                "collection_coverage_grid_size": max(0, int(request.collection_coverage_grid_size)),
                "collection_coverage_target_interval": max(0, int(request.collection_coverage_target_interval)),
                "collection_max_target_steps": max(1, int(request.collection_max_target_steps)),
                "collection_strategy": collection_strategy,
                "collection_route_target_interval": collection_agent_options.get("route_target_interval", 0),
                "collection_route_lateral_m": collection_agent_options.get("route_lateral_m", 0.0),
                "collection_multi_start": bool(request.collection_multi_start),
                "collection_multi_start_lateral_m": max(0.0, float(request.collection_multi_start_lateral_m)),
                "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
                "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
                "max_collection_min_goal_distance_m": max(0.0, float(request.max_collection_min_goal_distance_m)),
                "min_unique_region_cells": max(0, int(request.min_unique_region_cells)),
                "use_experience_corridor": bool(use_experience_corridor),
                "experience_route_min_spacing_m": max(0.25, float(request.experience_route_min_spacing_m)),
                "experience_route_max_points": max(2, int(request.experience_route_max_points)),
                "evaluation_allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
                "evaluation_reverse_recovery_after_steps": max(18, int(request.evaluation_reverse_recovery_after_steps)),
                "evaluation_local_subgoal_distance_m": max(1.0, float(request.evaluation_local_subgoal_distance_m)),
                "evaluation_use_model_support_subgoals": bool(request.evaluation_use_model_support_subgoals),
                "evaluation_use_model_support_field_subgoals": bool(request.evaluation_use_model_support_field_subgoals),
                "evaluation_use_model_support_graph_subgoals": bool(request.evaluation_use_model_support_graph_subgoals),
            },
            summary={
                "task_path": str(Path(request.task_path).resolve()),
                "quality_gate": quality_gate,
                "coverage": coverage,
                "route_metrics": route_metrics,
                "experience_route_point_count": experience_route_point_count,
                "experience_route": experience_route,
                "trajectory_plot_path": trajectory_plot_path,
                "diagnostics": diagnostics,
                "collection_acceptance": collection_acceptance,
                "collection_acceptances": collection_acceptances,
            },
        )
        payload = {
            "status": "collection_insufficient",
            "task": task.to_dict(),
            "output_dir": str(output_dir.resolve()),
            "model_dir": "",
            "collection": collections[0],
            "collections": collections,
            "training": {"status": "skipped", "reason": quality_gate["reason"]},
            "evaluation": {},
            "acceptance": {},
            "quality_gate": quality_gate,
            "coverage": coverage,
            "route_metrics": route_metrics,
            "trajectory_plot_path": trajectory_plot_path,
            "diagnostics": diagnostics,
            "region_navigation": {
                "collection_agent": "region_explorer",
                "evaluation_agent": request.evaluation_agent,
                "route_free": evaluation_route_free,
                "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
                "experience_corridor": bool(experience_route),
                "experience_route_point_count": experience_route_point_count,
            },
            "training_run_path": training_run["path"],
        }
        summary_path = output_dir / "region_self_supervised_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        payload["summary_path"] = str(summary_path.resolve())
        return payload

    model = _fit_lightweight_region_world_model(world_model_type, sequences)
    model.metadata.update(
        {
            "training_source": "beamng_region_self_supervised",
            "task_id": task.task_id,
            "episode_path": episode_paths[0],
            "episode_paths": episode_paths,
            "collection_rollout_count": rollout_count,
            "collection_coverage_cell_count": coverage["cell_count"],
            "collection_coverage_total_cells": coverage["total_cells"],
            "collection_coverage_ratio": coverage["ratio"],
            "unique_region_cells": coverage["cell_count"],
            "route_coverage_ratio": route_metrics["route_coverage_ratio"],
            "goal_zone_coverage": route_metrics["goal_zone_coverage"],
            "experience_route_point_count": experience_route_point_count,
        }
    )
    model_dir = output_dir / "model"
    metadata_path = model.save(model_dir)
    training = {
        "status": "completed",
        "model_type": model.model_type,
        "model_path": str(model_dir.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metrics": model.metadata,
    }

    if experience_route:
        evaluation_scenario = _with_experience_corridor(evaluation_scenario, experience_route)

    evaluation_kwargs = {
        "scenario": evaluation_scenario,
        "vehicle": request.vehicle,
        "max_steps": min(max(1, int(request.eval_steps)), task.max_steps),
        "seed": request.seed,
        "agent_name": request.evaluation_agent,
        "world_model_type": model.model_type,
        "world_model_path": str(model_dir.resolve()),
        "planner": request.planner,
        "planner_horizon": request.planner_horizon,
        "planner_samples": request.planner_samples,
        "planner_iterations": request.planner_iterations,
        "record": True,
        "beamng_gfx": request.beamng_gfx,
        "pre_run_hold_sec": 0.0,
        "step_delay_sec": request.step_delay_sec,
        "post_run_hold_sec": request.post_run_hold_sec,
        "close_beamng": request.close_beamng,
        "planner_config_overrides": _region_evaluation_planner_overrides(request),
        "agent_options": {
            "allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
            "reverse_recovery_after_steps": max(18, int(request.evaluation_reverse_recovery_after_steps)),
            "local_subgoal_distance_m": max(1.0, float(request.evaluation_local_subgoal_distance_m)),
            "use_model_support_subgoals": bool(request.evaluation_use_model_support_subgoals),
            "use_model_support_field_subgoals": bool(request.evaluation_use_model_support_field_subgoals),
            "use_model_support_graph_subgoals": bool(request.evaluation_use_model_support_graph_subgoals),
        }
        if request.evaluation_agent == "world_model_direct"
        else {},
    }
    evaluation = _run_region_beamng_episode_with_reconnect_retry(**evaluation_kwargs)
    acceptance = _navigation_acceptance(evaluation, task)
    trajectory_plot_path = _write_region_self_supervised_trajectory_plot(
        task,
        output_dir,
        episode_paths=episode_paths,
        evaluation=evaluation,
    )
    region_navigation = evaluation.get("region_navigation", {}) if isinstance(evaluation.get("region_navigation"), dict) else {}
    region_navigation_payload = {
        **region_navigation,
        "collection_agent": "region_explorer",
        "evaluation_agent": request.evaluation_agent,
        "route_free": evaluation_route_free,
        "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
        "experience_corridor": bool(experience_route),
        "experience_route_point_count": experience_route_point_count,
    }
    diagnostics = _region_self_supervised_diagnostics(
        quality_gate=quality_gate,
        collection_acceptance=collection_acceptance,
        acceptance=acceptance,
        training=training,
        region_navigation=region_navigation_payload,
    )
    model_quality_metrics = _world_model_training_quality_metrics(model.metadata)
    training_run = write_training_run_record(
        output_dir,
        preset_id="region_self_supervised_world_model",
        status="completed",
        dataset_root=episode_paths[0],
        adapter="beamng_episode",
        sequence_id=task.task_id,
        artifact_path=str(model_dir.resolve()),
        artifact_type="world_model",
        metrics={
            "goal_success": bool(acceptance.get("goal_success")),
            "goal_reached": bool(acceptance.get("goal_reached")),
            "min_goal_distance": acceptance.get("min_goal_distance"),
            "final_goal_distance": acceptance.get("final_goal_distance"),
            "collision_count": acceptance.get("collision_count"),
            **model_quality_metrics,
            "collection_goal_reached": bool(collection_acceptance.get("goal_reached")),
            "collection_min_goal_distance": collection_acceptance.get("min_goal_distance"),
            "collection_final_goal_distance": collection_acceptance.get("final_goal_distance"),
            "collection_distance_traveled": collection_distance,
            "collection_rollout_count": rollout_count,
            "collection_collision_count": collection_acceptance.get("collision_count"),
            "collection_progress_ratio": quality_gate.get("progress_ratio"),
            "required_collection_progress_ratio": quality_gate.get("required_progress_ratio"),
            "collection_coverage_cell_count": coverage["cell_count"],
            "collection_coverage_total_cells": coverage["total_cells"],
            "collection_coverage_ratio": coverage["ratio"],
            "unique_region_cells": coverage["cell_count"],
            "route_coverage_ratio": route_metrics["route_coverage_ratio"],
            "goal_zone_coverage": route_metrics["goal_zone_coverage"],
            "experience_route_point_count": experience_route_point_count,
        },
        history={
            "train_rmse": [model.metadata.get("train_rmse")],
            "train_mse": [model.metadata.get("train_mse")],
            "validation_rmse": [model.metadata.get("validation_rmse")],
            "validation_mse": [model.metadata.get("validation_mse")],
            "collection_min_goal_distance": [collection_acceptance.get("min_goal_distance")],
            "collection_progress_ratio": [quality_gate.get("progress_ratio")],
            "collection_coverage_ratio": [coverage["ratio"]],
            "route_coverage_ratio": [route_metrics["route_coverage_ratio"]],
            "goal_zone_coverage": [route_metrics["goal_zone_coverage"]],
            "evaluation_min_goal_distance": [acceptance.get("min_goal_distance")],
        },
        parameters={
            "world_model_type": world_model_type,
            "planner": request.planner,
            "planner_horizon": request.planner_horizon,
            "planner_samples": request.planner_samples,
            "planner_iterations": request.planner_iterations,
            "evaluation_agent": request.evaluation_agent,
            "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
            "collect_rollouts": rollout_count,
            "min_collection_goal_progress_ratio": float(request.min_collection_goal_progress_ratio),
            "collection_goal_bias_interval": max(0, int(request.collection_goal_bias_interval)),
            "collection_goal_corridor_interval": max(0, int(request.collection_goal_corridor_interval)),
            "collection_goal_corridor_lateral_m": max(0.0, float(request.collection_goal_corridor_lateral_m)),
            "collection_coverage_grid_size": max(0, int(request.collection_coverage_grid_size)),
            "collection_coverage_target_interval": max(0, int(request.collection_coverage_target_interval)),
            "collection_max_target_steps": max(1, int(request.collection_max_target_steps)),
            "collection_strategy": collection_strategy,
            "collection_route_target_interval": collection_agent_options.get("route_target_interval", 0),
            "collection_route_lateral_m": collection_agent_options.get("route_lateral_m", 0.0),
            "collection_multi_start": bool(request.collection_multi_start),
            "collection_multi_start_lateral_m": max(0.0, float(request.collection_multi_start_lateral_m)),
            "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
            "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
            "max_collection_min_goal_distance_m": max(0.0, float(request.max_collection_min_goal_distance_m)),
            "min_unique_region_cells": max(0, int(request.min_unique_region_cells)),
            "use_experience_corridor": bool(use_experience_corridor),
            "experience_route_min_spacing_m": max(0.25, float(request.experience_route_min_spacing_m)),
            "experience_route_max_points": max(2, int(request.experience_route_max_points)),
            "evaluation_allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
            "evaluation_reverse_recovery_after_steps": max(18, int(request.evaluation_reverse_recovery_after_steps)),
            "evaluation_local_subgoal_distance_m": max(1.0, float(request.evaluation_local_subgoal_distance_m)),
            "evaluation_use_model_support_subgoals": bool(request.evaluation_use_model_support_subgoals),
            "evaluation_use_model_support_field_subgoals": bool(request.evaluation_use_model_support_field_subgoals),
            "evaluation_use_model_support_graph_subgoals": bool(request.evaluation_use_model_support_graph_subgoals),
        },
        summary={
            "task_path": str(Path(request.task_path).resolve()),
            "quality_gate": quality_gate,
            "coverage": coverage,
            "route_metrics": route_metrics,
            "experience_route_point_count": experience_route_point_count,
            "experience_route": experience_route,
            "trajectory_plot_path": trajectory_plot_path,
            "model_quality_metrics": model_quality_metrics,
            "diagnostics": diagnostics,
            "collection_acceptance": collection_acceptance,
            "collection_acceptances": collection_acceptances,
            "acceptance": acceptance,
        },
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "task": task.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "collection": collections[0],
        "collections": collections,
        "training": training,
        "evaluation": evaluation,
        "acceptance": acceptance,
        "quality_gate": quality_gate,
        "coverage": coverage,
        "route_metrics": route_metrics,
        "trajectory_plot_path": trajectory_plot_path,
        "diagnostics": diagnostics,
        "region_navigation": region_navigation_payload,
        "training_run_path": training_run["path"],
    }
    world_model_config = _register_region_self_supervised_world_model_config(
        request=request,
        task_id=task.task_id,
        model_dir=str(model_dir.resolve()),
        model_type=model.model_type,
        training_run_path=str(training_run["path"]),
        acceptance=acceptance,
        quality_gate=quality_gate,
        model_metadata=model.metadata,
        experience_corridor_used=bool(experience_route),
    )
    if world_model_config:
        payload["world_model_config"] = world_model_config
    summary_path = output_dir / "region_self_supervised_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def run_region_world_model_evaluation(request: RegionWorldModelEvaluationRequest) -> dict[str, Any]:
    task = load_navigation_region_task(request.task_path)
    if not str(request.world_model_path or "").strip():
        raise ValueError("world_model_path is required for direct region world-model evaluation.")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "region_world_model_eval" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario = _route_free_region_scenario(task.to_beamng_scenario(mode="evaluation"))
    experience_route = (
        _experience_route_from_model_metadata(
            task,
            request.world_model_path,
            min_spacing_m=max(0.25, float(request.experience_route_min_spacing_m)),
            max_points=max(2, int(request.experience_route_max_points)),
        )
        if bool(request.use_experience_corridor) and request.evaluation_agent == "world_model_direct"
        else []
    )
    if experience_route:
        scenario = _with_experience_corridor(scenario, experience_route)
    experience_route_point_count = len(experience_route)
    planner_config_overrides = _region_evaluation_planner_overrides(request)
    direct_agent_options = {
        "allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
        "reverse_recovery_after_steps": max(18, int(request.evaluation_reverse_recovery_after_steps)),
        "local_subgoal_distance_m": max(1.0, float(request.evaluation_local_subgoal_distance_m)),
        "use_model_support_subgoals": bool(request.evaluation_use_model_support_subgoals),
        "use_model_support_field_subgoals": bool(request.evaluation_use_model_support_field_subgoals),
        "use_model_support_graph_subgoals": bool(request.evaluation_use_model_support_graph_subgoals),
    } if request.evaluation_agent == "world_model_direct" else {}
    evaluation = _run_region_beamng_episode_with_reconnect_retry(
        scenario=scenario,
        vehicle=request.vehicle,
        max_steps=min(max(1, int(request.eval_steps)), task.max_steps),
        seed=request.seed,
        agent_name=request.evaluation_agent,
        world_model_type=request.world_model_type,
        world_model_path=request.world_model_path,
        planner=request.planner,
        planner_horizon=request.planner_horizon,
        planner_samples=request.planner_samples,
        planner_iterations=request.planner_iterations,
        record=True,
        beamng_gfx=request.beamng_gfx,
        pre_run_hold_sec=request.pre_run_hold_sec,
        step_delay_sec=request.step_delay_sec,
        post_run_hold_sec=request.post_run_hold_sec,
        close_beamng=request.close_beamng,
        agent_options=direct_agent_options,
        planner_config_overrides=planner_config_overrides,
    )
    acceptance = _navigation_acceptance(evaluation, task)
    region_navigation = evaluation.get("region_navigation", {}) if isinstance(evaluation.get("region_navigation"), dict) else {}
    baselines: dict[str, Any] = {
        "route_free": {
            "evaluation": evaluation,
            "acceptance": acceptance,
            "region_navigation": {
                **region_navigation,
                "evaluation_agent": request.evaluation_agent,
                "route_free": True,
                "evaluation_route_mode": "route_free",
                "agent_options": direct_agent_options,
                "experience_corridor": bool(experience_route),
                "experience_route_point_count": experience_route_point_count,
            },
        }
    }
    route_guided_evaluation: dict[str, Any] | None = None
    route_guided_acceptance: dict[str, Any] | None = None
    if request.include_route_guided_baseline:
        route_guided_evaluation = _run_region_beamng_episode_with_reconnect_retry(
            scenario=_route_guided_region_scenario(task),
            vehicle=request.vehicle,
            max_steps=min(max(1, int(request.eval_steps)), task.max_steps),
            seed=request.seed,
            agent_name="route_world_model",
            world_model_type=request.world_model_type,
            world_model_path=request.world_model_path,
            planner=request.planner,
            planner_horizon=request.planner_horizon,
            planner_samples=request.planner_samples,
            planner_iterations=request.planner_iterations,
            record=True,
            beamng_gfx=request.beamng_gfx,
            pre_run_hold_sec=0.0,
            step_delay_sec=request.step_delay_sec,
            post_run_hold_sec=request.post_run_hold_sec,
            close_beamng=request.close_beamng,
            planner_config_overrides=planner_config_overrides,
        )
        route_guided_acceptance = _navigation_acceptance(route_guided_evaluation, task)
        route_guided_region_navigation = (
            route_guided_evaluation.get("region_navigation", {}) if isinstance(route_guided_evaluation.get("region_navigation"), dict) else {}
        )
        baselines["route_guided"] = {
            "evaluation": route_guided_evaluation,
            "acceptance": route_guided_acceptance,
            "region_navigation": {
                **route_guided_region_navigation,
                "evaluation_agent": "route_world_model",
                "route_free": False,
                "evaluation_route_mode": "task_route",
            },
        }
    comparison = _region_evaluation_comparison(
        route_free_acceptance=acceptance,
        route_free_evaluation=evaluation,
        route_guided_acceptance=route_guided_acceptance,
        route_guided_evaluation=route_guided_evaluation,
    )
    trajectory_plot_path = ""
    if request.write_trajectory_plot:
        trajectory_plot_path = str((output_dir / "region_world_model_trajectory.svg").resolve())
        traces = {"route_free": load_episode_trace(evaluation.get("episode_path", ""))}
        if route_guided_evaluation is not None:
            traces["route_guided"] = load_episode_trace(route_guided_evaluation.get("episode_path", ""))
        _write_region_trajectory_svg(task, Path(trajectory_plot_path), traces=traces)
    summary_path = output_dir / "region_world_model_evaluation_summary.json"
    training_run = write_training_run_record(
        output_dir,
        preset_id="region_world_model_evaluation",
        status="completed",
        dataset_root=str(Path(request.task_path).resolve()),
        adapter="beamng_region_task",
        sequence_id=task.task_id,
        artifact_path=str(summary_path.resolve()),
        artifact_type="world_model_evaluation",
        metrics=comparison,
        history={key: [value] for key, value in _numeric_metric_items(comparison)},
        parameters={
            "world_model_type": request.world_model_type,
            "world_model_path": str(Path(request.world_model_path).resolve()),
            "planner": request.planner,
            "planner_horizon": request.planner_horizon,
            "planner_samples": request.planner_samples,
            "planner_iterations": request.planner_iterations,
            "evaluation_agent": request.evaluation_agent,
            "evaluation_route_mode": "route_free",
            "include_route_guided_baseline": bool(request.include_route_guided_baseline),
            "use_experience_corridor": bool(request.use_experience_corridor),
            "evaluation_allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
            "evaluation_reverse_recovery_after_steps": max(18, int(request.evaluation_reverse_recovery_after_steps)),
            "evaluation_local_subgoal_distance_m": max(1.0, float(request.evaluation_local_subgoal_distance_m)),
            "evaluation_use_model_support_subgoals": bool(request.evaluation_use_model_support_subgoals),
            "evaluation_use_model_support_field_subgoals": bool(request.evaluation_use_model_support_field_subgoals),
            "evaluation_use_model_support_graph_subgoals": bool(request.evaluation_use_model_support_graph_subgoals),
        },
        summary={
            "task_path": str(Path(request.task_path).resolve()),
            "world_model_path": str(Path(request.world_model_path).resolve()),
            "summary_path": str(summary_path.resolve()),
            "trajectory_plot_path": trajectory_plot_path,
            "route_free_episode_path": str(evaluation.get("episode_path") or ""),
            "route_guided_episode_path": str(route_guided_evaluation.get("episode_path") or "") if route_guided_evaluation else "",
            "comparison": comparison,
        },
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "task": task.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "model_dir": str(Path(request.world_model_path).resolve()),
        "evaluation": evaluation,
        "acceptance": acceptance,
        "baselines": baselines,
        "comparison": comparison,
        "trajectory_plot_path": trajectory_plot_path,
        "region_navigation": {
            **region_navigation,
            "evaluation_agent": request.evaluation_agent,
            "route_free": True,
            "evaluation_route_mode": "route_free",
            "agent_options": direct_agent_options,
            "experience_corridor": bool(experience_route),
            "experience_route_point_count": experience_route_point_count,
        },
        "experience_route": experience_route,
        "training_run_path": training_run["path"],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def compare_region_world_models_from_collection(request: RegionWorldModelComparisonRequest) -> dict[str, Any]:
    manifest_path = Path(request.collection_manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Collection manifest not found: {manifest_path}")
    collection = _read_json(manifest_path)
    if not isinstance(collection, dict) or str(collection.get("status") or "") != "completed":
        raise ValueError("Collection manifest must be a completed BeamNG region collection.")
    task = _navigation_task_from_collection_manifest(collection, manifest_path)
    task_path = str(request.task_path or collection.get("task_path") or "").strip()
    if not task_path:
        raise ValueError("Region world model comparison requires task_path in the request or collection manifest.")

    world_model_types: list[str] = []
    for raw_name in request.world_model_types:
        normalized = _normalize_lightweight_region_world_model_type(raw_name)
        if normalized not in world_model_types:
            world_model_types.append(normalized)
    if not world_model_types:
        raise ValueError("At least one world model type is required for comparison.")

    stamp = time.strftime("%Y%m%dT%H%M%S")
    output_dir = Path(request.output_dir or ROOT / "outputs" / "region_world_model_compare" / _safe_name(task.task_id) / stamp)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_rows: list[dict[str, Any]] = []
    for model_type in world_model_types:
        model_output_dir = output_dir / model_type
        training = train_region_world_model_from_collection(
            RegionWorldModelTrainingRequest(
                collection_manifest_path=str(manifest_path.resolve()),
                world_model_type=model_type,
                output_dir=str(model_output_dir / "training"),
                register_world_model_config=False,
            )
        )
        evaluation = run_region_world_model_evaluation(
            RegionWorldModelEvaluationRequest(
                task_path=task_path,
                world_model_type=model_type,
                world_model_path=str(training["model_dir"]),
                vehicle=request.vehicle,
                output_dir=str(model_output_dir / "evaluation"),
                eval_steps=request.eval_steps,
                seed=request.seed,
                planner=request.planner,
                planner_horizon=request.planner_horizon,
                planner_samples=request.planner_samples,
                planner_iterations=request.planner_iterations,
                planner_goal_weight=request.planner_goal_weight,
                planner_progress_weight=request.planner_progress_weight,
                planner_risk_weight=request.planner_risk_weight,
                planner_heading_weight=request.planner_heading_weight,
                evaluation_agent=request.evaluation_agent,
                evaluation_allow_reverse_recovery=request.evaluation_allow_reverse_recovery,
                evaluation_reverse_recovery_after_steps=request.evaluation_reverse_recovery_after_steps,
                evaluation_local_subgoal_distance_m=request.evaluation_local_subgoal_distance_m,
                evaluation_use_model_support_subgoals=request.evaluation_use_model_support_subgoals,
                evaluation_use_model_support_field_subgoals=request.evaluation_use_model_support_field_subgoals,
                evaluation_use_model_support_graph_subgoals=request.evaluation_use_model_support_graph_subgoals,
                use_experience_corridor=request.use_experience_corridor,
                experience_route_min_spacing_m=request.experience_route_min_spacing_m,
                experience_route_max_points=request.experience_route_max_points,
                include_route_guided_baseline=request.include_route_guided_baseline,
                write_trajectory_plot=request.write_trajectory_plot,
                beamng_gfx=request.beamng_gfx,
                close_beamng=request.close_beamng,
                step_delay_sec=request.step_delay_sec,
                pre_run_hold_sec=request.pre_run_hold_sec,
                post_run_hold_sec=request.post_run_hold_sec,
            )
        )
        model_rows.append(
            {
                "world_model_type": model_type,
                "model_dir": training["model_dir"],
                "training": training,
                "evaluation": evaluation,
                "comparison": dict(evaluation.get("comparison") if isinstance(evaluation.get("comparison"), dict) else {}),
                "training_run_path": training.get("training_run_path", ""),
                "evaluation_training_run_path": evaluation.get("training_run_path", ""),
                "summary_path": evaluation.get("summary_path", ""),
            }
        )

    best = min(model_rows, key=lambda row: _finite_or_inf(row.get("comparison", {}).get("route_free_min_goal_distance")))
    summary_path = output_dir / "region_world_model_comparison_summary.json"
    metrics = _region_world_model_comparison_metrics(model_rows, best)
    training_run = write_training_run_record(
        output_dir,
        preset_id="region_world_model_comparison",
        status="completed",
        dataset_root=str(manifest_path.resolve()),
        adapter="beamng_episode_collection",
        sequence_id=task.task_id,
        artifact_path=str(summary_path.resolve()),
        artifact_type="world_model_comparison",
        metrics=metrics,
        history={key: [value] for key, value in _numeric_metric_items(metrics)},
        parameters={
            "collection_manifest_path": str(manifest_path.resolve()),
            "world_model_types": world_model_types,
            "task_path": str(Path(task_path).resolve()),
            "planner": request.planner,
            "planner_horizon": request.planner_horizon,
            "planner_samples": request.planner_samples,
            "planner_iterations": request.planner_iterations,
            "include_route_guided_baseline": bool(request.include_route_guided_baseline),
            "evaluation_agent": request.evaluation_agent,
            "evaluation_allow_reverse_recovery": bool(request.evaluation_allow_reverse_recovery),
        },
        summary={
            "task_path": str(Path(task_path).resolve()),
            "collection_manifest_path": str(manifest_path.resolve()),
            "best_world_model_type": best["world_model_type"],
            "best_comparison": best["comparison"],
            "model_summaries": [
                {
                    "world_model_type": row["world_model_type"],
                    "model_dir": row["model_dir"],
                    "comparison": row["comparison"],
                    "summary_path": row["summary_path"],
                }
                for row in model_rows
            ],
        },
    )
    payload: dict[str, Any] = {
        "status": "completed",
        "task": task.to_dict(),
        "task_path": str(Path(task_path).resolve()),
        "collection_manifest_path": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "models": model_rows,
        "best": best,
        "metrics": metrics,
        "training_run_path": training_run["path"],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def _region_world_model_comparison_metrics(model_rows: list[dict[str, Any]], best: dict[str, Any]) -> dict[str, Any]:
    best_comparison = best.get("comparison", {}) if isinstance(best.get("comparison"), dict) else {}
    metrics: dict[str, Any] = {
        "model_count": len(model_rows),
        "best_route_free_min_goal_distance": best_comparison.get("route_free_min_goal_distance"),
        "best_route_free_final_goal_distance": best_comparison.get("route_free_final_goal_distance"),
        "best_route_free_goal_success": bool(best_comparison.get("route_free_goal_success")),
        "best_route_guided_goal_success": bool(best_comparison.get("route_guided_goal_success")),
    }
    for row in model_rows:
        model_type = _safe_name(str(row.get("world_model_type") or "model")).lower()
        comparison = row.get("comparison") if isinstance(row.get("comparison"), dict) else {}
        for key, value in comparison.items():
            metrics[f"{model_type}_{key}"] = value
    return metrics


def _region_evaluation_planner_overrides(request: Any) -> dict[str, float]:
    candidates = {
        "goal_weight": request.planner_goal_weight,
        "progress_weight": request.planner_progress_weight,
        "risk_weight": request.planner_risk_weight,
        "heading_weight": request.planner_heading_weight,
    }
    overrides: dict[str, float] = {}
    for key, value in candidates.items():
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            overrides[key] = number
    return overrides


def save_manual_navigation_task(request: ManualNavigationTaskRequest) -> dict[str, Any]:
    region = [_coerce_point2(point, "region polygon") for point in request.region_polygon]
    if len(region) < 3:
        raise ValueError("Manual navigation task requires at least three region points.")
    start_pos = _coerce_point3(request.start_pos, "start position")
    goal_pos = _coerce_point2(request.goal_pos, "goal position")
    route = [_coerce_point2(point, "expert route") for point in request.expert_route]
    if not route:
        route = [(start_pos[0], start_pos[1]), goal_pos]
    task = NavigationRegionTask(
        task_id=_safe_name(request.task_id or "manual_region_nav"),
        map_id=str(request.map_id or request.level),
        level=str(request.level or "gridmap_v2"),
        region_polygon=region,
        start_pos=start_pos,
        start_yaw=float(request.start_yaw),
        goal_pos=goal_pos,
        goal_radius=float(request.goal_radius),
        expert_route=route,
        max_steps=int(request.max_steps),
        max_collision_count=int(request.max_collision_count),
        cost={
            "out_of_region_weight": float(request.out_of_region_weight),
            "boundary_weight": float(request.boundary_weight),
            "boundary_margin_m": float(request.boundary_margin_m),
        },
        beamng={
            "vehicle_model": str(request.vehicle_model or "pickup"),
            "camera_mode": str(request.camera_mode or "follow"),
            "draw_route": True,
            "drive_mode": str(request.collection_drive_mode or "ai_line"),
            "collection_drive_mode": str(request.collection_drive_mode or "ai_line"),
            "evaluation_drive_mode": str(request.evaluation_drive_mode or "manual"),
            "evaluation_route_mode": str(request.evaluation_route_mode or "expert"),
            "manual_control_is_adas": False,
            "ai_line_speed": float(request.ai_line_speed),
            "steps_per_action": int(request.steps_per_action),
            "weather": "sunny",
        },
    )
    if not task.contains_point((task.start_pos[0], task.start_pos[1])):
        raise ValueError("Start position must be inside the selected region.")
    if not task.contains_point(task.goal_pos):
        raise ValueError("Goal position must be inside the selected region.")

    output_path = Path(request.output_path or ROOT / "configs" / "tasks" / f"{task.task_id}.yaml")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to save task YAML files.") from exc
    output_path.write_text(yaml.safe_dump(task.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"status": "saved", "task_path": str(output_path.resolve()), "task": task.to_dict()}


def analyze_navigation_task(task_path: str | Path) -> dict[str, Any]:
    task = load_navigation_region_task(task_path)
    route = task.expert_route or [(task.start_pos[0], task.start_pos[1]), task.goal_pos]
    start_xy = (float(task.start_pos[0]), float(task.start_pos[1]))
    goal_xy = (float(task.goal_pos[0]), float(task.goal_pos[1]))
    segment_lengths = [
        math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
        for a, b in zip(route, route[1:], strict=False)
    ]
    first_heading = math.atan2(float(route[1][1]) - float(route[0][1]), float(route[1][0]) - float(route[0][0])) if len(route) >= 2 else math.nan
    route_in_region = all(task.contains_point((float(point[0]), float(point[1]))) for point in route)
    return {
        "task_id": task.task_id,
        "level": task.level,
        "start": [*start_xy, float(task.start_pos[2])],
        "goal": list(goal_xy),
        "goal_radius": float(task.goal_radius),
        "start_in_region": task.contains_point(start_xy),
        "goal_in_region": task.contains_point(goal_xy),
        "route_in_region": bool(route_in_region),
        "region_waypoint_count": len(task.region_polygon),
        "route_waypoint_count": len(route),
        "route_length_m": float(sum(segment_lengths)),
        "straight_line_m": float(math.hypot(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])),
        "max_route_segment_m": float(max(segment_lengths)) if segment_lengths else 0.0,
        "start_yaw": float(task.start_yaw),
        "first_route_heading": float(first_heading),
        "start_yaw_error": float(_wrap_angle(first_heading - task.start_yaw)) if math.isfinite(first_heading) else math.nan,
        "beamng_preview_recommended": True,
    }


def _navigation_preview_scenario(
    task: NavigationRegionTask,
    *,
    camera_mode: str = "topdown",
    camera_height_m: float = 150.0,
) -> dict[str, Any]:
    scenario = task.to_beamng_scenario(mode="evaluation")
    beamng = scenario.setdefault("metadata", {}).setdefault("beamng", {})
    beamng["drive_mode"] = "manual"
    beamng["preview_mode"] = True
    beamng["camera_mode"] = str(camera_mode or "topdown")
    beamng["camera_height_m"] = float(camera_height_m)
    beamng["draw_route"] = True
    beamng["draw_task_markers"] = True
    beamng["steps_per_action"] = 1
    beamng["route"] = [list(point) for point in (task.expert_route or [(task.start_pos[0], task.start_pos[1]), task.goal_pos])]
    return scenario


def preview_navigation_task_in_beamng(
    task_path: str | Path,
    *,
    vehicle: str = "configs/vehicles/ugv_medium.yaml",
    beamng_gfx: str = "vk",
    camera_mode: str = "topdown",
    camera_height_m: float = 150.0,
    hold_open_sec: float = 3.0,
) -> dict[str, Any]:
    task = load_navigation_region_task(task_path)
    analysis = analyze_navigation_task(task_path)
    scenario = _navigation_preview_scenario(task, camera_mode=camera_mode, camera_height_m=camera_height_m)
    result = run_episode(
        backend_name="beamng",
        scenario=scenario,
        agent_name="stop",
        seed=7,
        max_steps=1,
        record=False,
        backend_options={"connection": BeamNGConnectionConfig(gfx=beamng_gfx or None)},
        vehicle=vehicle,
        pre_run_hold_sec=2.0,
        step_delay_sec=0.0,
        post_run_hold_sec=max(0.0, float(hold_open_sec)),
        close_backend=False,
    )
    payload = result.to_dict()
    payload["analysis"] = analysis
    payload["preview"] = {
        "mode": "beamng_manual_static",
        "realtime": False,
        "camera_mode": str(camera_mode or "topdown"),
        "camera_height_m": float(camera_height_m),
        "route_drawn": True,
        "region_drawn": True,
        "start_goal_drawn": True,
    }
    return payload


class BeamNGNavigationPreviewSession:
    """Long-lived BeamNG preview session for live region/task editing."""

    def __init__(
        self,
        *,
        vehicle: str = "configs/vehicles/ugv_medium.yaml",
        beamng_gfx: str = "vk",
        enable_point_picker: bool = True,
    ) -> None:
        self.vehicle = vehicle
        self.beamng_gfx = beamng_gfx
        self.enable_point_picker = enable_point_picker
        self._backend: BeamNGBackend | None = None
        self._level: str | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _busy_payload(level: str | None) -> dict[str, Any]:
        return {
            "available": False,
            "message": "BeamNG preview is busy loading or updating.",
            "level": level,
        }

    def update(
        self,
        task_path: str | Path,
        *,
        camera_mode: str = "topdown",
        camera_height_m: float = 150.0,
    ) -> dict[str, Any]:
        with self._lock:
            task = load_navigation_region_task(task_path)
            analysis = analyze_navigation_task(task_path)
            scenario = _navigation_preview_scenario(task, camera_mode=camera_mode, camera_height_m=camera_height_m)
            beamng = scenario.get("metadata", {}).get("beamng", {})
            level = str(beamng.get("level", task.level))
            lifecycle = "updated"
            if self._backend is None or self._level != level:
                self.close()
                vehicle_config = load_vehicle_config(self.vehicle) if self.vehicle else None
                self._backend = BeamNGBackend(
                    connection=BeamNGConnectionConfig(
                        gfx=self.beamng_gfx or None,
                        enable_point_picker=self.enable_point_picker,
                    ),
                    vehicle_config=vehicle_config,
                )
                self._backend.reset(scenario)
                self._level = level
                lifecycle = "started"
            else:
                self._backend.update_navigation_preview(scenario)
            metrics = self._backend.get_metrics()
            return {
                "status": lifecycle,
                "analysis": analysis,
                "metrics": metrics,
                "preview": {
                    "mode": "beamng_manual_live",
                    "realtime": True,
                    "camera_mode": str(camera_mode or "topdown"),
                    "camera_height_m": float(camera_height_m),
                    "route_drawn": True,
                    "region_drawn": True,
                    "start_goal_drawn": True,
                },
            }

    def current_pose(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return self._busy_payload(self._level)
        try:
            if self._backend is None:
                return {
                    "available": False,
                    "message": "BeamNG preview session has not started.",
                    "level": self._level,
                }
            try:
                pose = dict(self._backend.get_current_vehicle_pose())
            except Exception as exc:
                return {
                    "available": False,
                    "message": str(exc),
                    "level": self._level,
                }
            pose.setdefault("available", True)
            pose.setdefault("level", self._level)
            return pose
        finally:
            self._lock.release()

    def consume_picker_pick(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            return self._busy_payload(self._level)
        try:
            if self._backend is None:
                return {
                    "available": False,
                    "message": "BeamNG preview session has not started.",
                    "level": self._level,
                }
            try:
                pick = dict(self._backend.consume_point_picker())
            except Exception as exc:
                return {
                    "available": False,
                    "message": str(exc),
                    "level": self._level,
                }
            pick.setdefault("available", False)
            pick.setdefault("level", self._level)
            return pick
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.close()
            self._backend = None
            self._level = None


def _route_free_region_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(scenario, default=str))
    metadata = cleaned.setdefault("metadata", {})
    beamng = metadata.setdefault("beamng", {})
    if isinstance(beamng, dict):
        beamng.pop("route", None)
        beamng["draw_route"] = False
        beamng["drive_mode"] = "manual"
        beamng["evaluation_route_mode"] = "none"
    task_metadata = metadata.get("task")
    if isinstance(task_metadata, dict):
        task_metadata.pop("expert_route", None)
    return cleaned


def _with_experience_corridor(scenario: dict[str, Any], experience_route: list[list[float]]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(scenario, default=str))
    route = _normalize_point_route(experience_route)
    if len(route) < 2:
        return cleaned
    metadata = cleaned.setdefault("metadata", {})
    task_metadata = metadata.setdefault("task", {})
    if isinstance(task_metadata, dict):
        task_metadata.pop("expert_route", None)
        task_metadata["experience_route"] = route
    beamng = metadata.setdefault("beamng", {})
    if isinstance(beamng, dict):
        beamng.pop("route", None)
        beamng["draw_route"] = False
        beamng["drive_mode"] = "manual"
        beamng["evaluation_route_mode"] = "none"
    return cleaned


def _normalize_point_route(points: list[list[float]] | list[tuple[float, float]]) -> list[list[float]]:
    route: list[list[float]] = []
    for point in points:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            route.append([x, y])
    return route


def _route_guided_region_scenario(task: NavigationRegionTask) -> dict[str, Any]:
    scenario = json.loads(json.dumps(task.to_beamng_scenario(mode="evaluation"), default=str))
    metadata = scenario.setdefault("metadata", {})
    beamng = metadata.setdefault("beamng", {})
    route = task.expert_route or [(task.start_pos[0], task.start_pos[1]), task.goal_pos]
    beamng["route"] = [list(point) for point in route]
    beamng["draw_route"] = True
    beamng["drive_mode"] = "manual"
    beamng["evaluation_route_mode"] = "expert"
    return scenario


def _collection_region_scenario(task: NavigationRegionTask, *, strategy: str) -> dict[str, Any]:
    normalized = str(strategy or "region_explorer").strip().lower()
    if normalized in {"route_aware", "route-aware", "curriculum", "route_curriculum"}:
        return _route_guided_region_scenario(task)
    return _route_free_region_scenario(task.to_beamng_scenario(mode="evaluation"))


def _collection_rollout_scenario(
    task: NavigationRegionTask,
    base_scenario: dict[str, Any],
    *,
    strategy: str,
    rollout_index: int,
    rollout_count: int,
    multi_start: bool,
    lateral_m: float,
    seed: int,
) -> dict[str, Any]:
    scenario = json.loads(json.dumps(base_scenario, default=str))
    normalized = str(strategy or "region_explorer").strip().lower()
    if not multi_start or rollout_index <= 0 or normalized not in {"route_aware", "route-aware", "curriculum", "route_curriculum"}:
        return scenario
    route = task.expert_route
    if len(route) < 2:
        return scenario
    indices = _route_multi_start_indices(task=task, rollout_count=max(1, int(rollout_count)))
    route_index = indices[min(max(0, int(rollout_index)), len(indices) - 1)]
    point = route[route_index]
    previous_point = route[max(0, route_index - 1)]
    next_point = route[min(len(route) - 1, route_index + 1)]
    dx = float(next_point[0]) - float(previous_point[0])
    dy = float(next_point[1]) - float(previous_point[1])
    yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else float(task.start_yaw)
    start_xy = _route_multi_start_lateral_point(task, point, previous_point, next_point, lateral_m=max(0.0, float(lateral_m)), seed=seed)
    metadata = scenario.setdefault("metadata", {})
    beamng = metadata.setdefault("beamng", {})
    start_pos = [float(start_xy[0]), float(start_xy[1]), float(task.start_pos[2])]
    vehicle_start = beamng.setdefault("vehicle_start", {})
    vehicle_start["pos"] = start_pos
    vehicle_start["yaw"] = float(yaw)
    vehicle_start["original_yaw"] = float(task.start_yaw)
    vehicle_start["yaw_source"] = "collection_multi_start_route"
    vehicle_start["rot_quat"] = _beamng_yaw_to_quat(yaw)
    remaining_route = [list(item) for item in route[route_index:]]
    if len(remaining_route) < 2:
        remaining_route = [list(point), list(task.goal_pos)]
    beamng["route"] = remaining_route
    task_metadata = metadata.get("task")
    if isinstance(task_metadata, dict):
        start_pose = task_metadata.setdefault("start_pose", {})
        if isinstance(start_pose, dict):
            start_pose["pos"] = start_pos
            start_pose["yaw"] = float(yaw)
        task_metadata["expert_route"] = remaining_route
    return scenario


def _route_multi_start_indices(*, task: NavigationRegionTask, rollout_count: int) -> list[int]:
    route_count = len(task.expert_route)
    if route_count <= 1:
        return [0]
    if rollout_count <= 1:
        return [0]
    min_goal_distance = max(float(task.goal_radius) + 2.0, float(task.goal_radius) * 1.25)
    eligible = [
        index
        for index, point in enumerate(task.expert_route)
        if math.hypot(float(point[0]) - float(task.goal_pos[0]), float(point[1]) - float(task.goal_pos[1])) > min_goal_distance
    ]
    if not eligible:
        eligible = [0]
    if len(eligible) == 1:
        return [eligible[0] for _ in range(rollout_count)]
    last = len(eligible) - 1
    indices: list[int] = []
    for rollout_index in range(rollout_count):
        if rollout_index <= 0:
            eligible_index = 0
        else:
            eligible_index = int(math.ceil(rollout_index * last / max(1, rollout_count - 1)))
        indices.append(eligible[min(last, max(0, eligible_index))])
    return indices


def _route_multi_start_lateral_point(
    task: NavigationRegionTask,
    point: tuple[float, float],
    previous_point: tuple[float, float],
    next_point: tuple[float, float],
    *,
    lateral_m: float,
    seed: int,
) -> tuple[float, float]:
    if lateral_m <= 0.0:
        return (float(point[0]), float(point[1]))
    dx = float(next_point[0]) - float(previous_point[0])
    dy = float(next_point[1]) - float(previous_point[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return (float(point[0]), float(point[1]))
    rng = np.random.default_rng(seed)
    lateral = float(rng.uniform(-lateral_m, lateral_m))
    candidate = (float(point[0]) - dy / length * lateral, float(point[1]) + dx / length * lateral)
    return candidate if task.contains_point(candidate) else (float(point[0]), float(point[1]))


def _beamng_yaw_to_quat(yaw: float) -> list[float]:
    beamng_yaw = -float(yaw) - math.pi * 0.5
    half = beamng_yaw * 0.5
    return [0.0, 0.0, round(math.sin(half), 6), round(math.cos(half), 6)]


def _navigation_task_from_collection_manifest(collection: dict[str, Any], manifest_path: Path) -> NavigationRegionTask:
    raw_task = collection.get("task")
    if isinstance(raw_task, dict) and raw_task:
        return NavigationRegionTask.from_dict(raw_task)
    task_path = str(collection.get("task_path") or "").strip()
    if task_path:
        return load_navigation_region_task(task_path)
    raise ValueError(f"Collection manifest has no task data: {manifest_path}")


def _collection_coverage_metrics(task: NavigationRegionTask, episode_paths: list[str], *, grid_size: int) -> dict[str, Any]:
    grid = max(2, int(grid_size))
    xs = [float(point[0]) for point in task.region_polygon]
    ys = [float(point[1]) for point in task.region_polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    total_cells = _coverage_total_cells(task, grid, min_x, min_y, span_x, span_y)
    visited: set[tuple[int, int]] = set()
    for episode_path in episode_paths:
        for row in load_episode_trace(episode_path):
            x = _float_or_nan(row.get("x"))
            y = _float_or_nan(row.get("y"))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            point = (float(x), float(y))
            if not _point_in_or_on_task_region(task, point):
                continue
            col = min(grid - 1, max(0, int((point[0] - min_x) / span_x * grid)))
            row_index = min(grid - 1, max(0, int((point[1] - min_y) / span_y * grid)))
            visited.add((col, row_index))
    cell_count = len(visited)
    ratio = float(cell_count / total_cells) if total_cells > 0 else math.nan
    return {
        "grid_size": grid,
        "cell_count": cell_count,
        "total_cells": total_cells,
        "ratio": ratio,
        "visited_cells": [[int(col), int(row)] for col, row in sorted(visited)],
    }


def _collection_route_metrics(task: NavigationRegionTask, episode_paths: list[str], *, route_radius_m: float | None = None) -> dict[str, Any]:
    route = task.expert_route or [(task.start_pos[0], task.start_pos[1]), task.goal_pos]
    radius = float(route_radius_m if route_radius_m is not None else max(task.goal_radius, 8.0))
    covered_indices: set[int] = set()
    goal_episode_hits = 0
    valid_episode_count = 0
    for episode_path in episode_paths:
        trace = load_episode_trace(episode_path)
        if not trace:
            continue
        valid_episode_count += 1
        hit_goal = False
        for row in trace:
            x = _float_or_nan(row.get("x"))
            y = _float_or_nan(row.get("y"))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            point = (float(x), float(y))
            for index, route_point in enumerate(route):
                if math.hypot(point[0] - float(route_point[0]), point[1] - float(route_point[1])) <= radius:
                    covered_indices.add(index)
            if math.hypot(point[0] - task.goal_pos[0], point[1] - task.goal_pos[1]) <= task.goal_radius:
                hit_goal = True
        if hit_goal:
            goal_episode_hits += 1
    route_count = len(route)
    route_ratio = float(len(covered_indices) / route_count) if route_count > 0 else math.nan
    goal_zone_coverage = float(goal_episode_hits / valid_episode_count) if valid_episode_count > 0 else 0.0
    return {
        "route_radius_m": radius,
        "route_waypoint_count": route_count,
        "route_covered_waypoint_count": len(covered_indices),
        "route_covered_indices": [int(index) for index in sorted(covered_indices)],
        "route_coverage_ratio": route_ratio,
        "goal_zone_episode_count": goal_episode_hits,
        "episode_count": valid_episode_count,
        "goal_zone_coverage": goal_zone_coverage,
    }


def _experience_route_from_episode_traces(
    task: NavigationRegionTask,
    episode_paths: list[str],
    *,
    min_spacing_m: float,
    max_points: int,
) -> list[list[float]]:
    episode_routes: list[tuple[float, list[tuple[float, float]]]] = []
    for episode_path in episode_paths:
        route: list[tuple[float, float]] = []
        for row in load_episode_trace(episode_path):
            x = _float_or_nan(row.get("x"))
            y = _float_or_nan(row.get("y"))
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            point = (float(x), float(y))
            if _point_in_or_on_task_region(task, point):
                route.append(point)
        if route:
            episode_routes.append((_task_goal_progress(task, route[0]), route))
    if not episode_routes:
        return []
    spacing = max(0.25, float(min_spacing_m))
    limit = max(2, int(max_points))
    merged: list[list[float]] = []

    def append_point(point: tuple[float, float], *, force: bool = False) -> None:
        if not force and merged:
            previous = merged[-1]
            if math.hypot(float(point[0]) - previous[0], float(point[1]) - previous[1]) < spacing:
                return
        if merged and math.hypot(float(point[0]) - merged[-1][0], float(point[1]) - merged[-1][1]) <= 1e-6:
            return
        merged.append([float(point[0]), float(point[1])])

    append_point((float(task.start_pos[0]), float(task.start_pos[1])), force=True)
    for _, route in sorted(episode_routes, key=lambda item: item[0]):
        for point in route:
            if len(merged) >= limit - 1:
                break
            append_point(point)
        if len(merged) >= limit - 1:
            break
    append_point((float(task.goal_pos[0]), float(task.goal_pos[1])), force=True)
    return merged


def _experience_route_from_model_metadata(
    task: NavigationRegionTask,
    model_path: str | Path,
    *,
    min_spacing_m: float,
    max_points: int,
) -> list[list[float]]:
    path = Path(model_path)
    model_json = path / "model.json" if path.is_dir() else path
    try:
        model_data = _read_json(model_json)
    except (OSError, json.JSONDecodeError):
        return []
    config = model_data.get("config") if isinstance(model_data.get("config"), dict) else {}
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    raw_paths = metadata.get("episode_paths")
    if not isinstance(raw_paths, list):
        raw_path = metadata.get("episode_path")
        raw_paths = [raw_path] if raw_path else []
    episode_paths = [str(path) for path in raw_paths if str(path or "").strip()]
    if not episode_paths:
        return []
    return _experience_route_from_episode_traces(
        task,
        episode_paths,
        min_spacing_m=min_spacing_m,
        max_points=max_points,
    )


def _task_goal_progress(task: NavigationRegionTask, point: tuple[float, float]) -> float:
    start_distance = math.hypot(float(task.start_pos[0]) - float(task.goal_pos[0]), float(task.start_pos[1]) - float(task.goal_pos[1]))
    if start_distance <= 1e-9:
        return 1.0
    distance = math.hypot(float(point[0]) - float(task.goal_pos[0]), float(point[1]) - float(task.goal_pos[1]))
    return max(0.0, min(1.0, (start_distance - distance) / start_distance))


def _coverage_total_cells(
    task: NavigationRegionTask,
    grid: int,
    min_x: float,
    min_y: float,
    span_x: float,
    span_y: float,
) -> int:
    count = 0
    for col in range(grid):
        for row in range(grid):
            point = (min_x + (col + 0.5) / grid * span_x, min_y + (row + 0.5) / grid * span_y)
            if task.contains_point(point):
                count += 1
    return max(1, count)


def _point_in_or_on_task_region(task: NavigationRegionTask, point: tuple[float, float]) -> bool:
    if task.contains_point(point):
        return True
    vertices = task.region_polygon
    for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=False):
        if _point_segment_distance(point, start, end) <= 1e-6:
            return True
    return False


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    closest = (sx + t * dx, sy + t * dy)
    return math.hypot(px - closest[0], py - closest[1])


def _episode_trace_to_dataset_sequence(episode_path: str | Path, task: NavigationRegionTask) -> DatasetSequence:
    rows = load_episode_trace(episode_path)
    frames: list[DatasetFrame] = []
    for index, row in enumerate(rows):
        x = _float_or_nan(row.get("x"))
        y = _float_or_nan(row.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        frames.append(
            DatasetFrame(
                frame_id=f"{index:06d}",
                timestamp=float(row.get("timestamp") if row.get("timestamp") is not None else index),
                vehicle_state=VehicleState(
                    x=x,
                    y=y,
                    z=_finite_or_default(row.get("z"), task.start_pos[2]),
                    yaw=_finite_or_default(row.get("yaw"), 0.0),
                    pitch=_finite_or_default(row.get("pitch"), 0.0),
                    roll=_finite_or_default(row.get("roll"), 0.0),
                    speed=_finite_or_default(row.get("speed"), 0.0),
                ),
                action=Action(
                    steer=_finite_or_default(row.get("steer"), 0.0),
                    throttle=_finite_or_default(row.get("throttle"), 0.0),
                    brake=_finite_or_default(row.get("brake"), 0.0),
                    gear=_int_or_none(row.get("gear")),
                ),
                metadata={"source_step_index": row.get("step_index")},
            )
        )
    if len(frames) < 2:
        raise ValueError("Self-supervised world-model training requires at least two recorded states.")
    return DatasetSequence(
        dataset_id=f"beamng_region_{task.task_id}",
        dataset_type="beamng_episode",
        sequence_id=Path(episode_path).name,
        root=str(Path(episode_path).resolve()),
        frames=frames,
        goal=task.goal_pos,
        metadata={
            "task_id": task.task_id,
            "source": "beamng_region_self_supervised",
            "task_start_pos": [float(task.start_pos[0]), float(task.start_pos[1])],
            "task_goal_pos": [float(task.goal_pos[0]), float(task.goal_pos[1])],
        },
    )


def _looks_like_beamng_reconnect_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "BNGDisconnectedError" in text or "Connecting to the simulator failed" in text or "ConnectionResetError" in text


def _run_region_beamng_episode_with_reconnect_retry(**kwargs: Any) -> dict[str, Any]:
    try:
        return _run_region_beamng_episode(**kwargs)
    except Exception as exc:
        if not _looks_like_beamng_reconnect_error(exc):
            raise
        time.sleep(6.0)
        return _run_region_beamng_episode(**kwargs)


def _run_region_beamng_episode(
    *,
    scenario: dict[str, Any],
    vehicle: str,
    max_steps: int,
    seed: int,
    world_model_type: str,
    world_model_path: str,
    planner: str,
    planner_horizon: int,
    planner_samples: int,
    planner_iterations: int,
    record: bool,
    beamng_gfx: str,
    pre_run_hold_sec: float,
    step_delay_sec: float,
    post_run_hold_sec: float,
    close_beamng: bool,
    agent_name: str = "route_world_model",
    agent_options: dict[str, Any] | None = None,
    algorithm_name: str = "",
    algorithm_model_path: str = "",
    planner_config_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    planner_config = {
        "horizon": max(1, int(planner_horizon)),
        "num_samples": max(4, int(planner_samples)),
        "iterations": max(1, int(planner_iterations)),
    }
    planner_config.update(dict(planner_config_overrides or {}))
    if agent_name == "model_mpc":
        agent_options_payload = {
            "world_model_name": world_model_type,
            "world_model_path": world_model_path,
            "algorithm_name": algorithm_name,
            "algorithm_model_path": algorithm_model_path or world_model_path,
            "planner_config": planner_config,
        }
        route = scenario.get("metadata", {}).get("beamng", {}).get("route", [])
        if route:
            agent_options_payload["route"] = route
    elif agent_name == "world_model_direct":
        agent_options_payload = {
            "world_model_name": world_model_type,
            "world_model_path": world_model_path,
            "planner_name": planner or "navigation_mpc",
            "planner_config": planner_config,
        }
    elif agent_name == "region_explorer":
        agent_options_payload = {}
    else:
        agent_options_payload = {
            "world_model_name": world_model_type,
            "execution_mode": "model_guided_route_tracker",
        }
        if world_model_path:
            agent_options_payload["world_model_path"] = world_model_path
        if planner:
            agent_options_payload["planner_name"] = planner
            agent_options_payload["planner_config"] = planner_config
    if agent_options:
        agent_options_payload.update(dict(agent_options))
    result = run_episode(
        backend_name="beamng",
        scenario=scenario,
        agent_name=agent_name,
        seed=seed,
        max_steps=max_steps,
        record=record,
        output_root=DEFAULT_OUTPUT_ROOT,
        backend_options={"connection": BeamNGConnectionConfig(gfx=beamng_gfx or None)},
        agent_options=agent_options_payload,
        vehicle=vehicle,
        pre_run_hold_sec=pre_run_hold_sec,
        step_delay_sec=step_delay_sec,
        post_run_hold_sec=post_run_hold_sec,
        close_backend=close_beamng,
    )
    payload = result.to_dict()
    payload["region_navigation"] = {
        "world_model_type": world_model_type,
        "world_model_path": world_model_path or None,
        "evaluation_agent": agent_name,
        "algorithm_name": algorithm_name or None,
        "algorithm_model_path": algorithm_model_path or None,
        "planner": planner or None,
        "scenario_id": scenario.get("scenario_id"),
        "beamng_gfx": beamng_gfx,
    }
    return payload


def _episode_behavior_counts(trace: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, int]:
    reverse_count = 0
    stuck_recovery_count = 0
    for row in trace:
        gear = _int_or_none(row.get("gear"))
        diagnostics = row.get("agent_diagnostics") if isinstance(row.get("agent_diagnostics"), dict) else {}
        executed = diagnostics.get("executed_action") if isinstance(diagnostics.get("executed_action"), dict) else {}
        executed_gear = _int_or_none(executed.get("gear")) if executed else None
        if (gear is not None and gear < 0) or (executed_gear is not None and executed_gear < 0):
            reverse_count += 1
        if bool(diagnostics.get("stuck_recovery")):
            stuck_recovery_count += 1
    final_diagnostics = metrics.get("agent_diagnostics") if isinstance(metrics.get("agent_diagnostics"), dict) else {}
    if bool(final_diagnostics.get("stuck_recovery")) and stuck_recovery_count == 0:
        stuck_recovery_count = 1
    return {"reverse_count": int(reverse_count), "stuck_recovery_count": int(stuck_recovery_count)}


def _region_evaluation_comparison(
    *,
    route_free_acceptance: dict[str, Any],
    route_free_evaluation: dict[str, Any],
    route_guided_acceptance: dict[str, Any] | None,
    route_guided_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    route_free_metrics = route_free_evaluation.get("metrics", {}) if isinstance(route_free_evaluation.get("metrics"), dict) else {}
    comparison: dict[str, Any] = {
        "route_free_goal_success": bool(route_free_acceptance.get("goal_success")),
        "route_free_goal_reached": bool(route_free_acceptance.get("goal_reached")),
        "route_free_min_goal_distance": route_free_acceptance.get("min_goal_distance"),
        "route_free_final_goal_distance": route_free_acceptance.get("final_goal_distance"),
        "route_free_collision_count": int(route_free_acceptance.get("collision_count", 0) or 0),
        "route_free_distance_traveled": route_free_acceptance.get("distance_traveled", route_free_metrics.get("horizontal_distance_traveled")),
        "route_free_stuck_recovery_count": int(route_free_acceptance.get("stuck_recovery_count", 0) or 0),
        "route_free_reverse_count": int(route_free_acceptance.get("reverse_count", 0) or 0),
    }
    if route_guided_acceptance is not None and route_guided_evaluation is not None:
        route_guided_metrics = route_guided_evaluation.get("metrics", {}) if isinstance(route_guided_evaluation.get("metrics"), dict) else {}
        comparison.update(
            {
                "route_guided_goal_success": bool(route_guided_acceptance.get("goal_success")),
                "route_guided_goal_reached": bool(route_guided_acceptance.get("goal_reached")),
                "route_guided_min_goal_distance": route_guided_acceptance.get("min_goal_distance"),
                "route_guided_final_goal_distance": route_guided_acceptance.get("final_goal_distance"),
                "route_guided_collision_count": int(route_guided_acceptance.get("collision_count", 0) or 0),
                "route_guided_distance_traveled": route_guided_acceptance.get(
                    "distance_traveled", route_guided_metrics.get("horizontal_distance_traveled")
                ),
                "route_guided_stuck_recovery_count": int(route_guided_acceptance.get("stuck_recovery_count", 0) or 0),
                "route_guided_reverse_count": int(route_guided_acceptance.get("reverse_count", 0) or 0),
            }
        )
    return comparison


def _write_region_trajectory_svg(task: NavigationRegionTask, output_path: Path, *, traces: dict[str, list[dict[str, Any]]]) -> None:
    points: list[tuple[float, float]] = list(task.region_polygon)
    points.extend(task.expert_route or [])
    points.append((float(task.start_pos[0]), float(task.start_pos[1])))
    points.append((float(task.goal_pos[0]), float(task.goal_pos[1])))
    for trace in traces.values():
        for row in trace:
            x = _float_or_nan(row.get("x"))
            y = _float_or_nan(row.get("y"))
            if math.isfinite(x) and math.isfinite(y):
                points.append((float(x), float(y)))
    if not points:
        return
    width = 900.0
    height = 640.0
    margin = 44.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((width - margin * 2.0) / span_x, (height - margin * 2.0) / span_y)

    def project(point: tuple[float, float]) -> tuple[float, float]:
        return (
            margin + (point[0] - min_x) * scale,
            height - margin - (point[1] - min_y) * scale,
        )

    def polyline(raw_points: list[tuple[float, float]]) -> str:
        return " ".join(f"{project(point)[0]:.1f},{project(point)[1]:.1f}" for point in raw_points)

    colors = {
        "collection": "#8aa2ff",
        "route_free": "#35d399",
        "route_guided": "#c084fc",
        "expert_route": "#f59e0b",
    }
    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="640" viewBox="0 0 900 640">',
        '<rect width="900" height="640" fill="#08131c"/>',
        '<text x="24" y="32" fill="#dbeafe" font-family="Arial" font-size="18">BeamNG region trajectory comparison</text>',
    ]
    if len(task.region_polygon) >= 3:
        lines.append(
            f'<polygon points="{polyline(task.region_polygon)}" fill="#0f2230" stroke="#7dd3fc" stroke-width="2" fill-opacity="0.55"/>'
        )
    route = task.expert_route or [(task.start_pos[0], task.start_pos[1]), task.goal_pos]
    if len(route) >= 2:
        lines.append(f'<polyline points="{polyline(route)}" fill="none" stroke="{colors["expert_route"]}" stroke-width="4" stroke-dasharray="10 7"/>')
    for label, trace in traces.items():
        trace_points = [
            (float(row["x"]), float(row["y"]))
            for row in trace
            if math.isfinite(_float_or_nan(row.get("x"))) and math.isfinite(_float_or_nan(row.get("y")))
        ]
        if len(trace_points) >= 2:
            color = colors.get(label, "#e5e7eb")
            lines.append(f'<polyline points="{polyline(trace_points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        elif len(trace_points) == 1:
            x, y = project(trace_points[0])
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors.get(label, "#e5e7eb")}"/>')
    start_x, start_y = project((float(task.start_pos[0]), float(task.start_pos[1])))
    goal_x, goal_y = project((float(task.goal_pos[0]), float(task.goal_pos[1])))
    lines.extend(
        [
            f'<circle cx="{start_x:.1f}" cy="{start_y:.1f}" r="7" fill="#22c55e"/>',
            f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="9" fill="#ef4444"/>',
            f'<circle cx="{goal_x:.1f}" cy="{goal_y:.1f}" r="{max(3.0, task.goal_radius * scale):.1f}" fill="none" stroke="#ef4444" stroke-width="2" stroke-opacity="0.5"/>',
            f'<text x="{start_x + 10.0:.1f}" y="{start_y - 8.0:.1f}" fill="#bbf7d0" font-family="Arial" font-size="12">start</text>',
            f'<text x="{goal_x + 12.0:.1f}" y="{goal_y - 10.0:.1f}" fill="#fecaca" font-family="Arial" font-size="12">goal</text>',
        ]
    )
    legend_y = 58
    for label in ["expert_route", *traces.keys()]:
        color = colors.get(label, "#e5e7eb")
        lines.append(f'<rect x="24" y="{legend_y}" width="18" height="4" fill="{color}"/>')
        lines.append(f'<text x="50" y="{legend_y + 5}" fill="#cbd5e1" font-family="Arial" font-size="13">{escape(label)}</text>')
        legend_y += 22
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_region_self_supervised_trajectory_plot(
    task: NavigationRegionTask,
    output_dir: Path,
    *,
    episode_paths: list[str],
    evaluation: dict[str, Any] | None = None,
) -> str:
    traces: dict[str, list[dict[str, Any]]] = {}
    collection_trace: list[dict[str, Any]] = []
    for episode_path in episode_paths:
        collection_trace.extend(load_episode_trace(episode_path))
    if collection_trace:
        traces["collection"] = collection_trace
    if evaluation:
        route_free_trace = load_episode_trace(str(evaluation.get("episode_path") or ""))
        if route_free_trace:
            traces["route_free"] = route_free_trace
    if not traces:
        return ""
    path = output_dir / "region_self_supervised_trajectory.svg"
    _write_region_trajectory_svg(task, path, traces=traces)
    return str(path.resolve())


def _navigation_acceptance(evaluation: dict[str, Any], task: NavigationRegionTask) -> dict[str, Any]:
    trace = load_episode_trace(evaluation.get("episode_path", ""))
    final = trace[-1] if trace else {}
    x = _float_or_nan(final.get("x"))
    y = _float_or_nan(final.get("y"))
    final_distance = math.hypot(x - task.goal_pos[0], y - task.goal_pos[1]) if math.isfinite(x) and math.isfinite(y) else math.nan
    in_region = task.contains_point((x, y)) if math.isfinite(x) and math.isfinite(y) else False
    min_distance = math.inf
    min_step: int | None = None
    reached_in_region = False
    for row in trace:
        row_x = _float_or_nan(row.get("x"))
        row_y = _float_or_nan(row.get("y"))
        if not math.isfinite(row_x) or not math.isfinite(row_y):
            continue
        distance = math.hypot(row_x - task.goal_pos[0], row_y - task.goal_pos[1])
        if distance < min_distance:
            min_distance = distance
            min_step = int(row.get("step_index") or 0)
        if distance <= task.goal_radius and task.contains_point((row_x, row_y)):
            reached_in_region = True
    if not math.isfinite(min_distance):
        min_distance = math.nan
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation.get("metrics"), dict) else {}
    collision_count = int(metrics.get("collision_count", 0) or 0)
    drive_mode = str(metrics.get("drive_mode", "manual")).lower()
    model_controlled = drive_mode != "ai_line"
    goal_reached = math.isfinite(min_distance) and min_distance <= task.goal_radius
    final_goal_reached = math.isfinite(final_distance) and final_distance <= task.goal_radius and bool(in_region)
    throttles = [_float_or_nan(row.get("throttle")) for row in trace]
    throttles = [value for value in throttles if math.isfinite(value)]
    steers = [_float_or_nan(row.get("steer")) for row in trace]
    steers = [value for value in steers if math.isfinite(value)]
    behavior_counts = _episode_behavior_counts(trace, metrics)
    distance_traveled = _float_or_nan(metrics.get("horizontal_distance_traveled"))
    if not math.isfinite(distance_traveled):
        distance_traveled = _trajectory_length(trace)
    return {
        "goal_success": bool(final_goal_reached and reached_in_region and model_controlled and collision_count <= task.max_collision_count),
        "goal_reached": bool(goal_reached),
        "final_goal_reached": bool(final_goal_reached),
        "final_goal_distance": final_distance,
        "min_goal_distance": min_distance,
        "min_goal_step": min_step,
        "goal_radius": task.goal_radius,
        "final_in_region": bool(in_region),
        "reached_in_region": bool(reached_in_region),
        "model_controlled": bool(model_controlled),
        "drive_mode": drive_mode,
        "route_waypoint_count": int(metrics.get("route_waypoint_count", 0) or 0),
        "mean_throttle": float(sum(throttles) / len(throttles)) if throttles else math.nan,
        "mean_abs_steer": float(sum(abs(value) for value in steers) / len(steers)) if steers else math.nan,
        "collision_count": collision_count,
        "distance_traveled": distance_traveled,
        "stuck_recovery_count": behavior_counts["stuck_recovery_count"],
        "reverse_count": behavior_counts["reverse_count"],
        "max_collision_count": task.max_collision_count,
    }


def _demo_acceptance_run_summary(run_index: int, seed: int, payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), dict) else {}
    metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
    trace = load_episode_trace(evaluation.get("episode_path", ""))
    return {
        "run_index": int(run_index),
        "seed": int(seed),
        "status": str(payload.get("status") or "completed"),
        "goal_success": bool(acceptance.get("goal_success")),
        "goal_reached": bool(acceptance.get("goal_reached")),
        "final_goal_reached": bool(acceptance.get("final_goal_reached")),
        "collision_count": int(acceptance.get("collision_count", metrics.get("collision_count", 0)) or 0),
        "final_distance": _float_or_nan(acceptance.get("final_goal_distance")),
        "min_distance": _float_or_nan(acceptance.get("min_goal_distance")),
        "trajectory_length_m": _trajectory_length(trace),
        "trajectory_step_count": len(trace),
        "average_speed": _average_trace_speed(trace),
        "recovery_triggered": _recovery_triggered(trace, metrics),
        "episode_path": str(evaluation.get("episode_path") or ""),
        "summary_path": str(payload.get("summary_path") or ""),
    }


def _demo_acceptance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "goal_reached": False,
            "collision_count": 0,
            "final_distance": math.nan,
            "trajectory_length_m": math.nan,
            "average_speed": math.nan,
            "recovery_triggered": False,
        }
    final_distances = [_float_or_nan(row.get("final_distance")) for row in rows]
    trajectory_lengths = [_float_or_nan(row.get("trajectory_length_m")) for row in rows]
    average_speeds = [_float_or_nan(row.get("average_speed")) for row in rows]
    return {
        "goal_reached": all(bool(row.get("goal_reached")) for row in rows),
        "goal_success": all(bool(row.get("goal_success")) for row in rows),
        "collision_count": sum(int(row.get("collision_count", 0) or 0) for row in rows),
        "final_distance": _mean_finite(final_distances),
        "best_final_distance": _min_finite(final_distances),
        "trajectory_length_m": _mean_finite(trajectory_lengths),
        "average_speed": _mean_finite(average_speeds),
        "recovery_triggered": any(bool(row.get("recovery_triggered")) for row in rows),
    }


def _trajectory_length(trace: list[dict[str, Any]]) -> float:
    distance = 0.0
    previous: tuple[float, float] | None = None
    for row in trace:
        x = _float_or_nan(row.get("x"))
        y = _float_or_nan(row.get("y"))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        current = (x, y)
        if previous is not None:
            distance += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return float(distance) if previous is not None else math.nan


def _average_trace_speed(trace: list[dict[str, Any]]) -> float:
    speeds = [_float_or_nan(row.get("speed")) for row in trace]
    return _mean_finite(speeds)


def _recovery_triggered(trace: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
    diagnostics = metrics.get("agent_diagnostics") if isinstance(metrics.get("agent_diagnostics"), dict) else {}
    if bool(diagnostics.get("stuck_recovery")):
        return True
    for row in trace:
        row_diagnostics = row.get("agent_diagnostics") if isinstance(row.get("agent_diagnostics"), dict) else {}
        if bool(row_diagnostics.get("stuck_recovery")):
            return True
    return False


def _mean_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _min_finite(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(min(finite)) if finite else math.nan


def _collection_quality_gate(
    task: NavigationRegionTask,
    collection_acceptance: dict[str, Any],
    *,
    min_progress_ratio: float,
    route_coverage_ratio: float | None = None,
    min_route_coverage_ratio: float = 0.0,
    goal_zone_coverage: float | None = None,
    min_goal_zone_coverage: float = 0.0,
    max_collection_min_goal_distance_m: float = 0.0,
    unique_region_cells: int | None = None,
    min_unique_region_cells: int = 0,
) -> dict[str, Any]:
    required = max(0.0, float(min_progress_ratio))
    required_route = max(0.0, float(min_route_coverage_ratio))
    required_goal_zone = max(0.0, float(min_goal_zone_coverage))
    required_goal_distance = max(0.0, float(max_collection_min_goal_distance_m))
    required_unique_cells = max(0, int(min_unique_region_cells))
    start_distance = math.hypot(task.start_pos[0] - task.goal_pos[0], task.start_pos[1] - task.goal_pos[1])
    min_distance = _float_or_nan(collection_acceptance.get("min_goal_distance"))
    if not math.isfinite(min_distance) or start_distance <= 1e-9:
        progress_ratio = 0.0
    else:
        progress_ratio = max(0.0, min(1.0, (start_distance - min_distance) / start_distance))
    goal_reached = bool(collection_acceptance.get("goal_reached"))
    route_ratio = _float_or_nan(route_coverage_ratio)
    if not math.isfinite(route_ratio):
        route_ratio = 0.0
    goal_zone_ratio = _float_or_nan(goal_zone_coverage)
    if not math.isfinite(goal_zone_ratio):
        goal_zone_ratio = 0.0
    progress_passed = required <= 0.0 or goal_reached or progress_ratio >= required
    route_passed = required_route <= 0.0 or route_ratio >= required_route
    goal_zone_passed = required_goal_zone <= 0.0 or goal_zone_ratio >= required_goal_zone
    goal_distance_passed = required_goal_distance <= 0.0 or goal_reached or (math.isfinite(min_distance) and min_distance <= required_goal_distance)
    unique_cells = max(0, int(unique_region_cells or 0))
    unique_cells_passed = required_unique_cells <= 0 or unique_cells >= required_unique_cells
    passed = bool(progress_passed and route_passed and goal_zone_passed and goal_distance_passed and unique_cells_passed)
    if passed:
        reason = "passed"
    elif not progress_passed:
        reason = "collection_goal_progress_below_threshold"
    elif not route_passed:
        reason = "collection_route_coverage_below_threshold"
    elif not goal_zone_passed:
        reason = "collection_goal_zone_coverage_below_threshold"
    elif not goal_distance_passed:
        reason = "collection_min_goal_distance_above_threshold"
    else:
        reason = "collection_unique_region_cells_below_threshold"
    return {
        "passed": bool(passed),
        "reason": reason,
        "progress_ratio": float(progress_ratio),
        "required_progress_ratio": float(required),
        "route_coverage_ratio": float(route_ratio),
        "required_route_coverage_ratio": float(required_route),
        "goal_zone_coverage": float(goal_zone_ratio),
        "required_goal_zone_coverage": float(required_goal_zone),
        "required_collection_min_goal_distance_m": float(required_goal_distance),
        "unique_region_cells": int(unique_cells),
        "required_unique_region_cells": int(required_unique_cells),
        "start_goal_distance": float(start_distance),
        "collection_min_goal_distance": min_distance,
        "collection_goal_reached": goal_reached,
    }


def _region_self_supervised_diagnostics(
    *,
    quality_gate: dict[str, Any],
    collection_acceptance: dict[str, Any],
    acceptance: dict[str, Any],
    training: dict[str, Any],
    region_navigation: dict[str, Any],
) -> dict[str, Any]:
    training_metrics = training.get("metrics") if isinstance(training.get("metrics"), dict) else {}
    segment_sample_count = training_metrics.get("segment_sample_count") if isinstance(training_metrics.get("segment_sample_count"), dict) else {}
    missing_training_segments = [
        name for name in ("middle", "goal") if int(segment_sample_count.get(name, 0) or 0) <= 0
    ]
    if not bool(quality_gate.get("passed")):
        status = "collection_insufficient"
        message = "Self-supervised collection did not cover enough goal-directed motion to train a useful navigation model."
        next_actions = [
            "Collect wider coverage inside the region with more rollouts or stronger goal/corridor bias.",
            "Lower the progress gate only for smoke tests, not for navigation acceptance.",
        ]
    elif bool(acceptance.get("goal_success")):
        status = "accepted"
        message = "Model-controlled evaluation reached and held the goal inside the configured region."
        next_actions: list[str] = []
    elif not bool(acceptance.get("model_controlled", True)):
        status = "not_model_controlled"
        message = "Evaluation did not run under model/manual control, so it cannot validate the learned model."
        next_actions = ["Switch evaluation to world_model_direct or model_mpc before accepting the run."]
    elif int(acceptance.get("collision_count", 0) or 0) > int(acceptance.get("max_collision_count", 0) or 0):
        status = "collision_limit_exceeded"
        message = "The vehicle collided more than allowed during model-controlled evaluation."
        next_actions = [
            "Reduce target speed or throttle during evaluation.",
            "Collect more recovery and low-speed steering examples near obstacles.",
        ]
    elif bool(acceptance.get("goal_reached")) and not bool(acceptance.get("final_goal_reached")):
        status = "goal_hold_failed"
        message = "The vehicle entered the goal radius but did not finish stopped inside it."
        next_actions = ["Verify terminal braking and goal latch behavior for the selected controller."]
    elif missing_training_segments:
        status = "training_coverage_insufficient"
        message = "The learned dynamics model was trained without enough middle/goal segment samples for route-free navigation."
        next_actions = [
            "Collect middle/goal segment examples by increasing route-aware rollouts, step count, and goal-zone coverage.",
            "Require nonzero middle and goal segment counts before treating a model as navigation-ready.",
            "Inspect the trajectory plot and route coverage before rerunning route-free evaluation.",
        ]
    else:
        status = "navigation_model_insufficient"
        message = "The local learned dynamics model ran under model control but did not produce a successful start-to-goal navigation policy."
        next_actions = [
            "Collect wider coverage inside the region with multiple rollouts and corridor-biased exploration.",
            "Run a task-route evaluation to isolate local control quality from route-free navigation.",
            "Add a traversability or cost-map learner before expecting route-free global navigation.",
        ]

    evidence = {
        "route_free": bool(region_navigation.get("route_free")),
        "evaluation_agent": str(region_navigation.get("evaluation_agent") or ""),
        "evaluation_route_mode": str(region_navigation.get("evaluation_route_mode") or ""),
        "experience_corridor": bool(region_navigation.get("experience_corridor")),
        "experience_route_point_count": int(region_navigation.get("experience_route_point_count", 0) or 0),
        "model_controlled": bool(acceptance.get("model_controlled", True)),
        "goal_success": bool(acceptance.get("goal_success")),
        "goal_reached": bool(acceptance.get("goal_reached")),
        "final_goal_reached": bool(acceptance.get("final_goal_reached")),
        "min_goal_distance": acceptance.get("min_goal_distance"),
        "final_goal_distance": acceptance.get("final_goal_distance"),
        "goal_radius": acceptance.get("goal_radius"),
        "collision_count": acceptance.get("collision_count"),
        "max_collision_count": acceptance.get("max_collision_count"),
        "collection_progress_ratio": quality_gate.get("progress_ratio"),
        "required_collection_progress_ratio": quality_gate.get("required_progress_ratio"),
        "collection_min_goal_distance": collection_acceptance.get("min_goal_distance"),
        "train_rmse": training_metrics.get("train_rmse"),
        "train_mse": training_metrics.get("train_mse"),
        "validation_rmse": training_metrics.get("validation_rmse"),
        "validation_mse": training_metrics.get("validation_mse"),
        "segment_sample_count": segment_sample_count,
        "missing_training_segments": missing_training_segments,
    }
    return {
        "status": status,
        "message": message,
        "next_actions": next_actions,
        "evidence": evidence,
    }


def export_orfd_beamng_terrain_draft(
    dataset_root: str,
    adapter: str = "",
    sequence_id: str = "",
    *,
    frame_index: int = 0,
    output_dir: str | Path | None = None,
    grid_size: int = 64,
    terrain_size_m: float = 40.0,
) -> dict[str, Any]:
    if not dataset_root:
        raise ValueError("Dataset root is required.")
    registry = default_dataset_registry()
    resolved_adapter = registry.resolve(dataset_root, adapter or None)
    sequence_ids = resolved_adapter.list_sequences(dataset_root)
    selected = sequence_id or sequence_ids[0]
    sequence = resolved_adapter.load_sequence(dataset_root, selected)
    if not sequence.frames:
        raise ValueError("Selected sequence has no frames.")
    index = min(max(0, int(frame_index)), len(sequence.frames) - 1)
    frame = sequence.frames[index]
    out = Path(output_dir or ROOT / "outputs" / "beamng_terrain_drafts" / _safe_name(selected))
    out.mkdir(parents=True, exist_ok=True)

    depth_source = frame.depth_path or frame.lidar_path or frame.front_rgb_path
    if depth_source is None:
        raise ValueError("Selected frame has no depth, lidar, or RGB asset to derive a terrain draft.")
    height = _height_grid_from_asset(depth_source, grid_size=grid_size)
    height_png = out / "heightmap.png"
    mesh_obj = out / "terrain_mesh.obj"
    preview_png = out / "terrain_preview.png"
    manifest_path = out / "beamng_map_draft.json"
    _write_height_png(height, height_png)
    _write_mesh_obj(height, mesh_obj, terrain_size_m=terrain_size_m)
    _write_preview_array(height, preview_png)
    manifest = {
        "status": "draft_ready",
        "beamng_import_ready": False,
        "note": "This is a local ORFD-derived heightmap/mesh draft. It is not a packaged BeamNG level yet.",
        "dataset_root": str(Path(dataset_root).resolve()),
        "adapter": resolved_adapter.name,
        "sequence_id": selected,
        "frame_index": index,
        "frame_id": frame.frame_id,
        "source_asset": depth_source,
        "terrain_size_m": terrain_size_m,
        "grid_size": grid_size,
        "heightmap": str(height_png.resolve()),
        "mesh_obj": str(mesh_obj.resolve()),
        "preview": str(preview_png.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def backend_options(request: RunRequest) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.backend == "dataset_replay":
        if request.dataset_root:
            options["dataset_root"] = request.dataset_root
        if request.sequence_id:
            options["sequence_id"] = request.sequence_id
        if request.adapter:
            options["adapter"] = request.adapter
        options["load_assets"] = bool(request.load_assets)
    return options


def agent_options(request: RunRequest) -> dict[str, Any]:
    if request.agent not in {"world_model", "route_world_model"}:
        return {}
    options: dict[str, Any] = {"world_model_name": request.world_model_type}
    if request.world_model_path:
        options["world_model_path"] = request.world_model_path
    if request.planner:
        options["planner_name"] = request.planner
        options["planner_config"] = {
            "horizon": request.planner_horizon,
            "num_samples": request.planner_samples,
            "iterations": request.planner_iterations,
        }
    return options


def scenario_for_request(request: RunRequest) -> str | Path | dict[str, Any]:
    if request.backend == "dataset_replay" and request.dataset_root:
        return {
            "scenario_id": f"dataset_{Path(request.dataset_root).name}",
            "dataset_root": request.dataset_root,
            "sequence_id": request.sequence_id or None,
            "adapter": request.adapter or None,
            "load_assets": request.load_assets,
        }
    raw_path = Path(request.scenario)
    if raw_path.exists():
        return raw_path
    return CONFIG_ROOT / "scenarios" / f"{request.scenario}.yaml"


def episode_summaries() -> list[dict[str, Any]]:
    if not DEFAULT_OUTPUT_ROOT.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(DEFAULT_OUTPUT_ROOT.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        metadata = _read_json(path / "metadata.json")
        metrics = _read_json(path / "metrics.json")
        rows.append({"episode_id": metadata.get("episode_id", path.name), "path": str(path), "metadata": metadata, "metrics": metrics})
    return rows


def load_episode_trace(episode_path: str | Path, *, limit: int = 5000) -> list[dict[str, Any]]:
    steps_path = Path(episode_path) / "steps.jsonl"
    if not steps_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with steps_path.open("r", encoding="utf-8") as file:
        for _, line in zip(range(max(limit, 0)), file):
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation", {})
            state = observation.get("vehicle_state", {}) if isinstance(observation, dict) else {}
            action = record.get("action", {}) or {}
            info = record.get("info", {}) if isinstance(record.get("info"), dict) else {}
            rows.append(
                {
                    "step_index": record.get("step_index"),
                    "timestamp": _float_or_nan(observation.get("timestamp") if isinstance(observation, dict) else None),
                    "x": _float_or_nan(state.get("x")),
                    "y": _float_or_nan(state.get("y")),
                    "z": _float_or_nan(state.get("z")),
                    "yaw": _float_or_nan(state.get("yaw")),
                    "pitch": _float_or_nan(state.get("pitch")),
                    "roll": _float_or_nan(state.get("roll")),
                    "speed": _float_or_nan(state.get("speed")),
                    "reward": _float_or_nan(record.get("reward")),
                    "steer": _float_or_nan(action.get("steer")),
                    "throttle": _float_or_nan(action.get("throttle")),
                    "brake": _float_or_nan(action.get("brake")),
                    "gear": _int_or_none(action.get("gear")),
                    "goal": observation.get("goal") if isinstance(observation, dict) else None,
                    "agent_diagnostics": dict(info.get("agent_diagnostics") if isinstance(info.get("agent_diagnostics"), dict) else {}),
                }
            )
    return rows


def metric_value(metrics: dict[str, Any], *path: str) -> Any:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return math.nan
        current = current[key]
    return current


def display_value(value: Any) -> str:
    if value is None:
        return NAN_TEXT
    if isinstance(value, float) and math.isnan(value):
        return NAN_TEXT
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def unfinished_features() -> list[dict[str, str]]:
    return [
        {"name": "完整 ORFD 场景级 BeamNG level 自动打包", "status": UNFINISHED_TEXT},
        {"name": "完整原版 LE-WM 视觉 latent 训练", "status": UNFINISHED_TEXT},
        {"name": "UE5 实时桥接监控", "status": UNFINISHED_TEXT},
        {"name": "运行中断/暂停恢复", "status": UNFINISHED_TEXT},
    ]


def _registry_rows(registry: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses = registry.status()
    for name in registry.names():
        status = statuses[name]
        row = asdict(status) if is_dataclass(status) else dict(status)
        row["description"] = registry.get(name).description
        rows.append(row)
    return rows


def _algorithm_rows() -> list[dict[str, Any]]:
    registry = default_algorithm_registry()
    rows: list[dict[str, Any]] = []
    statuses = registry.status()
    for name in registry.names():
        spec = registry.get(name)
        status = statuses[name]
        row = asdict(status) if is_dataclass(status) else dict(status)
        row["description"] = spec.description or spec.manifest.display_name
        row["display_name"] = spec.manifest.display_name
        rows.append(row)
    return rows


def _world_model_config_row(raw: dict[str, Any]) -> dict[str, Any]:
    config_id = _safe_name(str(raw.get("id") or raw.get("label") or "world_model_config"))
    label = str(raw.get("label") or config_id)
    validation = raw.get("validation") if isinstance(raw.get("validation"), dict) else {}
    row = {
        "id": config_id,
        "label": label,
        "algorithm": str(raw.get("algorithm") or "stablewm_lewm"),
        "world_model": str(raw.get("world_model") or "le_wm"),
        "model_path": str(raw.get("model_path") or ""),
        "source_training_run_path": str(raw.get("source_training_run_path") or ""),
        "validation": dict(validation),
    }
    row["demo_ready"] = bool(raw["demo_ready"]) if "demo_ready" in raw else _world_model_config_demo_ready(row)
    return row


def _world_model_config_demo_ready(row: dict[str, Any]) -> bool:
    validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
    if "demo_ready" in validation:
        return bool(validation.get("demo_ready"))
    if not bool(validation.get("goal_success")):
        return False
    if not bool(validation.get("model_controlled", True)):
        return False
    collision_count = _coerce_int(validation.get("collision_count"), default=0)
    max_collision_count = _coerce_int(validation.get("max_collision_count"), default=0)
    if collision_count > max_collision_count:
        return False
    goal_radius = _coerce_float(validation.get("goal_radius"))
    final_distance = _coerce_float(validation.get("final_goal_distance"))
    if math.isfinite(goal_radius) and math.isfinite(final_distance) and final_distance > goal_radius:
        return False
    algorithm = str(row.get("algorithm") or "").lower()
    world_model = str(row.get("world_model") or "").lower()
    if algorithm == "world_model_direct" or world_model == "tiny_learned":
        route_mode = str(validation.get("evaluation_route_mode") or "").strip().lower()
        route_free = bool(validation.get("route_free")) or route_mode in {"route_free", "none", "direct"}
        route_free_direct = bool(validation.get("route_free_direct")) or route_mode == "route_free_direct"
        route_waypoint_count = _coerce_int(validation.get("route_waypoint_count"), default=-1)
        if not route_free and route_waypoint_count != 0:
            return False
        if (
            bool(validation.get("experience_corridor"))
            or bool(validation.get("model_support_subgoals"))
            or bool(validation.get("model_support_field_subgoals"))
            or bool(validation.get("model_support_graph_subgoals"))
        ):
            return route_free and route_waypoint_count == 0
        if not route_free_direct:
            return False
    return True


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _validation_float_or_none(validation: dict[str, Any], key: str) -> float | None:
    value = _coerce_float(validation.get(key))
    return value if math.isfinite(value) else None


def _validation_float_or_default(validation: dict[str, Any], key: str, default: float) -> float:
    value = _coerce_float(validation.get(key))
    return value if math.isfinite(value) else float(default)


def _infer_world_model_type(path: Path) -> str:
    metadata_path = path / "model.json" if path.is_dir() else path
    if metadata_path.name == "model.json" and metadata_path.exists():
        try:
            payload = _read_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            payload = {}
        model_type = str(payload.get("model_type") or "")
        if model_type in {"tiny_learned", "mlp_dynamics", "le_wm", "simple_kinematic"}:
            return model_type
    if path.suffix.lower() == ".ckpt":
        return "le_wm"
    return ""


def _training_config_row(raw: dict[str, Any]) -> dict[str, Any]:
    config_id = _safe_name(str(raw.get("id") or raw.get("label") or "training_config")).lower()
    label = str(raw.get("label") or config_id)
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    return {
        "id": config_id,
        "label": label,
        "training_preset_id": str(raw.get("training_preset_id") or raw.get("preset_id") or "tiny_world_model"),
        "dataset_root": str(raw.get("dataset_root") or ""),
        "adapter": str(raw.get("adapter") or ""),
        "sequence_id": str(raw.get("sequence_id") or ""),
        "split_path": str(raw.get("split_path") or ""),
        "output_path": str(raw.get("output_path") or raw.get("model_path") or raw.get("hdf5_path") or ""),
        "parameters": dict(parameters),
    }


def _resolve_training_config(training_config: str | dict[str, Any], *, config_path: str | Path | None = None) -> dict[str, Any]:
    if isinstance(training_config, dict):
        return _training_config_row(training_config)
    config_id = str(training_config or "").strip()
    if not config_id:
        raise ValueError("Training config id is required.")
    for row in training_config_entries(config_path):
        if row["id"] == config_id or row["label"] == config_id:
            return row
    raise ValueError(f"Training config not found: {config_id}")


def _materialize_training_config_assets(row: dict[str, Any]) -> None:
    if str(row.get("id") or "") != "smoke_tiny_world_model":
        return
    dataset_root = Path(str(row.get("dataset_root") or SMOKE_TRAINING_DATASET_ROOT))
    if dataset_root.exists():
        return
    create_mock_orfd_dataset(
        dataset_root,
        split="training",
        sequence_id="seq_0001",
        frame_count=8,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mapping_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return payload
    return load_yaml_file(path)


def _resolve_relative_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _trainer_manifest_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".yaml", ".yml"} and root.name not in INFERENCE_MANIFEST_FILENAMES else []
    if not root.exists():
        return []
    paths: list[Path] = []
    paths.extend(
        path
        for path in root.glob("*.yaml")
        if path.is_file() and ".template." not in path.name and path.name not in INFERENCE_MANIFEST_FILENAMES
    )
    paths.extend(
        path
        for path in root.glob("*.yml")
        if path.is_file() and ".template." not in path.name and path.name not in INFERENCE_MANIFEST_FILENAMES
    )
    for name in TRAINER_MANIFEST_FILENAMES:
        direct = root / name
        if direct.is_file():
            paths.append(direct)
        paths.extend(path for path in root.glob(f"*/{name}") if path.is_file())
    return paths


def _dataset_manifest_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name in DATASET_MANIFEST_FILENAMES else []
    if not root.exists():
        return []
    paths: list[Path] = []
    for name in DATASET_MANIFEST_FILENAMES:
        direct = root / name
        if direct.is_file():
            paths.append(direct)
        paths.extend(path for path in root.glob(f"*/{name}") if path.is_file())
    return paths


def _trainer_parameters(schema: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return validate_trainer_parameters(schema, overrides)


def _infer_trainer_parameter_schema(parameters: dict[str, Any]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for name, value in parameters.items():
        key = str(name)
        if isinstance(value, bool):
            value_type = "bool"
        elif isinstance(value, int):
            value_type = "int"
        elif isinstance(value, float):
            value_type = "float"
        else:
            value_type = "str"
        schema[key] = {"type": value_type, "default": value}
    return schema


def _default_trainer_arguments(schema: dict[str, Any]) -> list[str]:
    arguments = ["{dataset_root}", "--output", "{output_dir}"]
    for name in schema:
        key = str(name)
        flag = "--" + key.replace("_", "-")
        arguments.extend([flag, "{params." + key + "}"])
    return arguments


def _validate_trainer_launch(manifest: dict[str, Any]) -> None:
    launch = manifest.get("launch") if isinstance(manifest.get("launch"), dict) else {}
    kind = str(launch.get("kind") or "python_script")
    if kind == "python_script":
        entrypoint = _resolve_trainer_entrypoint(manifest)
        if not entrypoint.exists():
            raise FileNotFoundError(f"Trainer entrypoint not found: {entrypoint}")
    manifest_dir = Path(str(manifest["manifest_dir"]))
    resolve_trainer_working_directory(manifest, manifest_dir)
    command = build_trainer_command(manifest, arguments=[], manifest_dir=manifest_dir)
    if kind == "executable" and not str(launch.get("conda_env") or ""):
        executable = Path(command[0])
        if not executable.exists():
            raise FileNotFoundError(f"Trainer executable not found: {executable}")


def _trainer_dataset_compatibility(
    manifest: dict[str, Any],
    *,
    dataset_root: str,
    adapter: str,
    sequence_id: str,
    split_path: str,
) -> dict[str, Any]:
    input_spec = manifest.get("input") if isinstance(manifest.get("input"), dict) else {}
    issues: list[str] = []
    registry = default_dataset_registry()
    resolved_adapter = None
    resolved_adapter_name = adapter
    if not resolved_adapter_name:
        try:
            resolved_adapter = registry.resolve(dataset_root)
            resolved_adapter_name = resolved_adapter.name
        except (KeyError, ValueError):
            resolved_adapter_name = ""
    expected_format = input_spec.get("dataset_format", "any_registered_adapter")
    allowed_formats = expected_format if isinstance(expected_format, list) else [expected_format]
    allowed_formats = [str(value) for value in allowed_formats]
    if (
        "any_registered_adapter" not in allowed_formats
        and resolved_adapter_name
        and resolved_adapter_name not in allowed_formats
    ):
        issues.append(
            f"Dataset adapter '{resolved_adapter_name}' is incompatible; trainer accepts: "
            + ", ".join(allowed_formats)
        )

    available_modalities: set[str] = set()
    declared_modalities: set[str] = set()
    required_modalities = {str(value) for value in input_spec.get("required_modalities", [])}
    selected_ids: list[str] = []
    if required_modalities:
        try:
            if resolved_adapter is None:
                resolved_adapter = registry.resolve(dataset_root, adapter or None)
                resolved_adapter_name = resolved_adapter.name
            sequence_ids = resolved_adapter.list_sequences(dataset_root)
            if sequence_id:
                if sequence_id not in sequence_ids:
                    issues.append(f"Dataset sequence not found: {sequence_id}")
                else:
                    selected_ids = [sequence_id]
            else:
                selected_ids = sequence_ids[:1]
            if not selected_ids:
                issues.append("Dataset has no readable sequences.")
            for selected_id in selected_ids:
                sequence = resolved_adapter.load_sequence(dataset_root, selected_id)
                raw_expected = sequence.metadata.get("expected_modalities")
                if isinstance(raw_expected, list):
                    declared_modalities.update(str(value) for value in raw_expected)
                for frame in sequence.frames:
                    available_modalities.update(frame.available_assets())
        except (KeyError, ValueError, FileNotFoundError) as exc:
            issues.append(f"Dataset cannot satisfy trainer input requirements: {exc}")
    missing_modalities = sorted(required_modalities - available_modalities)
    if missing_modalities:
        issues.append("Dataset is missing required modalities: " + ", ".join(missing_modalities))

    split_required = bool(input_spec.get("split_required", False))
    split_payload: dict[str, Any] = {}
    if split_required and not split_path:
        issues.append("Trainer requires a dataset split definition.")
    if split_path:
        target = Path(split_path)
        if not target.is_file():
            issues.append(f"Dataset split definition not found: {target}")
        else:
            try:
                split_payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"Dataset split definition is invalid: {exc}")
            if split_payload:
                try:
                    validate_dataset_split_payload(split_payload)
                except ValueError as exc:
                    issues.append(f"Dataset split definition is invalid: {exc}")
                split_adapter = str(split_payload.get("adapter") or "")
                if split_adapter and resolved_adapter_name and split_adapter != resolved_adapter_name:
                    issues.append(
                        f"Dataset split adapter '{split_adapter}' does not match '{resolved_adapter_name}'."
                    )
                split_dataset_root = str(split_payload.get("dataset_root") or "").strip()
                if split_dataset_root and Path(split_dataset_root).resolve() != Path(dataset_root).resolve():
                    issues.append(
                        f"Dataset split root '{Path(split_dataset_root).resolve()}' does not match "
                        f"'{Path(dataset_root).resolve()}'."
                    )

    return {
        "compatible": not issues,
        "adapter": resolved_adapter_name,
        "accepted_dataset_formats": allowed_formats,
        "sequence_ids_checked": selected_ids,
        "declared_modalities": sorted(declared_modalities),
        "available_modalities": sorted(available_modalities),
        "required_modalities": sorted(required_modalities),
        "missing_modalities": missing_modalities,
        "split_required": split_required,
        "split_path": split_path,
        "issues": issues,
    }


def _trainer_command(
    manifest: dict[str, Any],
    dataset_root: str,
    output_dir: Path,
    parameters: dict[str, Any],
    adapter: str,
    sequence_id: str,
    split_path: str = "",
) -> list[str]:
    context = {
        "dataset_root": dataset_root,
        "output_dir": str(output_dir.resolve()),
        "adapter": adapter,
        "sequence_id": sequence_id,
        "split_path": split_path,
        "manifest_dir": str(Path(str(manifest["manifest_dir"])).resolve()),
        "params": _AttrDict(parameters),
    }
    arguments: list[str] = []
    for value in manifest.get("arguments", []):
        rendered = str(value).format_map(_AttrDict(context))
        arguments.append(rendered)
    return build_trainer_command(
        manifest,
        arguments=arguments,
        manifest_dir=Path(str(manifest["manifest_dir"])),
    )


def _inference_command(
    inference: dict[str, Any],
    *,
    manifest_dir: Path,
    artifact_path: str,
    dataset_root: str,
    output_dir: Path,
    parameters: dict[str, Any],
    adapter: str,
    sequence_id: str,
    split_path: str = "",
) -> list[str]:
    context = {
        "artifact_path": str(Path(artifact_path).resolve()),
        "checkpoint_path": str(Path(artifact_path).resolve()),
        "dataset_root": dataset_root,
        "output_dir": str(output_dir.resolve()),
        "adapter": adapter,
        "sequence_id": sequence_id,
        "split_path": split_path,
        "manifest_dir": str(manifest_dir.resolve()),
        "params": _AttrDict(parameters),
    }
    arguments = [str(value).format_map(_AttrDict(context)) for value in inference.get("arguments", [])]
    return build_trainer_command(inference, arguments=arguments, manifest_dir=manifest_dir)


def _normalize_inference_previews(value: Any, output_dir: Path) -> dict[str, str]:
    if isinstance(value, dict):
        raw = {str(key): str(path) for key, path in value.items()}
    elif isinstance(value, list):
        raw = {f"preview_{index + 1}": str(path) for index, path in enumerate(value)}
    elif value:
        raw = {"primary": str(value)}
    else:
        raw = {}
    previews: dict[str, str] = {}
    for key, path_value in raw.items():
        path = _resolve_trainer_output_path(output_dir, path_value)
        if path.is_file():
            previews[key] = str(path)
    return previews


def _artifact_size_bytes(path: Path, max_files: int = 10000) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for index, child in enumerate(path.rglob("*")):
        if index >= max_files:
            break
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _resolve_trainer_entrypoint(manifest: dict[str, Any]) -> Path:
    launch = manifest.get("launch") if isinstance(manifest.get("launch"), dict) else {}
    entrypoint = Path(str(launch.get("entrypoint") or manifest.get("entrypoint") or ""))
    if entrypoint.is_absolute():
        return entrypoint
    return (Path(str(manifest["manifest_dir"])) / entrypoint).resolve()


def _json_from_command_output(output: str, command: list[str]) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    payload: dict[str, Any] | None = None
    cursor = 0
    while cursor < len(output):
        json_start = output.find("{", cursor)
        if json_start < 0:
            break
        try:
            candidate, consumed = decoder.raw_decode(output[json_start:])
        except json.JSONDecodeError:
            cursor = json_start + 1
            continue
        if isinstance(candidate, dict):
            payload = candidate
        cursor = json_start + max(1, consumed)
    if payload is None:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Command emitted non-object JSON: {' '.join(command)}")
    return payload


def _numeric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def _timestamp() -> str:
    return f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


class _AttrDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return _AttrDict(value)
        return value


def _run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "command failed").strip())
    return _json_from_command_output(completed.stdout.strip(), command)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_or_default(value: Any, default: float) -> float:
    number = _float_or_nan(value)
    return float(number) if math.isfinite(number) else float(default)


def _finite_or_inf(value: Any) -> float:
    number = _float_or_nan(value)
    return float(number) if math.isfinite(number) else math.inf


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _coerce_point2(value: Any, label: str) -> tuple[float, float]:
    try:
        items = list(value)
        return (float(items[0]), float(items[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid {label}: expected [x, y].") from exc


def _coerce_point3(value: Any, label: str) -> tuple[float, float, float]:
    try:
        items = list(value)
        return (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid {label}: expected [x, y, z].") from exc


def _write_preview_image(asset_path: str, output_path: Path, *, modality: str = "") -> Path | None:
    try:
        image = _lidar_preview(_load_lidar_points(asset_path)) if modality == "lidar_points" else _load_asset_image(asset_path)
    except (OSError, ValueError):
        return None
    _write_preview_array(image, output_path)
    return output_path


def _load_asset_image(asset_path: str) -> np.ndarray:
    from scripts.export_lewm_hdf5 import _load_image

    suffix = Path(asset_path.rsplit("!", 1)[-1]).suffix.lower()
    if suffix in {".bin", ".pcd"}:
        points = _load_lidar_points(asset_path)
        return _lidar_preview(points)
    image = _load_image(asset_path)
    if image.ndim == 2:
        return image
    return image[..., :3]


def _height_grid_from_asset(asset_path: str, *, grid_size: int) -> np.ndarray:
    suffix = Path(asset_path.rsplit("!", 1)[-1]).suffix.lower()
    if suffix == ".bin":
        points = _load_lidar_points(asset_path)
        if points.size == 0:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        return _points_to_height_grid(points, grid_size=grid_size)
    image = _load_asset_image(asset_path)
    if image.ndim == 3:
        image = np.mean(image[..., :3], axis=2)
    image = _resize_nearest(np.asarray(image, dtype=np.float32), grid_size)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    low, high = np.percentile(finite, [2, 98])
    if math.isclose(float(low), float(high)):
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    height = np.clip((image - low) / (high - low), 0.0, 1.0)
    height[~np.isfinite(height)] = 0.0
    return height.astype(np.float32)


def _load_lidar_points(asset_path: str) -> np.ndarray:
    raw = _read_asset_bytes(asset_path)
    suffix = Path(asset_path.rsplit("!", 1)[-1]).suffix.lower()
    if suffix == ".npy":
        from io import BytesIO

        points = np.asarray(np.load(BytesIO(raw), allow_pickle=False), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("LiDAR .npy must be an Nx3 or Nx4 array.")
        return points[:, :3]
    if suffix == ".pcd":
        return _load_ascii_pcd_points(raw)
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size < 3:
        return np.empty((0, 3), dtype=np.float32)
    normalized_path = asset_path.replace("\\", "/").lower()
    candidates = (5, 4, 3) if "/lidar_data/" in normalized_path else (4, 5, 3)
    stride = next((candidate for candidate in candidates if values.size % candidate == 0), 3)
    return values[: values.size // stride * stride].reshape(-1, stride)[:, :3]


def _load_ascii_pcd_points(raw: bytes) -> np.ndarray:
    marker = b"DATA ascii"
    index = raw.find(marker)
    if index < 0:
        raise ValueError("Only ASCII PCD previews are currently supported.")
    start = raw.find(b"\n", index)
    if start < 0:
        return np.empty((0, 3), dtype=np.float32)
    rows: list[list[float]] = []
    for line in raw[start + 1 :].decode("utf-8", errors="replace").splitlines():
        values = line.strip().split()
        if len(values) < 3:
            continue
        rows.append([float(values[0]), float(values[1]), float(values[2])])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def _read_asset_bytes(asset_path: str) -> bytes:
    if asset_path.startswith("zip://"):
        import zipfile

        raw = asset_path.removeprefix("zip://")
        zip_path, member = raw.split("!", 1)
        with zipfile.ZipFile(zip_path) as archive:
            return archive.read(member)
    return Path(asset_path).read_bytes()


def _lidar_preview(points: np.ndarray, *, size: int = 512) -> np.ndarray:
    if points.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    xy = points[:, :2]
    finite = xy[np.isfinite(xy).all(axis=1)]
    if finite.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    mins = np.percentile(finite, 2, axis=0)
    maxs = np.percentile(finite, 98, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    coords = np.clip((finite - mins) / span * (size - 1), 0, size - 1).astype(int)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[size - 1 - coords[:, 1], coords[:, 0]] = 255
    return canvas


def _points_to_height_grid(points: np.ndarray, *, grid_size: int) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    xy = finite[:, :2]
    z = finite[:, 2]
    mins = np.percentile(xy, 2, axis=0)
    maxs = np.percentile(xy, 98, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    coords = np.clip((xy - mins) / span * (grid_size - 1), 0, grid_size - 1).astype(int)
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
    for (x, y), value in zip(coords, z, strict=False):
        row = grid_size - 1 - y
        grid[row, x] = value if np.isnan(grid[row, x]) else max(grid[row, x], value)
    fill = float(np.nanmedian(grid)) if np.isfinite(grid).any() else 0.0
    grid = np.nan_to_num(grid, nan=fill)
    low, high = np.percentile(grid, [2, 98])
    if math.isclose(float(low), float(high)):
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    return np.clip((grid - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _write_height_png(height: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(height, 0.0, 1.0)
    Image.fromarray((data * 65535).astype(np.uint16)).save(output_path)


def _write_preview_array(image: np.ndarray, output_path: Path) -> None:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim == 2:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            scaled = np.zeros(array.shape, dtype=np.uint8)
        else:
            low, high = np.percentile(finite, [2, 98])
            if math.isclose(float(low), float(high)):
                scaled = np.zeros(array.shape, dtype=np.uint8)
            else:
                scaled = np.clip((array - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(scaled).save(output_path)
        return
    Image.fromarray(np.clip(array[..., :3], 0, 255).astype(np.uint8)).save(output_path)


def _write_mesh_obj(height: np.ndarray, output_path: Path, *, terrain_size_m: float) -> None:
    rows, cols = height.shape
    scale_x = terrain_size_m / max(cols - 1, 1)
    scale_y = terrain_size_m / max(rows - 1, 1)
    z_scale = max(terrain_size_m * 0.12, 1.0)
    lines = ["# ORFD-derived local terrain mesh draft"]
    for row in range(rows):
        for col in range(cols):
            x = col * scale_x - terrain_size_m / 2.0
            y = row * scale_y
            z = float(height[row, col]) * z_scale
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = row * cols + col + 1
            b = a + 1
            c = a + cols
            d = c + 1
            lines.append(f"f {a} {c} {b}")
            lines.append(f"f {b} {c} {d}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    rows = np.linspace(0, height - 1, size).astype(int)
    cols = np.linspace(0, width - 1, size).astype(int)
    return image[rows][:, cols]


def _relative_to_root(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def _checkpoint_display_label(path: Path, output_root: Path) -> str:
    resolved = path.resolve()
    try:
        label_path = resolved.relative_to(output_root.resolve())
    except ValueError:
        try:
            label_path = resolved.relative_to(ROOT)
        except ValueError:
            label_path = resolved
    return str(label_path).replace("\\", "/")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "default"
