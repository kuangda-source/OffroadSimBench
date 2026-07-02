"""Qt-independent service helpers used by the desktop GUI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from html import escape
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.agents import default_agent_registry
from offroad_sim.algorithms import DataPrepRequest, TrainRequest, default_algorithm_registry
from offroad_sim.backends import BeamNGBackend, BeamNGConnectionConfig, default_backend_registry
from offroad_sim.core import Action, VehicleState
from offroad_sim.datasets import create_mock_orfd_dataset, default_dataset_registry
from offroad_sim.datasets import DatasetFrame, DatasetSequence
from offroad_sim.evaluation import run_episode
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT
from offroad_sim.planning import default_planner_registry
from offroad_sim.tasks import NavigationRegionTask, load_navigation_region_task
from offroad_sim.utils.yaml_io import load_yaml_file
from offroad_sim.vehicles import load_vehicle_config
from offroad_sim.world_models import TinyLearnedWorldModel, default_world_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
DEFAULT_NAVIGATION_TASK_PATH = CONFIG_ROOT / "tasks" / "beamng_johnson_valley_nav_001.yaml"
DEFAULT_LEWM_CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "region_navigation"
    / "johnson_valley_nav_test_train_v2_validated"
    / "model"
    / "lewm_cost_object.ckpt"
)
WORLD_MODEL_CONFIGS_PATH = CONFIG_ROOT / "world_model_configs.json"
DEFAULT_WORLD_MODEL_CONFIG_ID = "johnson_valley_lewm_validated"
DEFAULT_DEMO_CONFIG_ID = "johnson_valley_standard_demo"
TRAINING_CONFIGS_PATH = CONFIG_ROOT / "training_configs.json"
SMOKE_TRAINING_DATASET_ROOT = ROOT / "outputs" / "training_studio_smoke" / "datasets" / "mock_orfd"
SMOKE_TINY_MODEL_OUTPUT_DIR = ROOT / "outputs" / "training_studio_smoke" / "models" / "tiny_world_model"
SMOKE_TRAINING_SEQUENCE_ID = "training/seq_0001"
TRAINING_RUN_FILENAME = "training_run.json"
DATASET_MANIFEST_FILENAMES = ("dataset_manifest.yaml", "dataset_manifest.yml")
DATASET_MANIFEST_DIRS = (CONFIG_ROOT / "datasets",)
TRAINER_MANIFEST_FILENAMES = ("trainer.yaml", "trainer.yml")
TRAINER_MANIFEST_DIRS = (CONFIG_ROOT / "trainers", ROOT / "trainers")
NAN_TEXT = "NaN"
REGION_TRAINING_COLLECTION_FILENAME = "region_training_collection.json"
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
    min_route_coverage_ratio: float = 0.0
    min_goal_zone_coverage: float = 0.0
    eval_steps: int = 1000
    seed: int = 7
    planner: str = "navigation_mpc"
    planner_horizon: int = 6
    planner_samples: int = 32
    planner_iterations: int = 3
    evaluation_agent: str = "world_model_direct"
    evaluation_route_mode: str = "route_free"
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
    min_route_coverage_ratio: float = 0.0
    min_goal_zone_coverage: float = 0.0
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
    evaluation_agent: str = "world_model_direct"
    include_route_guided_baseline: bool = False
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
        "dataset_manifests": dataset_manifest_entries(),
        "training_configs": training_config_entries(),
        "training_presets": training_preset_entries(),
        "training_runs": training_run_entries(),
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
            "label": "ORFD / dataset -> StableWM HDF5",
            "kind": "export",
            "available": True,
            "status": "available",
            "description": "Export image sequences and actions into the HDF5 layout used by StableWM/LE-WM training.",
        },
        {
            "id": "lewm_cost_model",
            "label": "Train LE-WM cost model",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "Train the local lightweight LE-WM-compatible cost checkpoint from a StableWM HDF5 file.",
        },
        {
            "id": "tiny_world_model",
            "label": "Train tiny world model",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "Fit the built-in tiny learned dynamics model for quick dataset sanity checks.",
        },
        {
            "id": "beamng_region_training_data",
            "label": "Collect BeamNG region training data",
            "kind": "collection",
            "available": True,
            "status": "available",
            "description": "Collect reusable BeamNG region episodes for simulator-trained world models.",
        },
        {
            "id": "region_world_model_training",
            "label": "Train BeamNG region world model",
            "kind": "training",
            "available": True,
            "status": "available",
            "description": "Train the built-in tiny learned world model from collected BeamNG region episodes.",
        },
        {
            "id": "lewm_full_self_supervised",
            "label": "LE-WM full self-supervised training",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "Reserved adapter for the full visual latent LE-WM training stack.",
        },
        {
            "id": "tdmpc2_adapter",
            "label": "TD-MPC2 adapter",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "Reserved adapter slot for TD-MPC2-style model-based control experiments.",
        },
        {
            "id": "dreamerv3_adapter",
            "label": "DreamerV3 adapter",
            "kind": "training",
            "available": False,
            "status": UNFINISHED_TEXT,
            "description": "Reserved adapter slot for DreamerV3-style world model training.",
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
                "parameters": dict(row.get("parameters", {})),
                "input": dict(row.get("input", {})),
                "outputs": dict(row.get("outputs", {})),
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
    trainer_id = _safe_name(str(data.get("trainer_id") or data.get("id") or manifest_path.parent.name))
    if not trainer_id:
        raise ValueError(f"Trainer manifest has no trainer_id: {manifest_path}")
    entrypoint = str(data.get("entrypoint") or "").strip()
    if not entrypoint:
        raise ValueError(f"Trainer manifest has no entrypoint: {manifest_path}")
    return {
        "id": trainer_id,
        "label": str(data.get("display_name") or data.get("label") or trainer_id),
        "trainer_id": trainer_id,
        "runtime": str(data.get("runtime") or "python"),
        "entrypoint": entrypoint,
        "description": str(data.get("description") or ""),
        "arguments": list(data.get("arguments") or []),
        "parameters": dict(data.get("parameters") or {}),
        "input": dict(data.get("input") or {}),
        "outputs": dict(data.get("outputs") or {}),
        "manifest_path": str(manifest_path),
        "manifest_dir": str(manifest_path.parent),
    }


def import_trainer_manifest(source_path: str | Path, destination_root: str | Path | None = None) -> dict[str, Any]:
    """Install an external trainer manifest into the project trainer catalog."""

    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Trainer manifest not found: {source}")
    source_row = load_trainer_manifest(source)
    data = load_yaml_file(source)
    entrypoint = Path(str(data.get("entrypoint") or ""))
    if not entrypoint.is_absolute():
        data["entrypoint"] = str((source.parent / entrypoint).resolve())
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


def save_trainer_manifest(
    *,
    trainer_id: str = "",
    label: str = "",
    entrypoint: str,
    runtime: str = "python",
    arguments: list[Any] | None = None,
    parameters: dict[str, Any] | None = None,
    input_spec: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    description: str = "",
    destination_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create a trainer manifest from a local algorithm entrypoint."""

    entrypoint_path = Path(entrypoint).resolve()
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"Trainer entrypoint not found: {entrypoint_path}")
    manifest_id = _safe_name(str(trainer_id or label or entrypoint_path.stem)).lower()
    display_name = str(label or trainer_id or entrypoint_path.stem)
    data: dict[str, Any] = {
        "trainer_id": manifest_id,
        "display_name": display_name,
        "runtime": str(runtime or "python"),
        "entrypoint": str(entrypoint_path),
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
) -> dict[str, Any]:
    manifest = load_trainer_manifest(manifest_path)
    target_dir = Path(output_dir or ROOT / "outputs" / "training_runs" / f"{manifest['id']}_{_timestamp()}")
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_parameters = _trainer_parameters(manifest.get("parameters", {}), parameters or {})
    command = _trainer_command(manifest, dataset_root, target_dir, resolved_parameters, adapter, sequence_id)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
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
            artifact_path=str(target_dir.resolve()),
            artifact_type=str(manifest.get("outputs", {}).get("artifact_type") or "artifact"),
            parameters=resolved_parameters,
            summary={"command": command, "stderr": completed.stderr.strip()},
            logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
        )
        raise RuntimeError((completed.stderr or completed.stdout or "trainer command failed").strip())

    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    payload = _trainer_result_payload(manifest, target_dir, completed.stdout, command)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    history = payload.get("history") if isinstance(payload.get("history"), dict) else {}
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
    record = write_training_run_record(
        target_dir,
        preset_id=manifest["id"],
        status="completed",
        dataset_root=dataset_root,
        adapter=adapter,
        sequence_id=sequence_id,
        artifact_path=artifact_path,
        artifact_type=artifact_type,
        metrics=metrics,
        history=history,
        parameters=resolved_parameters,
        summary={"trainer_manifest_path": str(Path(manifest_path).resolve()), "command": command},
        logs={"stdout": str(stdout_path), "stderr": str(stderr_path)},
    )
    payload["output_dir"] = str(target_dir.resolve())
    payload["metrics"] = metrics
    payload["history"] = history
    payload["artifact_path"] = artifact_path
    payload["artifact_type"] = artifact_type
    payload["training_run_path"] = record["path"]
    payload["command"] = command
    return payload


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

    manifest_path = str(preset.get("manifest_path") or "").strip()
    if manifest_path:
        payload = run_trainer_manifest_job(
            manifest_path,
            dataset_root=dataset_root,
            output_dir=output_path,
            parameters=parameters,
            adapter=adapter,
            sequence_id=sequence_id,
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
    if manifest_path and not issues:
        try:
            manifest = load_trainer_manifest(manifest_path)
            entrypoint = _resolve_trainer_entrypoint(manifest)
            if not entrypoint.exists():
                issues.append(f"Trainer entrypoint not found: {entrypoint}")
            command_preview = _trainer_command(
                manifest,
                dataset_root,
                Path(output_path),
                parameters,
                str(row.get("adapter") or ""),
                str(row.get("sequence_id") or ""),
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
        },
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
    artifact_path: str = "",
    artifact_type: str = "",
    metrics: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    logs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_dir = Path(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    record_path = target_dir / TRAINING_RUN_FILENAME
    preset = next((row for row in training_preset_entries() if row["id"] == preset_id), {})
    payload: dict[str, Any] = {
        "run_id": _safe_name(target_dir.name or preset_id),
        "preset_id": preset_id,
        "preset_label": str(preset.get("label") or preset_id),
        "status": status,
        "dataset_root": dataset_root,
        "adapter": adapter,
        "sequence_id": sequence_id,
        "artifact_path": str(Path(artifact_path).resolve()) if artifact_path else "",
        "artifact_type": artifact_type,
        "metrics": dict(metrics or {}),
        "history": _normalize_metric_history(history or {}),
        "parameters": dict(parameters or {}),
        "summary": dict(summary or {}),
        "logs": _normalize_log_paths(logs or {}),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
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


def training_metric_history(record: dict[str, Any]) -> dict[str, list[float]]:
    history = _normalize_metric_history(record.get("history") if isinstance(record.get("history"), dict) else {})
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in _numeric_metric_items(metrics):
            history.setdefault(key, [value])
    return history


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

    metrics_from_file = _read_optional_json_mapping(_trainer_output_file(output_dir, outputs.get("metrics_file"), "metrics.json"))
    if metrics_from_file:
        metrics.update(metrics_from_file)

    history_from_file = _read_optional_json_mapping(_trainer_output_file(output_dir, outputs.get("history_file"), "history.json"))
    if history_from_file:
        history.update(history_from_file)

    event_history = _read_metric_events(_trainer_output_file(output_dir, outputs.get("events_file"), "events.jsonl"))
    if event_history:
        for key, values in event_history.items():
            history.setdefault(key, []).extend(values)

    normalized_history = _normalize_metric_history(history)
    if not normalized_history:
        normalized_history = {key: [value] for key, value in _numeric_metric_items(metrics)}
    for key, values in normalized_history.items():
        if key.lower() in {"step", "epoch", "iteration", "global_step"}:
            continue
        if values and key not in metrics:
            metrics[key] = values[-1]

    payload["metrics"] = dict(metrics)
    payload["history"] = normalized_history
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


def _read_optional_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _read_metric_events(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    history: dict[str, list[float]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for key, value in _numeric_metric_items(event):
            history.setdefault(key, []).append(value)
    return history


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
            "description": "BeamNG Johnson Valley region navigation with the validated LE-WM-compatible checkpoint.",
            "task_path": str(DEFAULT_NAVIGATION_TASK_PATH.resolve()),
            "task_relative_path": _relative_to_root(DEFAULT_NAVIGATION_TASK_PATH),
            "world_model_config_id": DEFAULT_WORLD_MODEL_CONFIG_ID,
            "planner": "navigation_mpc",
            "evaluation_agent": "model_mpc",
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
                "label": "Johnson Valley LE-WM validated",
                "algorithm": "stablewm_lewm",
                "world_model": "le_wm",
                "model_path": str(DEFAULT_LEWM_CHECKPOINT_PATH),
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
) -> dict[str, Any]:
    if not request.register_world_model_config:
        return {}
    if model_type != "tiny_learned" or not model_dir:
        return {}
    if not bool(acceptance.get("goal_success")):
        return {}
    validation = {
        "goal_success": bool(acceptance.get("goal_success")),
        "goal_reached": bool(acceptance.get("goal_reached")),
        "min_goal_distance": acceptance.get("min_goal_distance"),
        "final_goal_distance": acceptance.get("final_goal_distance"),
        "collision_count": acceptance.get("collision_count"),
        "quality_gate_passed": bool(quality_gate.get("passed")),
        "collection_progress_ratio": quality_gate.get("progress_ratio"),
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
    rows: dict[str, dict[str, Any]] = {
        "smoke_tiny_world_model": _training_config_row(
            {
                "id": "smoke_tiny_world_model",
                "label": "Smoke tiny world model",
                "training_preset_id": "tiny_world_model_script",
                "dataset_root": str(SMOKE_TRAINING_DATASET_ROOT),
                "adapter": "orfd",
                "sequence_id": SMOKE_TRAINING_SEQUENCE_ID,
                "output_path": str(SMOKE_TINY_MODEL_OUTPUT_DIR),
                "parameters": {"ridge": 0.0001},
            }
        ),
        "orfd_stablewm_hdf5": _training_config_row(
            {
                "id": "orfd_stablewm_hdf5",
                "label": "ORFD StableWM HDF5 export",
                "training_preset_id": "stablewm_hdf5",
                "dataset_root": "datasets/ORFD_Dataset_ICRA2022_ZIP",
                "adapter": "orfd",
                "sequence_id": "",
                "output_path": "outputs/stablewm/orfd_gui.h5",
                "parameters": {},
            }
        ),
        "orfd_tiny_world_model": _training_config_row(
            {
                "id": "orfd_tiny_world_model",
                "label": "ORFD tiny world model",
                "training_preset_id": "tiny_world_model",
                "dataset_root": "datasets/ORFD_Dataset_ICRA2022_ZIP",
                "adapter": "orfd",
                "sequence_id": "",
                "output_path": "outputs/models/orfd_tiny_world_model",
                "parameters": {},
            }
        ),
    }
    if config_path.exists():
        payload = _read_json(config_path)
        raw_rows: Any = payload.get("configs", payload) if isinstance(payload, dict) else payload
        if isinstance(raw_rows, list):
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
    }


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
        if name not in {"front_rgb", "depth", "label", "local_bev", "terrain_map"}:
            continue
        preview_path = _write_preview_image(asset_path, out / f"{index:06d}_{name}.png")
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
    algorithm = str(world_model_config.get("algorithm") or "stablewm_lewm")
    world_model = str(world_model_config.get("world_model") or "le_wm")
    model_path = str(world_model_config.get("model_path") or "")
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
        collection = _run_region_beamng_episode_with_reconnect_retry(
            scenario=collection_scenario,
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
            "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
            "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
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
    if request.world_model_type != "tiny_learned":
        raise ValueError("BeamNG region training currently supports world_model_type='tiny_learned'.")
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
    model = TinyLearnedWorldModel.fit(sequences)
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
    metrics = {
        "train_rmse": model.metadata.get("train_rmse"),
        "train_mse": model.metadata.get("train_mse"),
        "sequence_count": model.metadata.get("sequence_count"),
        "transition_count": model.metadata.get("transition_count"),
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
            "collection_min_goal_distance": [collection_metrics.get("collection_min_goal_distance")],
        },
        parameters={
            "world_model_type": request.world_model_type,
            "collection_manifest_path": str(manifest_path.resolve()),
        },
        summary={
            "task": task.to_dict(),
            "collection_manifest_path": str(manifest_path.resolve()),
            "collection_metrics": collection_metrics,
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
            "train_rmse": model.metadata.get("train_rmse"),
            "train_mse": model.metadata.get("train_mse"),
            "collection_rollout_count": len(episode_paths),
            "collection_min_goal_distance": collection_metrics.get("collection_min_goal_distance"),
            "collection_collision_count": collection_metrics.get("collection_collision_count"),
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
    if request.world_model_type != "tiny_learned":
        raise ValueError("Region self-supervised training currently supports world_model_type='tiny_learned'.")
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
    rollout_count = max(1, int(request.collect_rollouts))
    for rollout_index in range(rollout_count):
        collection = _run_region_beamng_episode_with_reconnect_retry(
            scenario=collection_scenario,
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
        sequences.append(_episode_trace_to_dataset_sequence(episode_path, task))
    collection_acceptance = min(collection_acceptances, key=lambda row: _finite_or_inf(row.get("min_goal_distance")))
    collection_distance = float(sum(collection_distances)) if collection_distances else math.nan
    coverage = _collection_coverage_metrics(
        task,
        episode_paths,
        grid_size=max(2, int(request.collection_coverage_grid_size)),
    )
    route_metrics = _collection_route_metrics(task, episode_paths)
    quality_gate = _collection_quality_gate(
        task,
        collection_acceptance,
        min_progress_ratio=request.min_collection_goal_progress_ratio,
        route_coverage_ratio=route_metrics["route_coverage_ratio"],
        min_route_coverage_ratio=request.min_route_coverage_ratio,
        goal_zone_coverage=route_metrics["goal_zone_coverage"],
        min_goal_zone_coverage=request.min_goal_zone_coverage,
    )
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
            },
            history={
                "collection_min_goal_distance": [collection_acceptance.get("min_goal_distance")],
                "collection_progress_ratio": [quality_gate.get("progress_ratio")],
                "collection_coverage_ratio": [coverage["ratio"]],
                "route_coverage_ratio": [route_metrics["route_coverage_ratio"]],
                "goal_zone_coverage": [route_metrics["goal_zone_coverage"]],
            },
            parameters={
                "world_model_type": request.world_model_type,
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
                "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
                "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
            },
            summary={
                "task_path": str(Path(request.task_path).resolve()),
                "quality_gate": quality_gate,
                "coverage": coverage,
                "route_metrics": route_metrics,
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
            "diagnostics": diagnostics,
            "region_navigation": {
                "collection_agent": "region_explorer",
                "evaluation_agent": request.evaluation_agent,
                "route_free": evaluation_route_free,
                "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
            },
            "training_run_path": training_run["path"],
        }
        summary_path = output_dir / "region_self_supervised_summary.json"
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        payload["summary_path"] = str(summary_path.resolve())
        return payload

    model = TinyLearnedWorldModel.fit(sequences)
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
    }
    evaluation = _run_region_beamng_episode_with_reconnect_retry(**evaluation_kwargs)
    acceptance = _navigation_acceptance(evaluation, task)
    region_navigation = evaluation.get("region_navigation", {}) if isinstance(evaluation.get("region_navigation"), dict) else {}
    region_navigation_payload = {
        **region_navigation,
        "collection_agent": "region_explorer",
        "evaluation_agent": request.evaluation_agent,
        "route_free": evaluation_route_free,
        "evaluation_route_mode": "route_free" if evaluation_route_free else "task_route",
    }
    diagnostics = _region_self_supervised_diagnostics(
        quality_gate=quality_gate,
        collection_acceptance=collection_acceptance,
        acceptance=acceptance,
        training=training,
        region_navigation=region_navigation_payload,
    )
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
            "train_rmse": model.metadata.get("train_rmse"),
            "train_mse": model.metadata.get("train_mse"),
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
        },
        history={
            "train_rmse": [model.metadata.get("train_rmse")],
            "train_mse": [model.metadata.get("train_mse")],
            "collection_min_goal_distance": [collection_acceptance.get("min_goal_distance")],
            "collection_progress_ratio": [quality_gate.get("progress_ratio")],
            "collection_coverage_ratio": [coverage["ratio"]],
            "route_coverage_ratio": [route_metrics["route_coverage_ratio"]],
            "goal_zone_coverage": [route_metrics["goal_zone_coverage"]],
            "evaluation_min_goal_distance": [acceptance.get("min_goal_distance")],
        },
        parameters={
            "world_model_type": request.world_model_type,
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
            "min_route_coverage_ratio": max(0.0, float(request.min_route_coverage_ratio)),
            "min_goal_zone_coverage": max(0.0, float(request.min_goal_zone_coverage)),
        },
        summary={
            "task_path": str(Path(request.task_path).resolve()),
            "quality_gate": quality_gate,
            "coverage": coverage,
            "route_metrics": route_metrics,
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
        },
    }
    summary_path = output_dir / "region_world_model_evaluation_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    payload["summary_path"] = str(summary_path.resolve())
    return payload


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
    return cleaned


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
        metadata={"task_id": task.task_id, "source": "beamng_region_self_supervised"},
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
) -> dict[str, Any]:
    planner_config = {
        "horizon": max(1, int(planner_horizon)),
        "num_samples": max(4, int(planner_samples)),
        "iterations": max(1, int(planner_iterations)),
    }
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
) -> dict[str, Any]:
    required = max(0.0, float(min_progress_ratio))
    required_route = max(0.0, float(min_route_coverage_ratio))
    required_goal_zone = max(0.0, float(min_goal_zone_coverage))
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
    passed = bool(progress_passed and route_passed and goal_zone_passed)
    if passed:
        reason = "passed"
    elif not progress_passed:
        reason = "collection_goal_progress_below_threshold"
    elif not route_passed:
        reason = "collection_route_coverage_below_threshold"
    else:
        reason = "collection_goal_zone_coverage_below_threshold"
    return {
        "passed": bool(passed),
        "reason": reason,
        "progress_ratio": float(progress_ratio),
        "required_progress_ratio": float(required),
        "route_coverage_ratio": float(route_ratio),
        "required_route_coverage_ratio": float(required_route),
        "goal_zone_coverage": float(goal_zone_ratio),
        "required_goal_zone_coverage": float(required_goal_zone),
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
        "train_rmse": (training.get("metrics") if isinstance(training.get("metrics"), dict) else {}).get("train_rmse"),
        "train_mse": (training.get("metrics") if isinstance(training.get("metrics"), dict) else {}).get("train_mse"),
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
    return {
        "id": config_id,
        "label": label,
        "algorithm": str(raw.get("algorithm") or "stablewm_lewm"),
        "world_model": str(raw.get("world_model") or "le_wm"),
        "model_path": str(raw.get("model_path") or ""),
        "source_training_run_path": str(raw.get("source_training_run_path") or ""),
        "validation": dict(validation),
    }


def _infer_world_model_type(path: Path) -> str:
    metadata_path = path / "model.json" if path.is_dir() else path
    if metadata_path.name == "model.json" and metadata_path.exists():
        try:
            payload = _read_json(metadata_path)
        except (OSError, json.JSONDecodeError):
            payload = {}
        model_type = str(payload.get("model_type") or "")
        if model_type in {"tiny_learned", "le_wm", "simple_kinematic"}:
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
        return [root] if root.suffix.lower() in {".yaml", ".yml"} else []
    if not root.exists():
        return []
    paths: list[Path] = []
    paths.extend(path for path in root.glob("*.yaml") if path.is_file())
    paths.extend(path for path in root.glob("*.yml") if path.is_file())
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
    values: dict[str, Any] = {}
    normalized_overrides = {str(name): value for name, value in overrides.items()}
    for name, raw_spec in schema.items():
        key = str(name)
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        if "default" in spec:
            values[key] = _coerce_trainer_value(spec.get("default"), str(spec.get("type") or "str"))
        if spec.get("required") is True and key not in values and key not in normalized_overrides:
            raise ValueError(f"Missing required parameter: {key}")
    for name, value in normalized_overrides.items():
        spec = schema.get(name) if isinstance(schema.get(name), dict) else {}
        values[str(name)] = _coerce_trainer_value(value, str(spec.get("type") or "str"))
    return values


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


def _coerce_trainer_value(value: Any, value_type: str) -> Any:
    normalized = value_type.lower()
    if normalized in {"int", "integer"}:
        return int(value)
    if normalized in {"float", "number"}:
        return float(value)
    if normalized in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    return value


def _trainer_command(
    manifest: dict[str, Any],
    dataset_root: str,
    output_dir: Path,
    parameters: dict[str, Any],
    adapter: str,
    sequence_id: str,
) -> list[str]:
    entrypoint = _resolve_trainer_entrypoint(manifest)
    runtime = str(manifest.get("runtime") or "python").lower()
    command = [sys.executable, str(entrypoint)] if runtime == "python" else [str(entrypoint)]
    context = {
        "dataset_root": dataset_root,
        "output_dir": str(output_dir.resolve()),
        "adapter": adapter,
        "sequence_id": sequence_id,
        "manifest_dir": str(Path(str(manifest["manifest_dir"])).resolve()),
        "params": _AttrDict(parameters),
    }
    for value in manifest.get("arguments", []):
        rendered = str(value).format_map(_AttrDict(context))
        command.append(rendered)
    return command


def _resolve_trainer_entrypoint(manifest: dict[str, Any]) -> Path:
    entrypoint = Path(str(manifest.get("entrypoint") or ""))
    if entrypoint.is_absolute():
        return entrypoint
    return (Path(str(manifest["manifest_dir"])) / entrypoint).resolve()


def _json_from_command_output(output: str, command: list[str]) -> dict[str, Any]:
    json_start = output.find("{")
    if json_start < 0:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}")
    payload, _ = json.JSONDecoder().raw_decode(output[json_start:])
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
    return time.strftime("%Y%m%dT%H%M%S")


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


def _write_preview_image(asset_path: str, output_path: Path) -> Path | None:
    try:
        image = _load_asset_image(asset_path)
    except (OSError, ValueError):
        return None
    _write_preview_array(image, output_path)
    return output_path


def _load_asset_image(asset_path: str) -> np.ndarray:
    from scripts.export_lewm_hdf5 import _load_image

    suffix = Path(asset_path.rsplit("!", 1)[-1]).suffix.lower()
    if suffix == ".bin":
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
    values = np.frombuffer(raw, dtype=np.float32)
    if values.size < 3:
        return np.empty((0, 3), dtype=np.float32)
    stride = 4 if values.size % 4 == 0 else 3
    return values[: values.size // stride * stride].reshape(-1, stride)[:, :3]


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
