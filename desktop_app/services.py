"""Qt-independent service helpers used by the desktop GUI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from offroad_sim.agents import default_agent_registry
from offroad_sim.algorithms import DataPrepRequest, TrainRequest, default_algorithm_registry
from offroad_sim.backends import BeamNGConnectionConfig, default_backend_registry
from offroad_sim.datasets import default_dataset_registry
from offroad_sim.evaluation import run_episode
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT
from offroad_sim.planning import default_planner_registry
from offroad_sim.tasks import NavigationRegionTask, load_navigation_region_task
from offroad_sim.utils.yaml_io import load_yaml_file
from offroad_sim.world_models import TinyLearnedWorldModel, default_world_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
NAN_TEXT = "NaN"
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
    planner: str = "le_wm_cem"
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
    planner: str = "le_wm_cem"
    beamng_gfx: str = "vk"
    close_beamng: bool = True
    step_delay_sec: float = 0.0
    pre_run_hold_sec: float = 0.0
    post_run_hold_sec: float = 0.0


@dataclass(slots=True)
class RegionNavigationClosedLoopRequest:
    task_path: str
    algorithm: str = "local_lewm_cost"
    vehicle: str = "configs/vehicles/ugv_medium.yaml"
    output_dir: str = ""
    collect_steps: int = 160
    eval_steps: int = 120
    seed: int = 7
    planner: str = "le_wm_cem"
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
    vehicle_model: str = "pickup"
    collection_drive_mode: str = "ai_line"
    evaluation_drive_mode: str = "manual"
    ai_line_speed: float = 10.0
    steps_per_action: int = 18
    camera_mode: str = "orbit"


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
        "episodes": episode_summaries(),
    }


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
    return {
        "model_type": model.model_type,
        "model_path": str(metadata_path),
        "output_dir": str(Path(output_dir).resolve()),
        "metrics": model.metadata,
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
    return _run_json_command(command)


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
    return _run_json_command(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_lewm_cost_model.py"),
            input_hdf5,
            "--output",
            output_dir,
        ]
    )


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

    collection = _run_region_beamng_episode(
        scenario=collection_scenario,
        vehicle=request.vehicle,
        max_steps=min(max(1, int(request.collect_steps)), task.max_steps),
        seed=request.seed,
        world_model_type="simple_kinematic",
        world_model_path="",
        planner="",
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

    algorithm = make_algorithm_adapter(request.algorithm)
    hdf5_path = output_dir / f"{_safe_name(task.task_id)}.h5"
    prep = algorithm.prepare_data(DataPrepRequest(episode_root=str(episode_path), output_path=str(hdf5_path), actions_from_state=True))
    hdf5 = {"output_hdf5": prep.output_path, **prep.metadata}

    model_dir = output_dir / "model"
    trained = algorithm.train(TrainRequest(input_path=str(hdf5["output_hdf5"]), output_dir=str(model_dir)))
    training = {"output_dir": trained.output_dir, "checkpoint_path": trained.checkpoint_path, **trained.metadata}
    model_path = str(training.get("output_dir") or model_dir)

    evaluation = _run_region_beamng_episode(
        scenario=evaluation_scenario,
        vehicle=request.vehicle,
        max_steps=min(max(1, int(request.eval_steps)), task.max_steps),
        seed=request.seed,
        world_model_type="le_wm",
        world_model_path=model_path,
        planner=request.planner,
        record=True,
        beamng_gfx=request.beamng_gfx,
        pre_run_hold_sec=0.0,
        step_delay_sec=request.step_delay_sec,
        post_run_hold_sec=request.post_run_hold_sec,
        close_beamng=request.close_beamng,
    )
    acceptance = _navigation_acceptance(evaluation, task)
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
    }
    summary_path = output_dir / "region_navigation_summary.json"
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
        beamng={
            "vehicle_model": str(request.vehicle_model or "pickup"),
            "camera_mode": str(request.camera_mode or "orbit"),
            "draw_route": True,
            "drive_mode": str(request.collection_drive_mode or "ai_line"),
            "collection_drive_mode": str(request.collection_drive_mode or "ai_line"),
            "evaluation_drive_mode": str(request.evaluation_drive_mode or "manual"),
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


def _run_region_beamng_episode(
    *,
    scenario: dict[str, Any],
    vehicle: str,
    max_steps: int,
    seed: int,
    world_model_type: str,
    world_model_path: str,
    planner: str,
    record: bool,
    beamng_gfx: str,
    pre_run_hold_sec: float,
    step_delay_sec: float,
    post_run_hold_sec: float,
    close_beamng: bool,
) -> dict[str, Any]:
    planner_config = {"horizon": 4, "num_samples": 16, "iterations": 2}
    agent_options_payload: dict[str, Any] = {"world_model_name": world_model_type}
    if world_model_path:
        agent_options_payload["world_model_path"] = world_model_path
    if planner:
        agent_options_payload["planner_name"] = planner
        agent_options_payload["planner_config"] = planner_config
    result = run_episode(
        backend_name="beamng",
        scenario=scenario,
        agent_name="route_world_model",
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
        "planner": planner or None,
        "scenario_id": scenario.get("scenario_id"),
        "beamng_gfx": beamng_gfx,
    }
    return payload


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
    goal_success = math.isfinite(min_distance) and min_distance <= task.goal_radius
    return {
        "goal_success": bool(goal_success and reached_in_region and collision_count <= task.max_collision_count),
        "goal_reached": bool(goal_success),
        "final_goal_distance": final_distance,
        "min_goal_distance": min_distance,
        "min_goal_step": min_step,
        "goal_radius": task.goal_radius,
        "final_in_region": bool(in_region),
        "reached_in_region": bool(reached_in_region),
        "collision_count": collision_count,
        "max_collision_count": task.max_collision_count,
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
            rows.append(
                {
                    "step_index": record.get("step_index"),
                    "x": _float_or_nan(state.get("x")),
                    "y": _float_or_nan(state.get("y")),
                    "speed": _float_or_nan(state.get("speed")),
                    "reward": _float_or_nan(record.get("reward")),
                    "steer": _float_or_nan(action.get("steer")),
                    "throttle": _float_or_nan(action.get("throttle")),
                    "brake": _float_or_nan(action.get("brake")),
                    "goal": observation.get("goal") if isinstance(observation, dict) else None,
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_json_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "command failed").strip())
    output = completed.stdout.strip()
    json_start = output.find("{")
    if json_start < 0:
        raise RuntimeError(f"Command did not emit JSON: {' '.join(command)}")
    payload, _ = json.JSONDecoder().raw_decode(output[json_start:])
    return payload


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


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


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "default"
