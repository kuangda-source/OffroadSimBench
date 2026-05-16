"""Qt-independent service helpers used by the desktop GUI."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from offroad_sim.agents import default_agent_registry
from offroad_sim.backends import default_backend_registry
from offroad_sim.datasets import default_dataset_registry
from offroad_sim.evaluation import run_episode
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT
from offroad_sim.planning import default_planner_registry
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
    if request.agent != "world_model":
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
        {"name": "LE-WM 训练向导", "status": UNFINISHED_TEXT},
        {"name": "UE5 实时桥接监控", "status": UNFINISHED_TEXT},
        {"name": "实时相机/深度图面板", "status": UNFINISHED_TEXT},
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
