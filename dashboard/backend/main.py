"""FastAPI application for local benchmark control and inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from offroad_sim.backends import default_backend_registry
from offroad_sim.datasets import default_dataset_registry
from offroad_sim.evaluation import run_episode
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT
from offroad_sim.evaluation.streaming import recorded_observation_payload, stream_episode_events
from offroad_sim.planning import default_planner_registry
from offroad_sim.utils.yaml_io import load_yaml_file
from offroad_sim.world_models import default_world_model_registry


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
AGENTS = ["random", "stop", "rule_based", "world_model"]


class RunEpisodeRequest(BaseModel):
    backend: str = "gym_heightmap"
    scenario: str = "forest_trail_001"
    agent: str = "rule_based"
    seed: int = 7
    max_steps: int = Field(default=1200, ge=1, le=100_000)
    record: bool = True
    record_arrays: bool = False
    world_model_type: str = "simple_kinematic"
    world_model: str | None = None
    planner: str | None = None
    planner_horizon: int = Field(default=10, ge=1, le=200)
    planner_samples: int = Field(default=128, ge=4, le=5000)
    planner_iterations: int = Field(default=4, ge=1, le=100)
    dataset_root: str | None = None
    sequence_id: str | None = None
    adapter: str | None = None
    load_assets: bool = False


app = FastAPI(title="OffroadSimBench Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    return _config_entries(CONFIG_ROOT / "scenarios", "scenario_id")


@app.get("/vehicles")
def vehicles() -> list[dict[str, Any]]:
    return _config_entries(CONFIG_ROOT / "vehicles", "vehicle_id")


@app.get("/agents")
def agents() -> list[dict[str, str]]:
    return [{"name": name, "status": "available"} for name in AGENTS]


@app.get("/backends")
def backends() -> list[dict[str, Any]]:
    registry = default_backend_registry()
    rows = []
    for name, status in registry.status().items():
        row = asdict(status) if is_dataclass(status) else dict(status)
        row["description"] = registry.get(name).description
        rows.append(row)
    return rows


@app.get("/world_models")
def world_models() -> list[dict[str, Any]]:
    registry = default_world_model_registry()
    rows = []
    for name, status in registry.status().items():
        row = asdict(status) if is_dataclass(status) else dict(status)
        row["description"] = registry.get(name).description
        rows.append(row)
    return rows


@app.get("/planners")
def planners() -> list[dict[str, Any]]:
    registry = default_planner_registry()
    rows = []
    for name, status in registry.status().items():
        row = asdict(status) if is_dataclass(status) else dict(status)
        row["description"] = registry.get(name).description
        rows.append(row)
    return rows


@app.get("/datasets/inspect")
def inspect_dataset(dataset_root: str, adapter: str | None = None, sequence_id: str | None = None) -> dict[str, Any]:
    try:
        registry = default_dataset_registry()
        resolved_adapter = registry.resolve(dataset_root, adapter)
        sequences = resolved_adapter.list_sequences(dataset_root)
        selected = sequence_id or sequences[0]
        sequence = resolved_adapter.load_sequence(dataset_root, selected)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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


@app.get("/beamng/status")
def beamng_status() -> dict[str, Any]:
    status = default_backend_registry().status("beamng")
    return asdict(status) if is_dataclass(status) else dict(status)


@app.post("/run_episode")
def run_episode_endpoint(request: RunEpisodeRequest) -> dict[str, Any]:
    scenario_path = _resolve_config("scenarios", request.scenario)
    try:
        result = run_episode(
            backend_name=request.backend,
            scenario=_scenario_for_request(request, scenario_path),
            agent_name=request.agent,
            seed=request.seed,
            max_steps=request.max_steps,
            record=request.record,
            output_root=DEFAULT_OUTPUT_ROOT,
            record_arrays=request.record_arrays,
            backend_options=_backend_options(request),
            agent_options=_agent_options(request),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@app.get("/stream_episode")
def stream_episode_endpoint(
    backend: str = "gym_heightmap",
    scenario: str = "forest_trail_001",
    agent: str = "rule_based",
    seed: int = 7,
    max_steps: int = 1200,
    record: bool = True,
    record_arrays: bool = True,
    delay_ms: int = 0,
    world_model_type: str = "simple_kinematic",
    world_model: str | None = None,
    planner: str | None = None,
    planner_horizon: int = 10,
    planner_samples: int = 128,
    planner_iterations: int = 4,
    dataset_root: str | None = None,
    sequence_id: str | None = None,
    adapter: str | None = None,
    load_assets: bool = False,
) -> StreamingResponse:
    scenario_path = _resolve_config("scenarios", scenario)
    request = RunEpisodeRequest(
        backend=backend,
        scenario=scenario,
        agent=agent,
        seed=seed,
        max_steps=max_steps,
        record=record,
        record_arrays=record_arrays,
        world_model_type=world_model_type,
        world_model=world_model,
        planner=planner,
        planner_horizon=planner_horizon,
        planner_samples=planner_samples,
        planner_iterations=planner_iterations,
        dataset_root=dataset_root,
        sequence_id=sequence_id,
        adapter=adapter,
        load_assets=load_assets,
    )

    def event_source() -> Any:
        try:
            for payload in stream_episode_events(
                backend_name=backend,
                scenario=_scenario_for_request(request, scenario_path),
                agent_name=agent,
                seed=seed,
                max_steps=max_steps,
                record=record,
                output_root=DEFAULT_OUTPUT_ROOT,
                record_arrays=record_arrays,
                delay_ms=delay_ms,
                backend_options=_backend_options(request),
                agent_options=_agent_options(request),
            ):
                yield _sse(payload.get("event", "message"), payload)
        except Exception as exc:
            yield _sse("error", {"event": "error", "detail": str(exc)})

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/episodes")
def episodes() -> list[dict[str, Any]]:
    if not DEFAULT_OUTPUT_ROOT.exists():
        return []
    rows = []
    for path in sorted(DEFAULT_OUTPUT_ROOT.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        metadata = _read_json(path / "metadata.json")
        metrics = _read_json(path / "metrics.json")
        rows.append(
            {
                "episode_id": metadata.get("episode_id", path.name),
                "path": str(path),
                "metadata": metadata,
                "metrics": metrics,
            }
        )
    return rows


@app.get("/episodes/{episode_id}")
def episode_detail(episode_id: str) -> dict[str, Any]:
    path = _episode_path(episode_id)
    return {
        "episode_id": episode_id,
        "path": str(path),
        "metadata": _read_json(path / "metadata.json"),
        "metrics": _read_json(path / "metrics.json"),
        "steps_preview": _read_steps_preview(path / "steps.jsonl"),
    }


@app.get("/episodes/{episode_id}/steps")
def episode_steps(
    episode_id: str,
    limit: int = 5000,
    include_arrays: bool = True,
) -> list[dict[str, Any]]:
    path = _episode_path(episode_id)
    return _read_steps(path, limit=limit, include_arrays=include_arrays)


@app.get("/metrics/{episode_id}")
def episode_metrics(episode_id: str) -> dict[str, Any]:
    return _read_json(_episode_path(episode_id) / "metrics.json")


def _config_entries(root: Path, id_field: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*.yaml")):
        data = load_yaml_file(path)
        rows.append(
            {
                "id": str(data.get(id_field, path.stem)),
                "path": str(path),
                "summary": data,
            }
        )
    return rows


def _resolve_config(kind: str, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.exists():
        return raw_path
    candidate = CONFIG_ROOT / kind / f"{value}.yaml"
    if candidate.exists():
        return candidate
    raise HTTPException(status_code=404, detail=f"Unknown {kind[:-1]} config: {value}")


def _backend_options(request: RunEpisodeRequest) -> dict[str, Any]:
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


def _agent_options(request: RunEpisodeRequest) -> dict[str, Any]:
    if request.agent != "world_model":
        return {}
    options: dict[str, Any] = {"world_model_name": request.world_model_type}
    if request.world_model:
        options["world_model_path"] = request.world_model
    if request.planner:
        options["planner_name"] = request.planner
        options["planner_config"] = {
            "horizon": request.planner_horizon,
            "num_samples": request.planner_samples,
            "iterations": request.planner_iterations,
        }
    return options


def _scenario_for_request(request: RunEpisodeRequest, scenario_path: Path) -> Any:
    if request.backend == "dataset_replay" and request.dataset_root:
        return {
            "scenario_id": f"dataset_{Path(request.dataset_root).name}",
            "dataset_root": request.dataset_root,
            "sequence_id": request.sequence_id,
            "adapter": request.adapter,
            "load_assets": request.load_assets,
        }
    return scenario_path


def _episode_path(episode_id: str) -> Path:
    path = (DEFAULT_OUTPUT_ROOT / episode_id).resolve()
    output_root = DEFAULT_OUTPUT_ROOT.resolve()
    if output_root not in path.parents and path != output_root:
        raise HTTPException(status_code=400, detail="Invalid episode_id.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_steps_preview(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for _, line in zip(range(limit), file):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_steps(path: Path, limit: int, include_arrays: bool) -> list[dict[str, Any]]:
    steps_path = path / "steps.jsonl"
    if not steps_path.exists():
        return []
    rows = []
    with steps_path.open("r", encoding="utf-8") as file:
        for _, line in zip(range(max(limit, 0)), file):
            if not line.strip():
                continue
            record = json.loads(line)
            observation = record.get("observation")
            if isinstance(observation, dict):
                record["observation"] = recorded_observation_payload(
                    observation,
                    episode_path=path,
                    include_arrays=include_arrays,
                )
            rows.append(record)
    return rows


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=_json_default, separators=(',', ':'))}\n\n"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
