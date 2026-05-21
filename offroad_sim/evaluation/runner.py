"""Reusable episode runner for CLI, API, and smoke tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from offroad_sim.agents import make_agent
from offroad_sim.backends.registry import make_backend
from offroad_sim.replay import EpisodeRecorder
from offroad_sim.scenarios import ScenarioConfig, load_scenario_config
from offroad_sim.vehicles import VehicleConfig, load_vehicle_config


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_PATH = ROOT / "configs" / "scenarios" / "forest_trail_001.yaml"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "episodes"


@dataclass(slots=True)
class EpisodeRunResult:
    episode_id: str
    metrics: dict[str, Any]
    backend: str
    agent: str
    scenario_id: str
    steps: int
    done: bool
    terminated: bool
    truncated: bool
    episode_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "metrics": self.metrics,
            "backend": self.backend,
            "agent": self.agent,
            "scenario_id": self.scenario_id,
            "steps": self.steps,
            "done": self.done,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_path": str(self.episode_path) if self.episode_path else None,
        }


def run_episode(
    backend_name: str = "gym_heightmap",
    scenario: ScenarioConfig | str | Path | Mapping[str, Any] | None = None,
    agent_name: str = "rule_based",
    seed: int = 7,
    max_steps: int = 1200,
    record: bool = False,
    episode_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    record_arrays: bool = False,
    backend_options: Mapping[str, Any] | None = None,
    agent_options: Mapping[str, Any] | None = None,
    vehicle: VehicleConfig | str | Path | None = None,
    pre_run_hold_sec: float = 0.0,
    step_delay_sec: float = 0.0,
    post_run_hold_sec: float = 0.0,
    close_backend: bool = True,
) -> EpisodeRunResult:
    """Run one benchmark episode through any registered backend and agent."""

    scenario_config = _load_scenario(scenario)
    vehicle_config = _load_vehicle(vehicle)
    backend = _create_backend(backend_name, seed=seed, options=backend_options, vehicle_config=vehicle_config)
    agent = make_agent(agent_name, seed=seed, **dict(agent_options or {}))
    recorder = EpisodeRecorder(save_arrays=record_arrays) if record else None
    scenario_id = _scenario_id(scenario_config)
    episode_id = _episode_id(scenario_id, agent_name)
    result = None
    steps = 0
    done = False
    completed = False

    try:
        obs = backend.reset(scenario_config)
        agent.reset({"scenario_id": scenario_id, "backend": backend_name})
        if pre_run_hold_sec > 0.0:
            time.sleep(float(pre_run_hold_sec))
        if recorder is not None:
            recorder.start_episode(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "backend": backend_name,
                    "agent": agent_name,
                    "seed": seed,
                    "agent_options": dict(agent_options or {}),
                    "backend_options": dict(backend_options or {}),
                    "vehicle": vehicle_config.vehicle_id if vehicle_config is not None else None,
                }
            )

        for steps in range(1, max_steps + 1):
            action = agent.act(obs)
            result = backend.step(action)
            obs = result.observation
            step_info = dict(result.info)
            diagnostics = _agent_diagnostics(agent)
            if diagnostics:
                step_info["agent_diagnostics"] = diagnostics
            if recorder is not None:
                recorder.record_step(
                    observation=obs,
                    action=action,
                    reward=result.reward,
                    done=result.done,
                    info=step_info,
                )
            if result.done:
                done = True
                break
            if step_delay_sec > 0.0:
                time.sleep(float(step_delay_sec))

        metrics = backend.get_metrics()
        metrics.update(
            {
                "done": done,
                "terminated": bool(result.terminated) if result else False,
                "truncated": bool(result.truncated) if result else False,
                "steps": steps,
                "backend": backend_name,
                "agent": agent_name,
                "scenario_id": scenario_id,
            }
        )
        diagnostics = _agent_diagnostics(agent)
        if diagnostics:
            metrics["agent_diagnostics"] = diagnostics

        saved_path = None
        if recorder is not None:
            recorder.end_episode(metrics)
            saved_path = Path(episode_path) if episode_path is not None else Path(output_root) / episode_id
            saved_path = recorder.save(saved_path)

        completed = True
        return EpisodeRunResult(
            episode_id=episode_id,
            metrics=metrics,
            backend=backend_name,
            agent=agent_name,
            scenario_id=scenario_id,
            steps=steps,
            done=done,
            terminated=bool(result.terminated) if result else False,
            truncated=bool(result.truncated) if result else False,
            episode_path=saved_path,
        )
    finally:
        agent.close()
        if completed and (post_run_hold_sec > 0.0 or not close_backend):
            _hold_backend_vehicle(backend)
        if completed and post_run_hold_sec > 0.0:
            time.sleep(float(post_run_hold_sec))
        if close_backend:
            backend.close()


def _load_scenario(scenario: ScenarioConfig | str | Path | Mapping[str, Any] | None) -> Any:
    if isinstance(scenario, ScenarioConfig):
        return scenario
    if isinstance(scenario, Mapping):
        return dict(scenario)
    return load_scenario_config(scenario or DEFAULT_SCENARIO_PATH)


def _episode_id(scenario_id: str, agent_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario_id}_{agent_name}_{timestamp}"


def _scenario_id(scenario: Any) -> str:
    if isinstance(scenario, Mapping):
        return str(scenario.get("scenario_id", "adhoc_scenario"))
    return str(getattr(scenario, "scenario_id", "adhoc_scenario"))


def _load_vehicle(vehicle: VehicleConfig | str | Path | None) -> VehicleConfig | None:
    if vehicle is None:
        return None
    if isinstance(vehicle, VehicleConfig):
        return vehicle
    return load_vehicle_config(vehicle)


def _create_backend(
    name: str,
    *,
    seed: int,
    options: Mapping[str, Any] | None = None,
    vehicle_config: VehicleConfig | None = None,
) -> Any:
    kwargs = dict(options or {})
    if name == "beamng" and vehicle_config is not None:
        kwargs["vehicle_config"] = vehicle_config
    try:
        return make_backend(name, seed=seed, **kwargs)
    except TypeError as exc:
        if "seed" not in str(exc):
            raise
        return make_backend(name, **kwargs)


def _hold_backend_vehicle(backend: Any) -> None:
    hold_vehicle = getattr(backend, "hold_vehicle", None)
    if not callable(hold_vehicle):
        return
    try:
        hold_vehicle()
    except Exception:
        return


def _agent_diagnostics(agent: Any) -> dict[str, Any]:
    diagnostics = getattr(agent, "diagnostics", None)
    if not callable(diagnostics):
        return {}
    value = diagnostics()
    return dict(value) if isinstance(value, Mapping) else {}
