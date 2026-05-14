"""Streaming episode execution helpers for local demos and dashboards."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from offroad_sim.agents import make_agent
from offroad_sim.core import Action, Observation
from offroad_sim.evaluation.runner import DEFAULT_OUTPUT_ROOT, _episode_id, _load_scenario
from offroad_sim.evaluation.runner import _agent_diagnostics, _create_backend, _scenario_id
from offroad_sim.replay import EpisodeRecorder
from offroad_sim.scenarios import ScenarioConfig


TERRAIN_LAYERS = ("height", "traversability", "risk", "occupancy")
BEV_LAYERS = ("height", "traversability", "risk", "occupancy")


def stream_episode_events(
    backend_name: str = "gym_heightmap",
    scenario: ScenarioConfig | str | Path | Mapping[str, Any] | None = None,
    agent_name: str = "rule_based",
    seed: int = 7,
    max_steps: int = 1200,
    record: bool = True,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    record_arrays: bool = True,
    delay_ms: int = 0,
    backend_options: Mapping[str, Any] | None = None,
    agent_options: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run an episode and yield JSON-serializable progress events."""

    scenario_config = _load_scenario(scenario)
    backend = _create_backend(backend_name, seed=seed, options=backend_options)
    agent = make_agent(agent_name, seed=seed, **dict(agent_options or {}))
    recorder = EpisodeRecorder(save_arrays=record_arrays) if record else None
    scenario_id = _scenario_id(scenario_config)
    episode_id = _episode_id(scenario_id, agent_name)
    result = None
    steps = 0
    done = False
    saved_path: Path | None = None

    try:
        obs = backend.reset(scenario_config)
        agent.reset({"scenario_id": scenario_id, "backend": backend_name})
        if recorder is not None:
            recorder.start_episode(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "backend": backend_name,
                    "agent": agent_name,
                    "seed": seed,
                    "streamed": True,
                    "agent_options": dict(agent_options or {}),
                    "backend_options": dict(backend_options or {}),
                }
            )

        yield {
            "event": "start",
            "episode_id": episode_id,
            "scenario_id": scenario_id,
            "backend": backend_name,
            "agent": agent_name,
            "seed": seed,
            "max_steps": max_steps,
            "frame": frame_payload(
                step_index=0,
                observation=obs,
                action=None,
                reward=0.0,
                done=False,
                info=obs.info,
                include_terrain=True,
            ),
        }

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
            done = result.done
            include_terrain = steps == 1 or steps % 10 == 0 or done
            metrics = backend.get_metrics()
            metrics.update(
                {
                    "done": done,
                    "terminated": bool(result.terminated),
                    "truncated": bool(result.truncated),
                    "steps": steps,
                    "backend": backend_name,
                    "agent": agent_name,
                    "scenario_id": scenario_id,
                }
            )
            if diagnostics:
                metrics["agent_diagnostics"] = diagnostics

            yield {
                "event": "step",
                "episode_id": episode_id,
                "frame": frame_payload(
                    step_index=steps,
                    observation=obs,
                    action=action,
                    reward=result.reward,
                    done=done,
                    info=step_info,
                    include_terrain=include_terrain,
                ),
                "metrics": metrics,
            }
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            if done:
                break

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

        if recorder is not None:
            recorder.end_episode(metrics)
            saved_path = recorder.save(Path(output_root) / episode_id)

        yield {
            "event": "end",
            "episode_id": episode_id,
            "metrics": metrics,
            "episode_path": str(saved_path) if saved_path else None,
        }
    finally:
        agent.close()
        backend.close()


def frame_payload(
    *,
    step_index: int,
    observation: Observation,
    action: Action | None,
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
    include_terrain: bool = False,
) -> dict[str, Any]:
    """Build the frame shape consumed by the dashboard."""

    return {
        "step_index": int(step_index),
        "observation": observation_payload(
            observation,
            include_terrain=include_terrain,
            include_local_bev=True,
        ),
        "action": action_payload(action),
        "reward": float(reward),
        "done": bool(done),
        "info": dict(info or {}),
    }


def observation_payload(
    observation: Observation,
    *,
    include_terrain: bool = False,
    include_local_bev: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": float(observation.timestamp),
        "vehicle_state": {
            "x": float(observation.vehicle_state.x),
            "y": float(observation.vehicle_state.y),
            "z": float(observation.vehicle_state.z),
            "yaw": float(observation.vehicle_state.yaw),
            "pitch": float(observation.vehicle_state.pitch),
            "roll": float(observation.vehicle_state.roll),
            "speed": float(observation.vehicle_state.speed),
        },
        "goal": [float(observation.goal[0]), float(observation.goal[1])],
        "info": dict(observation.info),
        "local_bev": None,
        "terrain_map": None,
    }
    if include_local_bev and observation.local_bev is not None:
        payload["local_bev"] = array_layers_payload(observation.local_bev, BEV_LAYERS, max_size=25)
    if include_terrain and observation.terrain_map is not None:
        payload["terrain_map"] = array_layers_payload(observation.terrain_map, TERRAIN_LAYERS, max_size=48)
    return payload


def recorded_observation_payload(
    observation: dict[str, Any],
    episode_path: Path,
    *,
    include_arrays: bool = True,
) -> dict[str, Any]:
    payload = dict(observation)
    if not include_arrays:
        payload["local_bev"] = None
        payload["terrain_map"] = None
        return payload

    for field_name, layers, max_size in (
        ("local_bev", BEV_LAYERS, 25),
        ("terrain_map", TERRAIN_LAYERS, 48),
    ):
        value = payload.get(field_name)
        if isinstance(value, str):
            array_path = (episode_path / value).resolve()
            if episode_path.resolve() in array_path.parents and array_path.exists():
                payload[field_name] = array_layers_payload(np.load(array_path), layers, max_size=max_size)
            else:
                payload[field_name] = None
        elif value is None:
            payload[field_name] = None
    return payload


def array_layers_payload(value: Any, layer_names: tuple[str, ...], max_size: int) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        layer_arrays = [array]
        names = [layer_names[0] if layer_names else "value"]
    elif array.ndim == 3 and array.shape[0] <= 16:
        layer_arrays = [array[index] for index in range(array.shape[0])]
        names = [layer_names[index] if index < len(layer_names) else f"layer_{index}" for index in range(array.shape[0])]
    elif array.ndim == 3 and array.shape[-1] <= 16:
        layer_arrays = [array[:, :, index] for index in range(array.shape[-1])]
        names = [layer_names[index] if index < len(layer_names) else f"layer_{index}" for index in range(array.shape[-1])]
    else:
        layer_arrays = [array.reshape(array.shape[-2], array.shape[-1])]
        names = [layer_names[0] if layer_names else "value"]

    layers: dict[str, Any] = {}
    output_shape: list[int] | None = None
    for name, layer in zip(names, layer_arrays):
        sampled = _downsample_2d(layer, max_size=max_size)
        output_shape = [int(sampled.shape[0]), int(sampled.shape[1])]
        layers[name] = np.round(sampled.astype(float), 4).tolist()

    return {
        "shape": output_shape or [0, 0],
        "layers": layers,
    }


def action_payload(action: Action | None) -> dict[str, float] | None:
    if action is None:
        return None
    return {
        "steer": float(action.steer),
        "throttle": float(action.throttle),
        "brake": float(action.brake),
    }


def _downsample_2d(array: np.ndarray, max_size: int) -> np.ndarray:
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array for dashboard payload, got shape {array.shape}")
    height, width = array.shape
    stride = max(1, int(np.ceil(max(height, width) / max_size)))
    return array[::stride, ::stride]
