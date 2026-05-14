"""Reusable episode runner for CLI, API, and smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offroad_sim.agents import make_agent
from offroad_sim.backends.registry import make_backend
from offroad_sim.replay import EpisodeRecorder
from offroad_sim.scenarios import ScenarioConfig, load_scenario_config


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
    scenario: ScenarioConfig | str | Path | None = None,
    agent_name: str = "rule_based",
    seed: int = 7,
    max_steps: int = 1200,
    record: bool = False,
    episode_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    record_arrays: bool = False,
) -> EpisodeRunResult:
    """Run one local benchmark episode.

    First-stage automated runs are intentionally limited to the local
    gym_heightmap backend. BeamNG and UE5 expose the same backend interface but
    need external runtimes and are validated through connection/status checks.
    """

    if backend_name != "gym_heightmap":
        raise ValueError("Automated local episode runs currently support backend_name='gym_heightmap'.")

    scenario_config = _load_scenario(scenario)
    backend = make_backend(backend_name, seed=seed)
    agent = make_agent(agent_name, seed=seed)
    recorder = EpisodeRecorder(save_arrays=record_arrays) if record else None
    episode_id = _episode_id(scenario_config.scenario_id, agent_name)
    result = None
    steps = 0
    done = False

    try:
        obs = backend.reset(scenario_config)
        agent.reset({"scenario_id": scenario_config.scenario_id, "backend": backend_name})
        if recorder is not None:
            recorder.start_episode(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_config.scenario_id,
                    "backend": backend_name,
                    "agent": agent_name,
                    "seed": seed,
                }
            )

        for steps in range(1, max_steps + 1):
            action = agent.act(obs)
            result = backend.step(action)
            obs = result.observation
            if recorder is not None:
                recorder.record_step(
                    observation=obs,
                    action=action,
                    reward=result.reward,
                    done=result.done,
                    info=result.info,
                )
            if result.done:
                done = True
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
                "scenario_id": scenario_config.scenario_id,
            }
        )

        saved_path = None
        if recorder is not None:
            recorder.end_episode(metrics)
            saved_path = Path(episode_path) if episode_path is not None else Path(output_root) / episode_id
            saved_path = recorder.save(saved_path)

        return EpisodeRunResult(
            episode_id=episode_id,
            metrics=metrics,
            backend=backend_name,
            agent=agent_name,
            scenario_id=scenario_config.scenario_id,
            steps=steps,
            done=done,
            terminated=bool(result.terminated) if result else False,
            truncated=bool(result.truncated) if result else False,
            episode_path=saved_path,
        )
    finally:
        agent.close()
        backend.close()


def _load_scenario(scenario: ScenarioConfig | str | Path | None) -> ScenarioConfig:
    if isinstance(scenario, ScenarioConfig):
        return scenario
    return load_scenario_config(scenario or DEFAULT_SCENARIO_PATH)


def _episode_id(scenario_id: str, agent_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{scenario_id}_{agent_name}_{timestamp}"
