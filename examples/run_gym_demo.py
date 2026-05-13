from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from offroad_sim.agents import make_agent
from offroad_sim.backends import GymHeightmapBackend
from offroad_sim.replay import EpisodeRecorder
from offroad_sim.scenarios import load_scenario_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a GymHeightmapBackend demo episode.")
    parser.add_argument(
        "--scenario",
        default=str(ROOT / "configs" / "scenarios" / "forest_trail_001.yaml"),
        help="Path to scenario YAML.",
    )
    parser.add_argument(
        "--agent",
        choices=["random", "stop", "rule_based"],
        default="rule_based",
        help="Agent to run.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Backend and random-agent seed.")
    parser.add_argument("--max-steps", type=int, default=1200, help="Safety cap for the episode loop.")
    parser.add_argument("--record", default=None, help="Optional output directory for episode recording.")
    parser.add_argument(
        "--record-arrays",
        action="store_true",
        help="Persist observation arrays such as local BEV as .npy files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = load_scenario_config(args.scenario)
    backend = GymHeightmapBackend(seed=args.seed)
    agent = make_agent(args.agent, seed=args.seed)
    console = Console()

    obs = backend.reset(scenario)
    agent.reset({"scenario_id": scenario.scenario_id, "backend": scenario.backend})
    recorder = EpisodeRecorder(save_arrays=args.record_arrays) if args.record else None
    if recorder is not None:
        recorder.start_episode(
            {
                "scenario_id": scenario.scenario_id,
                "backend": scenario.backend,
                "agent": args.agent,
                "seed": args.seed,
            }
        )

    done = False
    result = None
    for _ in range(args.max_steps):
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
    metrics["done"] = done
    metrics["terminated"] = bool(result.terminated) if result else False
    metrics["truncated"] = bool(result.truncated) if result else False

    table = Table(title=f"GymHeightmapBackend Episode: {scenario.scenario_id}")
    table.add_column("Metric")
    table.add_column("Value")
    for key in [
        "success",
        "done",
        "total_reward",
        "episode_length",
        "elapsed_time_sec",
        "time_to_goal",
        "path_length",
        "average_speed",
        "max_speed",
        "collision_count",
        "rollover",
        "max_pitch",
        "max_roll",
        "average_terrain_risk",
        "control_smoothness",
        "distance_to_goal",
    ]:
        value = metrics.get(key)
        if isinstance(value, float):
            value = f"{value:.3f}"
        table.add_row(key, str(value))

    console.print(table)
    if recorder is not None:
        recorder.end_episode(metrics)
        output_path = recorder.save(args.record)
        console.print(f"Saved episode to {output_path}")
    backend.close()
    agent.close()


if __name__ == "__main__":
    main()
