"""Command line tools for OffroadSimBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from offroad_sim.agents.basic import make_agent
from offroad_sim.backends import default_backend_registry
from offroad_sim.evaluation import run_episode
from offroad_sim.replay import load_episode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offroad-sim", description="OffroadSimBench command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List scenarios, vehicles, agents, and backends.")
    list_parser.add_argument(
        "--kind",
        choices=["all", "scenarios", "vehicles", "agents", "backends"],
        default="all",
    )

    run_parser = subparsers.add_parser("run", help="Run one local benchmark episode.")
    run_parser.add_argument("--backend", default="gym_heightmap")
    run_parser.add_argument("--scenario", default=str(CONFIG_ROOT / "scenarios" / "forest_trail_001.yaml"))
    run_parser.add_argument("--agent", default="rule_based")
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--max-steps", type=int, default=1200)
    run_parser.add_argument("--record", nargs="?", const="auto", default=None)
    run_parser.add_argument("--record-arrays", action="store_true")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    replay_parser = subparsers.add_parser("replay", help="Inspect a saved episode directory.")
    replay_parser.add_argument("episode_path")
    replay_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "list":
        _print_listing(console, args.kind)
        return 0
    if args.command == "run":
        episode_path = None if args.record in {None, "auto"} else Path(args.record)
        result = run_episode(
            backend_name=args.backend,
            scenario=args.scenario,
            agent_name=args.agent,
            seed=args.seed,
            max_steps=args.max_steps,
            record=args.record is not None,
            episode_path=episode_path,
            record_arrays=args.record_arrays,
        )
        if args.json:
            console.print_json(json.dumps(result.to_dict(), default=str))
        else:
            _print_metrics(console, result.to_dict())
        return 0
    if args.command == "replay":
        player = load_episode(args.episode_path)
        payload = {
            "metadata": player.metadata,
            "metrics": player.get_metrics(),
        }
        if args.json:
            console.print_json(json.dumps(payload, default=str))
        else:
            console.print(f"Episode: {Path(args.episode_path)}")
            _print_metrics(console, payload["metrics"])
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _print_listing(console: Console, kind: str) -> None:
    if kind in {"all", "scenarios"}:
        _print_path_table(console, "Scenarios", CONFIG_ROOT / "scenarios", "*.yaml")
    if kind in {"all", "vehicles"}:
        _print_path_table(console, "Vehicles", CONFIG_ROOT / "vehicles", "*.yaml")
    if kind in {"all", "agents"}:
        table = Table(title="Agents")
        table.add_column("Name")
        table.add_column("Status")
        for name in ["random", "stop", "rule_based", "world_model"]:
            make_agent(name)
            table.add_row(name, "available")
        console.print(table)
    if kind in {"all", "backends"}:
        registry = default_backend_registry()
        table = Table(title="Backends")
        table.add_column("Name")
        table.add_column("Available")
        table.add_column("Message")
        for name, status in registry.status().items():
            table.add_row(name, str(status.available), status.message)
        console.print(table)


def _print_path_table(console: Console, title: str, root: Path, pattern: str) -> None:
    table = Table(title=title)
    table.add_column("ID")
    table.add_column("Path")
    for path in sorted(root.glob(pattern)):
        table.add_row(path.stem, str(path))
    console.print(table)


def _print_metrics(console: Console, payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", payload)
    table = Table(title=f"Episode {payload.get('episode_id', '')}".strip())
    table.add_column("Metric")
    table.add_column("Value")
    for key in [
        "success",
        "done",
        "total_reward",
        "episode_length",
        "steps",
        "elapsed_time_sec",
        "path_length",
        "average_speed",
        "collision_count",
        "average_terrain_risk",
        "distance_to_goal",
    ]:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            value = f"{value:.3f}"
        table.add_row(key, str(value))
    if payload.get("episode_path"):
        table.add_row("episode_path", str(payload["episode_path"]))
    console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
