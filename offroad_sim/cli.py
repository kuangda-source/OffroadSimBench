"""Command line tools for OffroadSimBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from offroad_sim.agents import default_agent_registry, make_agent
from offroad_sim.backends import default_backend_registry
from offroad_sim.datasets import default_dataset_registry
from offroad_sim.evaluation import run_episode
from offroad_sim.planning import default_planner_registry
from offroad_sim.replay import load_episode
from offroad_sim.world_models import default_world_model_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="offroad-sim", description="OffroadSimBench command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List scenarios, vehicles, agents, and backends.")
    list_parser.add_argument(
        "--kind",
        choices=["all", "scenarios", "vehicles", "agents", "backends", "datasets", "world_models", "planners"],
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
    run_parser.add_argument("--world-model-type", default="simple_kinematic")
    run_parser.add_argument("--world-model", default=None, help="Path to a saved world model.")
    run_parser.add_argument("--planner", default=None, help="Planner name, e.g. world_model_cem or le_wm_cem.")
    run_parser.add_argument("--planner-horizon", type=int, default=10)
    run_parser.add_argument("--planner-samples", type=int, default=128)
    run_parser.add_argument("--planner-iterations", type=int, default=4)
    run_parser.add_argument("--dataset-root", default=None, help="Dataset root for dataset_replay backend.")
    run_parser.add_argument("--sequence-id", default=None)
    run_parser.add_argument("--adapter", default=None, help="Dataset adapter name.")
    run_parser.add_argument("--load-assets", action="store_true")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect a dataset adapter and sequence.")
    inspect_parser.add_argument("dataset_root")
    inspect_parser.add_argument("--adapter", default=None)
    inspect_parser.add_argument("--sequence-id", default=None)
    inspect_parser.add_argument("--json", action="store_true")

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
            scenario=_scenario_for_run(args),
            agent_name=args.agent,
            seed=args.seed,
            max_steps=args.max_steps,
            record=args.record is not None,
            episode_path=episode_path,
            record_arrays=args.record_arrays,
            backend_options=_backend_options(args),
            agent_options=_agent_options(args),
        )
        if args.json:
            console.print_json(json.dumps(result.to_dict(), default=str))
        else:
            _print_metrics(console, result.to_dict())
        return 0
    if args.command == "inspect-dataset":
        payload = _inspect_dataset(args.dataset_root, adapter_name=args.adapter, sequence_id=args.sequence_id)
        if args.json:
            console.print_json(json.dumps(payload, default=str))
        else:
            _print_dataset_inspection(console, payload)
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
        table.add_column("Description")
        registry = default_agent_registry()
        for name in registry.names():
            table.add_row(name, registry.get(name).description)
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
    if kind in {"all", "datasets"}:
        table = Table(title="Dataset Adapters")
        table.add_column("Name")
        for name in default_dataset_registry().names():
            table.add_row(name)
        console.print(table)
    if kind in {"all", "world_models"}:
        table = Table(title="World Models")
        table.add_column("Name")
        table.add_column("Available")
        table.add_column("Message")
        registry = default_world_model_registry()
        for name, status in registry.status().items():
            table.add_row(name, str(status.available), status.message)
        console.print(table)
    if kind in {"all", "planners"}:
        table = Table(title="Planners")
        table.add_column("Name")
        table.add_column("Available")
        table.add_column("Message")
        registry = default_planner_registry()
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


def _backend_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.backend == "dataset_replay":
        if args.dataset_root:
            options["dataset_root"] = args.dataset_root
        if args.sequence_id:
            options["sequence_id"] = args.sequence_id
        if args.adapter:
            options["adapter"] = args.adapter
        options["load_assets"] = bool(args.load_assets)
    return options


def _agent_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.agent not in {"world_model", "route_world_model"}:
        return {}
    options: dict[str, Any] = {"world_model_name": args.world_model_type}
    if args.world_model:
        options["world_model_path"] = args.world_model
    if args.planner:
        options["planner_name"] = args.planner
        options["planner_config"] = {
            "horizon": args.planner_horizon,
            "num_samples": args.planner_samples,
            "iterations": args.planner_iterations,
        }
    return options


def _scenario_for_run(args: argparse.Namespace) -> Any:
    if args.backend == "dataset_replay" and args.dataset_root:
        return {
            "scenario_id": f"dataset_{Path(args.dataset_root).name}",
            "dataset_root": args.dataset_root,
            "sequence_id": args.sequence_id,
            "adapter": args.adapter,
            "load_assets": bool(args.load_assets),
        }
    return args.scenario


def _inspect_dataset(dataset_root: str, *, adapter_name: str | None = None, sequence_id: str | None = None) -> dict[str, Any]:
    registry = default_dataset_registry()
    adapter = registry.resolve(dataset_root, adapter_name)
    sequences = adapter.list_sequences(dataset_root)
    selected_sequence = sequence_id or sequences[0]
    sequence = adapter.load_sequence(dataset_root, selected_sequence)
    first = sequence.frames[0]
    last = sequence.frames[-1]
    asset_counts: dict[str, int] = {}
    for frame in sequence.frames:
        for name in frame.available_assets():
            asset_counts[name] = asset_counts.get(name, 0) + 1
    return {
        "dataset_root": str(Path(dataset_root).resolve()),
        "adapter": adapter.name,
        "sequences": sequences,
        "selected_sequence": selected_sequence,
        "dataset_id": sequence.dataset_id,
        "dataset_type": sequence.dataset_type,
        "frame_count": len(sequence.frames),
        "asset_counts": asset_counts,
        "first_frame": {"frame_id": first.frame_id, "timestamp": first.timestamp, "metadata": first.metadata},
        "last_frame": {"frame_id": last.frame_id, "timestamp": last.timestamp, "metadata": last.metadata},
        "metadata": sequence.metadata,
    }


def _print_dataset_inspection(console: Console, payload: dict[str, Any]) -> None:
    table = Table(title=f"Dataset {payload['dataset_id']}")
    table.add_column("Field")
    table.add_column("Value")
    for key in ["dataset_root", "adapter", "selected_sequence", "dataset_type", "frame_count"]:
        table.add_row(key, str(payload[key]))
    table.add_row("sequences", ", ".join(payload["sequences"]))
    table.add_row("asset_counts", json.dumps(payload["asset_counts"], default=str))
    console.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
