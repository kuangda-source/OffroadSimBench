from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from offroad_sim.replay import EpisodePlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved OffroadSimBench episode.")
    parser.add_argument("episode", help="Episode directory created by examples/run_gym_demo.py --record")
    parser.add_argument("--max-steps", type=int, default=8, help="Number of steps to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = EpisodePlayer.load(args.episode)
    console = Console()

    console.print("Metadata:")
    console.print(player.metadata)
    console.print("Metrics:")
    console.print(player.get_metrics())

    table = Table(title="Replay Steps")
    table.add_column("step")
    table.add_column("time")
    table.add_column("x")
    table.add_column("y")
    table.add_column("speed")
    table.add_column("reward")
    table.add_column("done")

    for index, step in enumerate(player.iter_steps()):
        if index >= args.max_steps:
            break
        obs = step["observation"]
        state = obs["vehicle_state"]
        table.add_row(
            str(step["step_index"]),
            f"{obs['timestamp']:.2f}",
            f"{state['x']:.2f}",
            f"{state['y']:.2f}",
            f"{state['speed']:.2f}",
            f"{step['reward']:.3f}",
            str(step["done"]),
        )

    console.print(table)


if __name__ == "__main__":
    main()

