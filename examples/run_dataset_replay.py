from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from offroad_sim.backends import DatasetReplayBackend
from offroad_sim.core import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a dataset sequence through DatasetReplayBackend.")
    parser.add_argument("dataset_root", help="Dataset root containing dataset.yaml or a recognizable layout.")
    parser.add_argument("--sequence", default=None, help="Sequence id to replay. Defaults to the first sequence.")
    parser.add_argument("--adapter", default=None, help="Optional adapter name override.")
    parser.add_argument("--load-assets", action="store_true", help="Load .npy sensor arrays into observations.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum replay steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = DatasetReplayBackend(
        args.dataset_root,
        sequence_id=args.sequence,
        adapter=args.adapter,
        load_assets=args.load_assets,
    )
    console = Console()

    obs = backend.reset()
    table = Table(title=f"DatasetReplayBackend: {obs.info['dataset_id']} / {obs.info['sequence_id']}")
    table.add_column("Step", justify="right")
    table.add_column("Frame")
    table.add_column("Time", justify="right")
    table.add_column("X", justify="right")
    table.add_column("Y", justify="right")
    table.add_column("Assets")

    steps_taken = 0
    while True:
        assets = ",".join(obs.info["assets"].keys()) or "none"
        table.add_row(
            str(steps_taken),
            obs.info["frame_id"],
            f"{obs.timestamp:.3f}",
            f"{obs.vehicle_state.x:.2f}",
            f"{obs.vehicle_state.y:.2f}",
            assets,
        )
        if obs.info["frame_index"] >= obs.info["frame_count"] - 1 or steps_taken >= args.max_steps:
            break
        result = backend.step(Action())
        obs = result.observation
        steps_taken += 1

    metrics = backend.get_metrics()
    console.print(table)
    console.print(metrics)
    backend.close()


if __name__ == "__main__":
    main()
