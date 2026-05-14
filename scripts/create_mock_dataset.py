from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offroad_sim.datasets import create_mock_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small OffroadSimBench mock dataset.")
    parser.add_argument("output", help="Dataset root directory to create.")
    parser.add_argument("--dataset-id", default="mock_offroad", help="Dataset identifier written to dataset.yaml.")
    parser.add_argument("--sequence-id", default="seq_0001", help="Sequence identifier to create.")
    parser.add_argument("--frames", type=int, default=12, help="Number of frames in the sequence.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generated sensor arrays.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = create_mock_dataset(
        args.output,
        dataset_id=args.dataset_id,
        sequence_id=args.sequence_id,
        frame_count=args.frames,
        seed=args.seed,
    )
    print(f"Created mock dataset at {path}")


if __name__ == "__main__":
    main()
