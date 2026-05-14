"""Create a tiny ORFD-like dataset for local phase-three smoke tests."""

from __future__ import annotations

import argparse

from offroad_sim.datasets import create_mock_orfd_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--split", default="training")
    parser.add_argument("--sequence-id", default="seq_0001")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    root = create_mock_orfd_dataset(
        args.output,
        split=args.split,
        sequence_id=args.sequence_id,
        frame_count=args.frames,
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
