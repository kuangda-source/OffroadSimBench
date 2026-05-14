"""Inspect a dataset through the adapter registry."""

from __future__ import annotations

import argparse
import json

from offroad_sim.cli import _inspect_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", default=None)
    args = parser.parse_args()

    payload = _inspect_dataset(args.dataset_root, adapter_name=args.adapter, sequence_id=args.sequence_id)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
