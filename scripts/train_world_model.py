"""Train a small switchable world model from a registered dataset adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offroad_sim.datasets import default_dataset_registry
from offroad_sim.world_models import TinyLearnedWorldModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids", default=None)
    parser.add_argument("--output", default="outputs/models/tiny_world_model")
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    registry = default_dataset_registry()
    adapter = registry.resolve(args.dataset_root, args.adapter)
    sequence_ids = args.sequence_ids or adapter.list_sequences(args.dataset_root)[:1]
    sequences = [adapter.load_sequence(args.dataset_root, sequence_id) for sequence_id in sequence_ids]
    model = TinyLearnedWorldModel.fit(sequences, ridge=args.ridge)
    model.metadata.update(
        {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "adapter": adapter.name,
            "sequence_ids": sequence_ids,
        }
    )
    metadata_path = model.save(args.output)
    payload = {
        "model_type": model.model_type,
        "model_path": str(metadata_path),
        "output_dir": str(Path(args.output).resolve()),
        "metrics": model.metadata,
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
