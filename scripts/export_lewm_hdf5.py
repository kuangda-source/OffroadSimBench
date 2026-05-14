"""Export registered dataset sequences to the HDF5 boundary expected by LE-WM-style pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from offroad_sim.datasets import default_dataset_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_hdf5")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids", default=None)
    args = parser.parse_args()

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required for HDF5 export. Install it in the active environment.") from exc

    registry = default_dataset_registry()
    adapter = registry.resolve(args.dataset_root, args.adapter)
    sequence_ids = args.sequence_ids or adapter.list_sequences(args.dataset_root)
    output = Path(args.output_hdf5)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output, "w") as h5:
        h5.attrs["source_dataset_root"] = str(Path(args.dataset_root).resolve())
        h5.attrs["adapter"] = adapter.name
        for sequence_id in sequence_ids:
            sequence = adapter.load_sequence(args.dataset_root, sequence_id)
            group = h5.create_group(sequence_id.replace("/", "__"))
            states = np.asarray(
                [
                    [
                        frame.vehicle_state.x,
                        frame.vehicle_state.y,
                        frame.vehicle_state.z,
                        frame.vehicle_state.yaw,
                        frame.vehicle_state.speed,
                    ]
                    for frame in sequence.frames
                ],
                dtype=np.float32,
            )
            group.create_dataset("states", data=states)
            group.create_dataset("timestamps", data=np.asarray([frame.timestamp for frame in sequence.frames], dtype=np.float32))
            group.attrs["metadata"] = json.dumps(sequence.metadata, default=str)
            group.attrs["asset_paths"] = json.dumps([frame.available_assets() for frame in sequence.frames], default=str)

    print(json.dumps({"output_hdf5": str(output.resolve()), "adapter": adapter.name, "sequence_ids": sequence_ids}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
