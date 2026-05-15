"""Export registered dataset sequences to the HDF5 boundary expected by LE-WM-style pipelines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from offroad_sim.datasets import default_dataset_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_hdf5")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--sequence-id", action="append", dest="sequence_ids", default=None)
    parser.add_argument("--image-size", type=int, default=64)
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

    ep_lengths: list[int] = []
    ep_offsets: list[int] = []
    state_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    goal_chunks: list[np.ndarray] = []
    pixel_chunks: list[np.ndarray] = []
    exported_sequence_ids: list[str] = []
    sequence_metadata: dict[str, object] = {}
    total_frames = 0
    all_pixels_available = True

    for sequence_id in sequence_ids:
        sequence = adapter.load_sequence(args.dataset_root, sequence_id)
        if not sequence.frames:
            continue
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
        goal = np.asarray(sequence.goal or (states[-1, 0], states[-1, 1]), dtype=np.float32)
        pixels = _load_pixels(sequence.frames, image_size=args.image_size)
        ep_offsets.append(total_frames)
        ep_lengths.append(len(states))
        state_chunks.append(states)
        action_chunks.append(_pseudo_actions(states))
        timestamp_chunks.append(np.asarray([frame.timestamp for frame in sequence.frames], dtype=np.float32))
        goal_chunks.append(np.repeat(goal[None, :], len(states), axis=0))
        if pixels is None:
            all_pixels_available = False
        else:
            pixel_chunks.append(pixels)
        exported_sequence_ids.append(sequence_id)
        sequence_metadata[sequence_id] = {
            "metadata": sequence.metadata,
            "asset_paths": [frame.available_assets() for frame in sequence.frames],
        }
        total_frames += len(states)

    if not state_chunks:
        raise SystemExit("No frames were exported. Check the dataset root, adapter, or sequence filter.")

    with h5py.File(output, "w") as h5:
        h5.attrs["schema"] = "stable_worldmodel_flat_v1"
        h5.attrs["source_dataset_root"] = str(Path(args.dataset_root).resolve())
        h5.attrs["adapter"] = adapter.name
        h5.attrs["sequence_ids"] = json.dumps(exported_sequence_ids)
        h5.attrs["sequence_metadata"] = json.dumps(sequence_metadata, default=str)
        h5.create_dataset("ep_len", data=np.asarray(ep_lengths, dtype=np.int64))
        h5.create_dataset("ep_offset", data=np.asarray(ep_offsets, dtype=np.int64))
        h5.create_dataset("state", data=np.concatenate(state_chunks, axis=0), compression="gzip")
        h5.create_dataset("action", data=np.concatenate(action_chunks, axis=0), compression="gzip")
        h5.create_dataset("timestamp", data=np.concatenate(timestamp_chunks, axis=0), compression="gzip")
        h5.create_dataset("goal", data=np.concatenate(goal_chunks, axis=0), compression="gzip")
        if all_pixels_available and pixel_chunks:
            h5.create_dataset("pixels", data=np.concatenate(pixel_chunks, axis=0), compression="gzip")

    print(
        json.dumps(
            {
                "output_hdf5": str(output.resolve()),
                "schema": "stable_worldmodel_flat_v1",
                "adapter": adapter.name,
                "sequence_ids": exported_sequence_ids,
                "total_frames": total_frames,
                "pixels_exported": bool(all_pixels_available and pixel_chunks),
            },
            indent=2,
        )
    )
    return 0


def _pseudo_actions(states: np.ndarray) -> np.ndarray:
    if len(states) < 2:
        return np.zeros((len(states), 3), dtype=np.float32)
    rows = []
    for current, nxt in zip(states, states[1:]):
        speed_delta = float(nxt[4] - current[4])
        yaw_delta = _wrap_angle(float(nxt[3] - current[3]))
        rows.append(
            [
                float(np.clip(yaw_delta / 0.25, -1.0, 1.0)),
                float(np.clip(speed_delta / 2.0 + 0.35, 0.0, 1.0)),
                float(np.clip(-speed_delta / 2.0, 0.0, 1.0)),
            ]
        )
    rows.append(rows[-1])
    return np.asarray(rows, dtype=np.float32)


def _load_pixels(frames: list[object], *, image_size: int) -> np.ndarray | None:
    images = []
    for frame in frames:
        path = getattr(frame, "front_rgb_path", None)
        if path is None or Path(path).suffix.lower() != ".npy":
            return None
        image = np.load(path)
        if image.ndim != 3 or image.shape[-1] < 3:
            return None
        images.append(_resize_nearest(image[..., :3], image_size).astype(np.uint8))
    return np.stack(images, axis=0) if images else None


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    rows = np.linspace(0, height - 1, size).astype(int)
    cols = np.linspace(0, width - 1, size).astype(int)
    return image[rows][:, cols]


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


if __name__ == "__main__":
    raise SystemExit(main())
