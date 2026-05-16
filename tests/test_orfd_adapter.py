from __future__ import annotations

import zipfile

import numpy as np

from offroad_sim.datasets import create_mock_orfd_dataset, default_dataset_registry
from scripts.export_lewm_hdf5 import _load_image


def test_orfd_adapter_reads_mock_layout(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "orfd", frame_count=4)
    registry = default_dataset_registry()

    adapter = registry.resolve(dataset_root)
    sequence_ids = adapter.list_sequences(dataset_root)
    sequence = adapter.load_sequence(dataset_root, sequence_ids[0])

    assert adapter.name == "orfd"
    assert sequence_ids == ["training/seq_0001"]
    assert sequence.dataset_type == "orfd"
    assert len(sequence.frames) == 4
    assert sequence.frames[0].metadata["pose_source"] == "synthetic_index_order"
    assert sequence.frames[0].available_assets()["front_rgb"].endswith(".npy")


def test_orfd_adapter_reads_zip_sequence(tmp_path) -> None:
    source_root = create_mock_orfd_dataset(tmp_path / "source", frame_count=3)
    sequence_dir = source_root / "training" / "seq_0001"
    zip_root = tmp_path / "orfd_zip"
    zip_root.joinpath("training").mkdir(parents=True)
    zip_path = zip_root / "training" / "seq_0001.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sequence_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(sequence_dir.parent).as_posix())

    adapter = default_dataset_registry().resolve(zip_root, "orfd")
    sequence_ids = adapter.list_sequences(zip_root)
    sequence = adapter.load_sequence(zip_root, sequence_ids[0])

    assert sequence_ids == ["training/seq_0001"]
    assert sequence.metadata["source_layout"] == "orfd_zip"
    assert len(sequence.frames) == 3
    assert sequence.frames[0].available_assets()["front_rgb"].startswith("zip://")


def test_stablewm_export_loads_png_from_zip(tmp_path) -> None:
    from PIL import Image

    zip_path = tmp_path / "seq.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        image = Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8), mode="RGB")
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        archive.writestr("seq/image_data/000001.png", buffer.getvalue())

    pixels = _load_image(f"zip://{zip_path}!seq/image_data/000001.png")

    assert pixels.shape == (2, 3, 3)
    assert pixels.dtype == np.uint8
