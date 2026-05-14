from __future__ import annotations

from offroad_sim.datasets import create_mock_orfd_dataset, default_dataset_registry


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
