from __future__ import annotations

from offroad_sim.backends import DatasetReplayBackend
from offroad_sim.core import Action
from offroad_sim.datasets import create_mock_dataset, default_dataset_registry


def test_registry_autodetects_manifest_dataset(tmp_path) -> None:
    dataset_root = create_mock_dataset(tmp_path / "dataset", frame_count=4)
    registry = default_dataset_registry()

    adapter = registry.resolve(dataset_root)
    sequence_ids = adapter.list_sequences(dataset_root)
    sequence = adapter.load_sequence(dataset_root, sequence_ids[0])

    assert adapter.name == "offroad_sim_v1"
    assert sequence_ids == ["seq_0001"]
    assert len(sequence) == 4
    assert sequence.dataset_id == "mock_offroad"
    assert sequence.frames[0].available_assets()["front_rgb"].endswith(".npy")


def test_dataset_replay_backend_steps_through_sequence(tmp_path) -> None:
    dataset_root = create_mock_dataset(tmp_path / "dataset", frame_count=5)
    backend = DatasetReplayBackend(dataset_root, load_assets=True)

    obs = backend.reset()
    assert obs.info["frame_id"] == "frame_000000"
    assert obs.front_rgb.shape == (8, 8, 3)

    done = False
    steps = 0
    while not done:
        result = backend.step(Action(throttle=1.0))
        done = result.done
        steps += 1

    metrics = backend.get_metrics()
    assert steps == 4
    assert result.terminated is True
    assert result.info["action_ignored"] is True
    assert metrics["frames_total"] == 5
    assert metrics["frames_played"] == 5
    assert metrics["episode_length"] == 4
    assert metrics["done"] is True


def test_dataset_replay_backend_can_switch_sequence_at_reset(tmp_path) -> None:
    dataset_root = create_mock_dataset(tmp_path / "dataset", sequence_id="seq_a", frame_count=2)
    create_mock_dataset(dataset_root, sequence_id="seq_b", frame_count=3)
    backend = DatasetReplayBackend(dataset_root, sequence_id="seq_a")

    first = backend.reset()
    second = backend.reset({"sequence_id": "seq_b", "adapter": "offroad_sim_v1"})

    assert first.info["sequence_id"] == "seq_a"
    assert second.info["sequence_id"] == "seq_b"
    assert backend.get_metrics()["frames_total"] == 3
