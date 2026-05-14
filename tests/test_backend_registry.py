from __future__ import annotations

import pytest

from offroad_sim.backends import DatasetReplayBackend, GymHeightmapBackend, default_backend_registry, make_backend


def test_default_backend_registry_lists_all_backends() -> None:
    registry = default_backend_registry()

    assert registry.names() == ["beamng", "dataset_replay", "gym_heightmap", "ue5"]
    assert registry.status("gym_heightmap").available is True
    assert registry.status("dataset_replay").available is True


def test_backend_factory_creates_core_backends() -> None:
    assert isinstance(make_backend("gym_heightmap"), GymHeightmapBackend)
    assert isinstance(make_backend("dataset_replay"), DatasetReplayBackend)


def test_backend_factory_rejects_unknown_backend() -> None:
    with pytest.raises(KeyError, match="Unknown backend"):
        make_backend("missing")
