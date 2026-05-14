from __future__ import annotations

import pytest

from offroad_sim.backends import BeamNGBackend, BeamNGUnavailableError


def test_beamng_backend_is_import_safe_without_runtime() -> None:
    backend = BeamNGBackend()
    status = BeamNGBackend.runtime_status()

    assert status.name == "beamng"
    assert isinstance(status.available, bool)
    assert backend.get_metrics()["connected"] is False


def test_beamng_connect_reports_clear_optional_dependency_error() -> None:
    status = BeamNGBackend.runtime_status()
    if status.available:
        pytest.skip("BeamNG runtime appears available; skip non-launching unavailable check.")

    backend = BeamNGBackend()
    with pytest.raises(BeamNGUnavailableError, match="BeamNGBackend is optional"):
        backend.connect()
