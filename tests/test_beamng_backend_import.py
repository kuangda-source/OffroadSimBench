from __future__ import annotations

import pytest

from offroad_sim.backends import BeamNGBackend, BeamNGUnavailableError
from offroad_sim.scenarios import load_scenario_config


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


def test_beamng_status_accepts_explicit_home(tmp_path) -> None:
    fake_home = tmp_path / "BeamNG.tech"
    fake_home.mkdir()
    (fake_home / "BeamNG.tech.exe").write_text("", encoding="utf-8")

    status = BeamNGBackend.runtime_status(fake_home)

    assert status.details is not None
    assert status.details["bng_home"] == str(fake_home)
    assert status.details["executable"] == str(fake_home / "BeamNG.tech.exe")


def test_beamng_uses_default_level_for_generic_scenario() -> None:
    backend = BeamNGBackend()
    scenario = load_scenario_config("configs/scenarios/forest_trail_001.yaml")

    assert backend._beamng_level_for_config(scenario) == backend.connection.level
