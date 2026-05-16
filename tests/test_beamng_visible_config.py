from __future__ import annotations

from offroad_sim.scenarios import load_scenario_config
from offroad_sim.scenarios.config import scenario_metadata_section


def test_beamng_visible_scenario_exposes_route_metadata() -> None:
    scenario = load_scenario_config("configs/scenarios/beamng_visible_autodrive.yaml")

    assert scenario.backend == "beamng"
    assert scenario.map == "gridmap_v2"
    beamng = scenario_metadata_section(scenario, "beamng")
    assert beamng["level"] == "gridmap_v2"
    assert len(beamng["route"]) >= 3
    assert beamng["camera_mode"] == "orbit"
    assert beamng["draw_route"] is True
