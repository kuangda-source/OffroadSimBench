from __future__ import annotations

from offroad_sim.backends import MockUE5Server, UE5Backend
from offroad_sim.core import Action
from offroad_sim.scenarios import load_scenario_config


def test_ue5_backend_round_trips_with_mock_server() -> None:
    scenario = load_scenario_config("configs/scenarios/forest_trail_001.yaml")

    with MockUE5Server() as server:
        backend = UE5Backend(host=server.host, port=server.port)
        obs = backend.reset(scenario)
        result = backend.step(Action(throttle=1.0, steer=0.1))

        assert obs.goal == scenario.task.goal
        assert result.observation.timestamp > obs.timestamp
        assert result.observation.vehicle_state.x > obs.vehicle_state.x
        assert result.info["backend"] == "ue5_mock"
        assert backend.get_metrics()["connected"] is True
        assert backend.get_metrics()["episode_length"] == 1
        backend.close()
        assert backend.get_metrics()["connected"] is False
