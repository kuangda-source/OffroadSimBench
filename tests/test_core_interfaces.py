from __future__ import annotations

from typing import Any

from offroad_sim.agents import OffroadAgent
from offroad_sim.backends import OffroadSimBackend
from offroad_sim.core import Action, EpisodeInfo, Observation, StepResult, VehicleState


class ConstantAgent(OffroadAgent):
    def act(self, obs: Observation) -> Action:
        return Action(steer=0.1, throttle=0.3, brake=0.0)


class EchoBackend(OffroadSimBackend):
    def __init__(self) -> None:
        self.observation = Observation(
            timestamp=0.0,
            vehicle_state=VehicleState(),
            goal=(10.0, 0.0),
        )
        self.steps = 0

    def reset(self, scenario_config: Any) -> Observation:
        self.steps = 0
        return self.observation

    def step(self, action: Action) -> StepResult:
        self.steps += 1
        return StepResult(
            observation=self.observation,
            reward=1.0,
            terminated=False,
            truncated=False,
            info={"action": action},
        )

    def get_observation(self) -> Observation:
        return self.observation

    def get_metrics(self) -> dict[str, Any]:
        return {"steps": self.steps}

    def close(self) -> None:
        return None


def test_core_data_types_can_be_instantiated() -> None:
    state = VehicleState(x=1.0, y=2.0, yaw=0.5, speed=3.0)
    obs = Observation(timestamp=1.25, vehicle_state=state, goal=(5.0, 6.0))
    action = Action.from_mapping({"steer": "0.2", "throttle": 0.4, "gear": -1})
    result = StepResult(obs, reward=1.0, terminated=False, truncated=True)
    episode = EpisodeInfo(episode_id="episode_001", scenario_id="forest_trail_001")

    assert action.steer == 0.2
    assert action.brake == 0.0
    assert action.gear == -1
    assert result.done is True
    assert episode.scenario_id == "forest_trail_001"


def test_agent_and_backend_interfaces_work_together() -> None:
    backend = EchoBackend()
    agent = ConstantAgent()

    obs = backend.reset({"scenario_id": "test"})
    agent.reset({"scenario_id": "test"})
    action = agent.act(obs)
    result = backend.step(action)

    assert result.info["action"].throttle == 0.3
    assert backend.get_metrics() == {"steps": 1}
