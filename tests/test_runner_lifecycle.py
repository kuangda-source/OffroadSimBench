from __future__ import annotations

from offroad_sim.core import Action, Observation, StepResult, VehicleState
from offroad_sim.evaluation import runner


class FakeBackend:
    def __init__(self) -> None:
        self.closed = False
        self.steps = 0
        self.held = False

    def reset(self, scenario_config=None) -> Observation:
        return Observation(timestamp=0.0, vehicle_state=VehicleState(), goal=(1.0, 0.0), info={})

    def step(self, action: Action) -> StepResult:
        self.steps += 1
        obs = Observation(
            timestamp=float(self.steps),
            vehicle_state=VehicleState(x=float(self.steps)),
            goal=(1.0, 0.0),
            info={},
        )
        return StepResult(observation=obs, reward=0.0, terminated=False, truncated=False, info={})

    def get_metrics(self) -> dict[str, object]:
        return {"steps_seen": self.steps}

    def close(self) -> None:
        self.closed = True

    def hold_vehicle(self) -> None:
        self.held = True


class FakeAgent:
    def __init__(self) -> None:
        self.closed = False

    def reset(self, scenario_info) -> None:
        return None

    def act(self, obs: Observation) -> Action:
        return Action(throttle=0.1)

    def close(self) -> None:
        self.closed = True


def test_run_episode_can_pace_and_leave_backend_open(monkeypatch) -> None:
    backend = FakeBackend()
    agent = FakeAgent()
    sleeps: list[float] = []
    monkeypatch.setattr(runner, "_create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(runner, "make_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    result = runner.run_episode(
        backend_name="beamng",
        scenario={"scenario_id": "visible"},
        max_steps=2,
        pre_run_hold_sec=8.0,
        step_delay_sec=0.05,
        close_backend=False,
    )

    assert result.steps == 2
    assert sleeps == [8.0, 0.05, 0.05]
    assert agent.closed is True
    assert backend.closed is False


def test_run_episode_holds_vehicle_before_visual_hold(monkeypatch) -> None:
    backend = FakeBackend()
    agent = FakeAgent()
    sleeps: list[float] = []
    monkeypatch.setattr(runner, "_create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(runner, "make_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(runner.time, "sleep", lambda value: sleeps.append(value))

    runner.run_episode(
        backend_name="beamng",
        scenario={"scenario_id": "visible"},
        max_steps=1,
        post_run_hold_sec=3.0,
        close_backend=False,
    )

    assert backend.held is True
    assert sleeps == [3.0]
    assert backend.closed is False
