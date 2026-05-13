from __future__ import annotations

from offroad_sim.core import Action, Observation, StepResult, VehicleState
from offroad_sim.evaluation import MetricsTracker


def make_obs(timestamp: float, x: float, speed: float = 1.0, risk: float = 0.2) -> Observation:
    return Observation(
        timestamp=timestamp,
        vehicle_state=VehicleState(x=x, y=0.0, speed=speed, pitch=0.1, roll=0.05),
        goal=(10.0, 0.0),
        info={"terrain_risk": risk},
    )


def test_metrics_tracker_accumulates_basic_episode_metrics() -> None:
    tracker = MetricsTracker()
    obs0 = make_obs(0.0, 0.0, speed=1.0)
    obs1 = make_obs(1.0, 2.0, speed=2.0, risk=0.3)
    obs2 = make_obs(2.0, 5.0, speed=3.0, risk=0.5)

    tracker.update(
        obs0,
        Action(steer=0.0, throttle=0.5, brake=0.0),
        StepResult(obs1, reward=1.5, terminated=False, truncated=False, info={"terrain_risk": 0.3}),
    )
    tracker.update(
        obs1,
        Action(steer=0.2, throttle=0.4, brake=0.0),
        StepResult(
            obs2,
            reward=2.5,
            terminated=True,
            truncated=False,
            info={"terrain_risk": 0.5, "success": True},
        ),
    )

    metrics = tracker.compute()
    assert metrics["success"] is True
    assert metrics["total_reward"] == 4.0
    assert metrics["episode_length"] == 2
    assert metrics["time_to_goal"] == 2.0
    assert metrics["path_length"] == 5.0
    assert metrics["average_speed"] == 2.5
    assert metrics["max_speed"] == 3.0
    assert metrics["average_terrain_risk"] == 0.4
    assert metrics["control_smoothness"] > 0.0


def test_metrics_tracker_reset_clears_state() -> None:
    tracker = MetricsTracker()
    obs0 = make_obs(0.0, 0.0)
    obs1 = make_obs(1.0, 1.0)

    tracker.update(
        obs0,
        Action(throttle=0.5),
        StepResult(obs1, reward=1.0, terminated=False, truncated=False, info={"collision": True}),
    )
    tracker.reset()

    metrics = tracker.compute()
    assert metrics["episode_length"] == 0
    assert metrics["collision_count"] == 0
    assert metrics["total_reward"] == 0.0

