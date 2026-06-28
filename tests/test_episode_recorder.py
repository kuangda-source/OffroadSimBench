from __future__ import annotations

from pathlib import Path

import numpy as np

from offroad_sim.backends import BeamNGConnectionConfig
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.replay import EpisodePlayer, EpisodeRecorder


def make_observation(step: int) -> Observation:
    return Observation(
        timestamp=float(step),
        vehicle_state=VehicleState(x=float(step), y=0.0, speed=1.0),
        goal=(10.0, 0.0),
        local_bev=np.zeros((4, 3, 3), dtype=np.float32),
        info={"terrain_risk": 0.2},
    )


def test_episode_recorder_saves_and_player_loads(tmp_path: Path) -> None:
    recorder = EpisodeRecorder()
    recorder.start_episode({"scenario_id": "forest_trail_001", "agent": "rule_based"})
    recorder.record_step(
        observation=make_observation(1),
        action=Action(throttle=0.5, gear=-1),
        reward=1.25,
        done=False,
        info={"distance_to_goal": 9.0},
    )
    recorder.end_episode({"success": False, "total_reward": 1.25})

    episode_path = recorder.save(tmp_path / "episode_001")
    player = EpisodePlayer.load(episode_path)
    steps = list(player.iter_steps())

    assert player.metadata["scenario_id"] == "forest_trail_001"
    assert player.metadata["step_count"] == 1
    assert player.get_metrics()["total_reward"] == 1.25
    assert steps[0]["action"]["throttle"] == 0.5
    assert steps[0]["action"]["gear"] == -1
    assert steps[0]["observation"]["vehicle_state"]["x"] == 1.0


def test_episode_recorder_can_save_arrays(tmp_path: Path) -> None:
    recorder = EpisodeRecorder(save_arrays=True)
    recorder.start_episode({"scenario_id": "array_test"})
    recorder.record_step(
        observation=make_observation(1),
        action=Action(throttle=0.5),
        reward=1.0,
        done=True,
        info={},
    )
    recorder.end_episode({"success": True})

    episode_path = recorder.save(tmp_path / "episode_arrays")
    step = next(EpisodePlayer.load(episode_path).iter_steps())

    local_bev_path = episode_path / step["observation"]["local_bev"]
    assert local_bev_path.exists()
    assert np.load(local_bev_path).shape == (4, 3, 3)


def test_episode_recorder_serializes_dataclass_metadata(tmp_path: Path) -> None:
    recorder = EpisodeRecorder()
    recorder.start_episode(
        {
            "scenario_id": "beamng_visible_autodrive",
            "backend_options": {"connection": BeamNGConnectionConfig(gfx="vk")},
        }
    )
    recorder.end_episode({"success": True})

    episode_path = recorder.save(tmp_path / "episode_dataclass")
    player = EpisodePlayer.load(episode_path)

    assert player.metadata["backend_options"]["connection"]["gfx"] == "vk"
