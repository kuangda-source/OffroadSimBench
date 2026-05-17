from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import h5py

from desktop_app import services
from offroad_sim.algorithms import DataPrepResult, TrainResult
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.replay import EpisodeRecorder
from scripts import export_episodes_hdf5


def _observation(step: int) -> Observation:
    return Observation(
        timestamp=float(step),
        vehicle_state=VehicleState(x=float(step), y=0.0, z=100.0, yaw=0.0, speed=float(step)),
        goal=(5.0, 0.0),
        info={"backend": "beamng"},
    )


def test_export_episodes_hdf5_can_derive_actions_from_state(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": "beamng_collect", "backend": "beamng"})
    for step in range(4):
        recorder.record_step(
            observation=_observation(step),
            action=Action(steer=0.0, throttle=0.0, brake=1.0),
            reward=0.0,
            done=False,
            info={},
        )
    recorder.end_episode({"horizontal_distance_traveled": 3.0})
    recorder.save(episode_dir)

    output_hdf5 = tmp_path / "beamng_map.h5"
    old_argv = sys.argv
    try:
        sys.argv = [
            "export_episodes_hdf5.py",
            str(episode_dir),
            str(output_hdf5),
            "--actions-from-state",
        ]
        export_episodes_hdf5.main()
    finally:
        sys.argv = old_argv

    with h5py.File(output_hdf5, "r") as h5:
        assert h5.attrs["action_source"] == "state_delta"
        assert h5["action"].shape == (4, 3)
        assert float(h5["action"][0, 1]) > 0.0
        assert float(h5["action"][0, 2]) == 0.0


def test_beamng_map_lewm_closed_loop_pipeline_orchestrates_steps(tmp_path: Path) -> None:
    collect_episode = tmp_path / "collect_episode"
    collect_episode.mkdir()
    eval_episode = tmp_path / "eval_episode"
    eval_episode.mkdir()

    def fake_visible(request: services.VisibleBeamNGDemoRequest) -> dict[str, object]:
        if request.world_model_type == "simple_kinematic":
            return {
                "episode_id": "collect",
                "episode_path": str(collect_episode),
                "metrics": {"horizontal_distance_traveled": 18.0},
                "visible_demo": {"world_model_type": request.world_model_type},
            }
        return {
            "episode_id": "eval",
            "episode_path": str(eval_episode),
            "metrics": {"horizontal_distance_traveled": 22.0, "agent_diagnostics": {"planner": request.planner}},
            "visible_demo": {"world_model_type": request.world_model_type, "planner": request.planner},
        }

    class FakeAlgorithm:
        algorithm_id = "fake_lewm"

        def __init__(self) -> None:
            self.prepare_requests = []
            self.train_requests = []

        def prepare_data(self, request):
            self.prepare_requests.append(request)
            return DataPrepResult(output_path=str(tmp_path / "map.h5"), metadata={"total_frames": 20})

        def train(self, request):
            self.train_requests.append(request)
            return TrainResult(output_dir=str(tmp_path / "model"), checkpoint_path=str(tmp_path / "model" / "lewm_cost_object.ckpt"))

    fake_algorithm = FakeAlgorithm()

    with (
        patch("desktop_app.services.run_visible_beamng_demo", side_effect=fake_visible) as visible,
        patch("desktop_app.services.make_algorithm_adapter", return_value=fake_algorithm) as make_algorithm,
    ):
        payload = services.run_beamng_map_lewm_closed_loop(
            services.BeamNGMapLeWMClosedLoopRequest(
                algorithm="fake_lewm",
                output_dir=str(tmp_path / "closed_loop"),
                collect_steps=20,
                eval_steps=10,
                close_beamng=True,
            )
        )

    assert payload["status"] == "completed"
    assert payload["collection"]["episode_id"] == "collect"
    assert payload["evaluation"]["episode_id"] == "eval"
    assert payload["hdf5"]["total_frames"] == 20
    assert payload["training"]["output_dir"] == str(tmp_path / "model")
    assert payload["algorithm"] == "fake_lewm"
    assert visible.call_count == 2
    make_algorithm.assert_called_once_with("fake_lewm")
    assert fake_algorithm.prepare_requests[0].actions_from_state is True
    assert fake_algorithm.train_requests[0].input_path == str(tmp_path / "map.h5")
    assert json.loads((Path(payload["output_dir"]) / "closed_loop_summary.json").read_text(encoding="utf-8"))["status"] == "completed"
