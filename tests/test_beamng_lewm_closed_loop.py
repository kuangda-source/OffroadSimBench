from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import h5py

from desktop_app import services
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

    with (
        patch("desktop_app.services.run_visible_beamng_demo", side_effect=fake_visible) as visible,
        patch("desktop_app.services.export_episodes_hdf5", return_value={"output_hdf5": str(tmp_path / "map.h5"), "total_frames": 20}, create=True) as export,
        patch("desktop_app.services.train_lewm_cost_model", return_value={"output_dir": str(tmp_path / "model"), "checkpoint_path": str(tmp_path / "model" / "lewm_cost_object.ckpt")}) as train,
    ):
        payload = services.run_beamng_map_lewm_closed_loop(
            services.BeamNGMapLeWMClosedLoopRequest(
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
    assert visible.call_count == 2
    assert export.call_args.kwargs["actions_from_state"] is True
    assert json.loads((Path(payload["output_dir"]) / "closed_loop_summary.json").read_text(encoding="utf-8"))["status"] == "completed"
