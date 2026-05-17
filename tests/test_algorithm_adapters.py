from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from offroad_sim.algorithms import (
    AlgorithmAdapter,
    AlgorithmCapabilities,
    AlgorithmManifest,
    DataPrepRequest,
    ScoreActionsRequest,
    UnsupportedCapabilityError,
    default_algorithm_registry,
)
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.replay import EpisodeRecorder


def test_algorithm_manifest_parses_capabilities() -> None:
    manifest = AlgorithmManifest.from_dict(
        {
            "algorithm_id": "demo_model",
            "display_name": "Demo Model",
            "entrypoint": "adapter:DemoAlgorithm",
            "capabilities": {"train": True, "score_actions": True},
            "input_contract": {"observations": ["state", "goal"]},
            "output_contract": {"mode": "action_cost"},
        }
    )

    assert manifest.algorithm_id == "demo_model"
    assert manifest.display_name == "Demo Model"
    assert manifest.capabilities.train is True
    assert manifest.capabilities.score_actions is True
    assert manifest.capabilities.act is False


def test_algorithm_adapter_rejects_unsupported_capability() -> None:
    adapter = AlgorithmAdapter(
        AlgorithmManifest(
            algorithm_id="empty",
            display_name="Empty",
            entrypoint="adapter:Empty",
            capabilities=AlgorithmCapabilities(),
        )
    )

    with pytest.raises(UnsupportedCapabilityError, match="empty.*act"):
        adapter.act(None)  # type: ignore[arg-type]


def test_default_algorithm_registry_exposes_local_lewm_cost() -> None:
    registry = default_algorithm_registry()

    assert "local_lewm_cost" in registry.names()
    spec = registry.get("local_lewm_cost")
    assert spec.manifest.capabilities.train is True
    assert spec.manifest.capabilities.score_actions is True
    assert registry.status("local_lewm_cost").available is True


def test_algorithm_registry_discovers_local_package(tmp_path: Path) -> None:
    package = tmp_path / "algorithms" / "demo"
    package.mkdir(parents=True)
    (package / "algorithm.yaml").write_text(
        """
algorithm_id: demo_algorithm
display_name: Demo Algorithm
entrypoint: adapter:DemoAlgorithm
capabilities:
  act: true
""".strip(),
        encoding="utf-8",
    )
    (package / "adapter.py").write_text(
        """
from offroad_sim.algorithms import AlgorithmAdapter

class DemoAlgorithm(AlgorithmAdapter):
    pass
""".strip(),
        encoding="utf-8",
    )

    registry = default_algorithm_registry(search_paths=[tmp_path / "algorithms"])

    assert "demo_algorithm" in registry.names()
    assert registry.create("demo_algorithm").algorithm_id == "demo_algorithm"


def test_local_lewm_cost_adapter_prepares_hdf5_from_episode(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    recorder = EpisodeRecorder()
    recorder.start_episode({"episode_id": "adapter_collect", "backend": "beamng"})
    for step in range(3):
        recorder.record_step(
            observation=Observation(
                timestamp=float(step),
                vehicle_state=VehicleState(x=float(step), y=0.0, z=100.0, yaw=0.0, speed=float(step)),
                goal=(5.0, 0.0),
            ),
            action=Action(brake=1.0),
            reward=0.0,
            done=False,
            info={},
        )
    recorder.end_episode({"horizontal_distance_traveled": 2.0})
    recorder.save(episode_dir)

    adapter = default_algorithm_registry().create("local_lewm_cost")
    output_hdf5 = tmp_path / "prepared.h5"
    result = adapter.prepare_data(DataPrepRequest(episode_root=episode_dir, output_path=output_hdf5, actions_from_state=True))

    assert result.output_path == str(output_hdf5.resolve())
    assert result.metadata["total_frames"] == 3
    with h5py.File(output_hdf5, "r") as h5:
        assert h5.attrs["action_source"] == "state_delta"


def test_local_lewm_cost_adapter_scores_action_candidates(tmp_path: Path) -> None:
    import torch

    from offroad_sim.algorithms.builtins.local_lewm_cost import LocalLeWMCostAlgorithm
    from offroad_sim.planning.lewm_cost_model import LeWMCostModel

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save(LeWMCostModel().eval(), model_dir / "lewm_cost_object.ckpt")
    adapter = LocalLeWMCostAlgorithm()
    adapter.load(model_dir)
    observation = Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=1.0),
        goal=(10.0, 0.0),
    )

    result = adapter.score_actions(
        ScoreActionsRequest(
            observation=observation,
            action_candidates=[
                [Action(throttle=0.8), Action(throttle=0.8)],
                [Action(brake=1.0), Action(brake=1.0)],
            ],
        )
    )

    assert len(result.costs) == 2
    assert result.costs[0] < result.costs[1]
