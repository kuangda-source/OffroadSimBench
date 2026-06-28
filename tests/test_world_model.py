from __future__ import annotations

import numpy as np

from offroad_sim.agents import WorldModelAgent, make_agent
from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.datasets import DatasetFrame, DatasetSequence, create_mock_orfd_dataset, default_dataset_registry
from offroad_sim.world_models import SimpleKinematicWorldModel, TinyLearnedWorldModel, default_world_model_registry, make_world_model


def _observation(risk: float = 0.0) -> Observation:
    terrain = np.zeros((4, 32, 32), dtype=np.float32)
    terrain[2, :, :] = risk
    return Observation(
        timestamp=0.0,
        vehicle_state=VehicleState(x=4.0, y=4.0, yaw=0.0, speed=1.0),
        goal=(20.0, 4.0),
        local_bev=np.zeros((4, 25, 25), dtype=np.float32),
        terrain_map=terrain,
    )


def test_simple_kinematic_world_model_predicts_forward_motion() -> None:
    model = SimpleKinematicWorldModel()
    prediction = model.predict(_observation(), Action(throttle=0.5), horizon=5)

    assert len(prediction.states) == 5
    assert prediction.final_state is not None
    assert prediction.final_state.x > 4.0
    assert prediction.metadata["horizon"] == 5


def test_simple_kinematic_world_model_save_load_roundtrip(tmp_path) -> None:
    model = SimpleKinematicWorldModel(dt=0.2)
    path = model.save(tmp_path / "model.json")

    loaded = SimpleKinematicWorldModel.load(path)

    assert loaded.dt == 0.2


def test_world_model_agent_slows_down_on_high_predicted_risk() -> None:
    agent = WorldModelAgent(risk_threshold=0.5)
    action = agent.act(_observation(risk=1.0))

    assert action.throttle <= 0.22
    assert action.brake >= 0.12


def test_make_agent_supports_world_model() -> None:
    assert isinstance(make_agent("world_model"), WorldModelAgent)


def test_world_model_registry_exposes_switchable_models() -> None:
    registry = default_world_model_registry()

    assert {"simple_kinematic", "tiny_learned", "le_wm"}.issubset(set(registry.names()))
    assert registry.status("tiny_learned").available is True


def test_tiny_learned_world_model_trains_and_loads(tmp_path) -> None:
    dataset_root = create_mock_orfd_dataset(tmp_path / "orfd", frame_count=5)
    adapter = default_dataset_registry().resolve(dataset_root, "orfd")
    sequence = adapter.load_sequence(dataset_root, "training/seq_0001")
    model = TinyLearnedWorldModel.fit([sequence])
    model_path = model.save(tmp_path / "tiny_model")

    loaded = make_world_model("tiny_learned", path=model_path.parent)
    prediction = loaded.predict(_observation(), Action(throttle=0.4), horizon=3)

    assert prediction.final_state is not None
    assert prediction.metadata["model_type"] == "tiny_learned"
    assert model.metadata["sample_count"] == 4


def test_tiny_learned_world_model_uses_recorded_transition_actions() -> None:
    sequence = DatasetSequence(
        dataset_id="beamng_episode",
        dataset_type="beamng_episode",
        sequence_id="recorded_actions",
        root=".",
        frames=[
            DatasetFrame("000000", 0.0, VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0), action=Action(throttle=0.0)),
            DatasetFrame("000001", 1.0, VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0), action=Action(throttle=1.0)),
            DatasetFrame("000002", 2.0, VehicleState(x=1.0, y=0.0, yaw=0.0, speed=1.0), action=Action(throttle=1.0)),
        ],
    )

    model = TinyLearnedWorldModel.fit([sequence], ridge=1e-8)
    prediction = model.predict(
        Observation(timestamp=0.0, vehicle_state=VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0), goal=(10.0, 0.0)),
        Action(throttle=1.0),
        horizon=1,
    )

    assert prediction.final_state is not None
    assert prediction.final_state.x > 0.2
    assert model.metadata["recorded_action_sample_count"] == 2


def test_tiny_learned_world_model_records_gear_feature() -> None:
    sequence = DatasetSequence(
        dataset_id="beamng_episode",
        dataset_type="beamng_episode",
        sequence_id="reverse_actions",
        root=".",
        frames=[
            DatasetFrame("000000", 0.0, VehicleState(x=0.0, y=0.0, yaw=0.0, speed=0.0), action=Action(throttle=0.0, gear=1)),
            DatasetFrame("000001", 1.0, VehicleState(x=-0.5, y=0.0, yaw=0.0, speed=0.3), action=Action(throttle=0.5, gear=-1)),
            DatasetFrame("000002", 2.0, VehicleState(x=0.2, y=0.0, yaw=0.0, speed=0.5), action=Action(throttle=0.6, gear=1)),
        ],
    )

    model = TinyLearnedWorldModel.fit([sequence], ridge=1e-4)

    assert "gear" in model.metadata["feature_names"]
    assert model.weights.shape[0] == len(model.metadata["feature_names"])


def test_tiny_learned_world_model_loads_legacy_weights_without_gear(tmp_path) -> None:
    model_dir = tmp_path / "legacy_tiny"
    model_dir.mkdir()
    np.savez(model_dir / "weights.npz", weights=np.zeros((7, 4), dtype=np.float64))
    (model_dir / "model.json").write_text(
        """
{
  "model_type": "tiny_learned",
  "config": {
    "dt": 1.0,
    "ridge": 0.0001,
    "metadata": {"feature_names": ["bias", "speed", "yaw_sin", "yaw_cos", "steer", "throttle", "brake"]}
  },
  "weights": "weights.npz"
}
""",
        encoding="utf-8",
    )

    loaded = TinyLearnedWorldModel.load(model_dir)

    assert loaded.weights.shape == (8, 4)
    assert np.allclose(loaded.weights[-1], 0.0)
    assert "gear" in loaded.metadata["feature_names"]
