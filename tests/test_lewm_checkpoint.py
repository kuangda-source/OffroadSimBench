from __future__ import annotations

from pathlib import Path

import pytest

from offroad_sim.core import Action, Observation, VehicleState
from offroad_sim.planning.lewm_checkpoint import (
    LeWMCheckpointFormatError,
    normalize_lewm_checkpoint_reference,
)
from offroad_sim.world_models.le_wm import LeWMWorldModel
from scripts.convert_lewm_hf_checkpoint import _resolve_output_path


def test_normalize_lewm_checkpoint_accepts_direct_object_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_a" / "lewm_cost_object.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")

    ref = normalize_lewm_checkpoint_reference(checkpoint)

    assert ref.run_name == str(checkpoint.with_name("lewm_cost"))
    assert ref.object_checkpoint == checkpoint
    assert ref.source_kind == "stablewm_object_file"


def test_normalize_lewm_checkpoint_resolves_existing_relative_object_file(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "run_rel" / "lewm_cost_object.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.chdir(tmp_path)

    ref = normalize_lewm_checkpoint_reference(Path("run_rel") / "lewm_cost_object.ckpt")

    assert ref.run_name == str(checkpoint.with_name("lewm_cost").resolve())
    assert ref.object_checkpoint == checkpoint.resolve()


def test_normalize_lewm_checkpoint_accepts_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    checkpoint = run_dir / "epoch_001_object.ckpt"
    checkpoint.write_bytes(b"placeholder")

    ref = normalize_lewm_checkpoint_reference(run_dir)

    assert ref.run_name == str(run_dir)
    assert ref.object_checkpoint == checkpoint
    assert ref.source_kind == "stablewm_run_dir"


def test_normalize_lewm_checkpoint_rejects_hf_weight_directory_until_converted(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf_lewm"
    hf_dir.mkdir()
    (hf_dir / "weights.pt").write_bytes(b"placeholder")
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LeWMCheckpointFormatError, match="convert_lewm_hf_checkpoint.py"):
        normalize_lewm_checkpoint_reference(hf_dir)


def test_convert_lewm_hf_checkpoint_output_path_defaults_to_object_checkpoint(tmp_path: Path) -> None:
    assert _resolve_output_path(tmp_path / "run") == tmp_path / "run" / "lewm_object.ckpt"
    assert _resolve_output_path(tmp_path / "custom_object.ckpt") == tmp_path / "custom_object.ckpt"


def test_lewm_world_model_status_reports_direct_object_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "lewm_status_object.ckpt"
    checkpoint.write_bytes(b"placeholder")

    status = LeWMWorldModel.runtime_status(checkpoint)

    assert status["details"]["checkpoint_run_name"] == str(checkpoint.with_name("lewm_status"))
    assert status["details"]["checkpoint_object_path"] == str(checkpoint)


def test_stablewm_lewm_adapter_scores_direct_object_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from offroad_sim.algorithms import ScoreActionsRequest
    from offroad_sim.algorithms.builtins.stablewm_lewm import StableWMLeWMAlgorithm
    from offroad_sim.planning.lewm_cost_model import LeWMCostModel

    checkpoint = tmp_path / "lewm_real_object.ckpt"
    torch.save(LeWMCostModel().eval(), checkpoint)
    adapter = StableWMLeWMAlgorithm()
    adapter.load(checkpoint)

    result = adapter.score_actions(
        ScoreActionsRequest(
            observation=Observation(timestamp=0.0, vehicle_state=VehicleState(), goal=(4.0, 0.0)),
            action_candidates=[
                [Action(throttle=0.8), Action(throttle=0.8)],
                [Action(brake=1.0), Action(brake=1.0)],
            ],
        )
    )

    assert len(result.costs) == 2
    assert result.metadata["source_kind"] == "stablewm_object_file"
    assert result.metadata["object_checkpoint"] == str(checkpoint)
