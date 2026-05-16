from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from offroad_sim.utils.runtime_env import prepare_stable_worldmodel_runtime


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("h5py") is None,
    reason="LE-WM smoke dependencies are not installed.",
)


def test_lewm_cost_model_checkpoint_roundtrip(tmp_path: Path) -> None:
    prepare_stable_worldmodel_runtime()
    import h5py  # type: ignore
    import torch
    from stable_worldmodel.policy import AutoCostModel

    from scripts.train_lewm_cost_model import main as train_main

    hdf5_path = tmp_path / "tiny.h5"
    with h5py.File(hdf5_path, "w") as h5:
        h5.create_dataset("ep_len", data=np.asarray([3], dtype=np.int64))
        h5.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset(
            "state",
            data=np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 1.2],
                    [2.2, 0.0, 0.0, 0.0, 1.2],
                ],
                dtype=np.float32,
            ),
        )
        h5.create_dataset("action", data=np.asarray([[0.0, 0.5, 0.0], [0.0, 0.4, 0.0], [0.0, 0.4, 0.0]], dtype=np.float32))
        h5.create_dataset("goal", data=np.asarray([[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]], dtype=np.float32))

    old_argv = os.sys.argv
    try:
        os.sys.argv = ["train_lewm_cost_model.py", str(hdf5_path), "--output", str(tmp_path / "model")]
        assert train_main() == 0
    finally:
        os.sys.argv = old_argv

    model = AutoCostModel(str(tmp_path / "model"))
    info = {
        "state": torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        "goal_state": torch.tensor([[[[4.0, 0.0]], [[4.0, 0.0]]]], dtype=torch.float32),
    }
    candidates = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    costs = model.get_cost(info, candidates)
    assert costs.shape == (1, 2)
    assert torch.isfinite(costs).all()


def test_train_fit_config_respects_episode_boundaries() -> None:
    from scripts.train_lewm_cost_model import _fit_config

    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [100.0, 0.0, 0.0, 0.0, 1.0],
            [101.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([[0.0, 0.2, 0.0]] * 4, dtype=np.float32)
    config = _fit_config(states, actions, np.asarray([2, 2]), np.asarray([0, 2]))

    assert config.dt == pytest.approx(1.0)


def test_episode_export_goal_pair_accepts_common_shapes() -> None:
    from scripts.export_episodes_hdf5 import _goal_pair

    vehicle = {"x": 3.0, "y": 4.0}

    assert _goal_pair({"x": 1, "y": 2}, vehicle) == [1.0, 2.0]
    assert _goal_pair(np.asarray([[5.0, 6.0]]), vehicle) == [5.0, 6.0]
    assert _goal_pair("unknown", vehicle) == [3.0, 4.0]
