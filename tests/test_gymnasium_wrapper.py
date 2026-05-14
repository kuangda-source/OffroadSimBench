from __future__ import annotations

import numpy as np

from offroad_sim.rl import OffroadGymEnv


def test_gymnasium_wrapper_reset_and_step() -> None:
    env = OffroadGymEnv(max_episode_steps=3)
    obs, info = env.reset(seed=11)

    assert env.observation_space.contains(obs)
    assert info["scenario_id"] == "forest_trail_001"

    next_obs, reward, terminated, truncated, step_info = env.step(np.asarray([0.0, 0.4, 0.0]))

    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["elapsed_steps"] == 1
    env.close()


def test_gymnasium_wrapper_time_limit_truncates() -> None:
    env = OffroadGymEnv(max_episode_steps=1)
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(np.asarray([0.0, 0.0, 0.0]))

    assert not terminated
    assert truncated
    env.close()
