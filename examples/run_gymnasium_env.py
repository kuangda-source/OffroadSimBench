from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from offroad_sim.rl import OffroadGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the Gymnasium wrapper with random actions.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = OffroadGymEnv(seed=args.seed, max_episode_steps=args.steps)
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = False
    truncated = False
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    Console().print_json(
        data={
            "state": obs["state"].tolist(),
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
    )
    env.close()


if __name__ == "__main__":
    main()
