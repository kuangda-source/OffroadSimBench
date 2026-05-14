from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from offroad_sim.evaluation import run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the world-model agent on the local heightmap backend.")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_episode(
        agent_name="world_model",
        seed=args.seed,
        max_steps=args.max_steps,
        record=args.record is not None,
        episode_path=args.record,
    )
    Console().print_json(data=result.to_dict())


if __name__ == "__main__":
    main()
