"""Run a short BeamNG episode with a switchable world-model agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offroad_sim.evaluation import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/scenarios/forest_trail_001.yaml")
    parser.add_argument("--world-model-type", default="tiny_learned")
    parser.add_argument("--world-model", default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--record-arrays", action="store_true")
    args = parser.parse_args()

    result = run_episode(
        backend_name="beamng",
        scenario=args.scenario,
        agent_name="world_model",
        max_steps=args.max_steps,
        record=args.record,
        record_arrays=args.record_arrays,
        agent_options={
            "world_model_name": args.world_model_type,
            "world_model_path": args.world_model,
        },
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
