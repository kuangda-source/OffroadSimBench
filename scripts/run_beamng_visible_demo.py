"""Launch a visible BeamNG autonomous-driving demo episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.services import VisibleBeamNGDemoRequest, run_visible_beamng_demo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--adapter", default="orfd")
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--world-model-type", default="simple_kinematic")
    parser.add_argument("--world-model", default="")
    parser.add_argument("--planner", default="")
    parser.add_argument("--scenario", default="beamng_visible_autodrive")
    parser.add_argument("--vehicle", default="configs/vehicles/ugv_medium.yaml")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    payload = run_visible_beamng_demo(
        VisibleBeamNGDemoRequest(
            dataset_root=args.dataset_root,
            adapter=args.adapter,
            sequence_id=args.sequence_id,
            world_model_type=args.world_model_type,
            world_model_path=args.world_model,
            planner=args.planner,
            scenario=args.scenario,
            vehicle=args.vehicle,
            max_steps=args.max_steps,
            seed=args.seed,
            record=not args.no_record,
        )
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
