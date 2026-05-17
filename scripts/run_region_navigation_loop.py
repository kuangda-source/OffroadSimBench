"""Run a region/start/goal navigation training and evaluation loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.services import RegionNavigationClosedLoopRequest, run_region_navigation_closed_loop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="configs/tasks/beamng_region_nav_001.yaml")
    parser.add_argument("--algorithm", default="local_lewm_cost")
    parser.add_argument("--vehicle", default="configs/vehicles/ugv_medium.yaml")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--collect-steps", type=int, default=160)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planner", default="le_wm_cem")
    parser.add_argument("--beamng-gfx", choices=["vk", "dx11"], default="vk")
    parser.add_argument("--step-delay-sec", type=float, default=0.0)
    parser.add_argument("--pre-run-hold-sec", type=float, default=0.0)
    parser.add_argument("--hold-open-sec", type=float, default=0.0)
    parser.add_argument("--keep-beamng-open", action="store_true")
    args = parser.parse_args()

    payload = run_region_navigation_closed_loop(
        RegionNavigationClosedLoopRequest(
            task_path=args.task,
            algorithm=args.algorithm,
            vehicle=args.vehicle,
            output_dir=args.output_dir,
            collect_steps=args.collect_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
            planner=args.planner,
            beamng_gfx=args.beamng_gfx,
            close_beamng=not args.keep_beamng_open,
            step_delay_sec=args.step_delay_sec,
            pre_run_hold_sec=args.pre_run_hold_sec,
            post_run_hold_sec=args.hold_open_sec,
        )
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
