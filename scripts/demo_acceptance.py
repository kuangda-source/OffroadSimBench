"""Run the standard BeamNG demo acceptance loop one to three times."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from desktop_app import services


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-config", default=services.DEFAULT_DEMO_CONFIG_ID)
    parser.add_argument("--runs", type=int, default=1, help="Number of acceptance runs, clamped to 1..3.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planner-horizon", type=int, default=6)
    parser.add_argument("--planner-samples", type=int, default=32)
    parser.add_argument("--planner-iterations", type=int, default=3)
    parser.add_argument("--beamng-gfx", default="vk")
    parser.add_argument("--step-delay-sec", type=float, default=0.0)
    parser.add_argument("--pre-run-hold-sec", type=float, default=0.0)
    parser.add_argument("--post-run-hold-sec", type=float, default=0.0)
    parser.add_argument("--keep-beamng-open", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = services.run_demo_acceptance(
        services.DemoAcceptanceRequest(
            demo_config_id=args.demo_config,
            runs=args.runs,
            max_steps=args.max_steps,
            seed=args.seed,
            planner_horizon=args.planner_horizon,
            planner_samples=args.planner_samples,
            planner_iterations=args.planner_iterations,
            beamng_gfx=args.beamng_gfx,
            close_beamng=not args.keep_beamng_open,
            step_delay_sec=args.step_delay_sec,
            pre_run_hold_sec=args.pre_run_hold_sec,
            post_run_hold_sec=args.post_run_hold_sec,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if bool(report.get("accepted")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
