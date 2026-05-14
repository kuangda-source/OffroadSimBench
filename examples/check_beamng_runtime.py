"""Inspect and optionally smoke-test the local BeamNG backend."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from offroad_sim.backends import BeamNGBackend, BeamNGConnectionConfig
from offroad_sim.core import Action
from offroad_sim.scenarios import load_scenario_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bng-home", default=None, help="BeamNG.tech installation directory.")
    parser.add_argument("--connect", action="store_true", help="Attempt a real beamngpy connection.")
    parser.add_argument("--no-launch", action="store_true", help="Connect without launching BeamNG.")
    parser.add_argument("--steps", type=int, default=1, help="Number of stop-action steps for --connect.")
    parser.add_argument("--scenario", default=str(ROOT / "configs" / "scenarios" / "forest_trail_001.yaml"))
    args = parser.parse_args()

    status = BeamNGBackend.runtime_status(args.bng_home)
    print(json.dumps(asdict(status), indent=2))
    if not args.connect:
        return 0
    if not status.available:
        print("BeamNG runtime is not ready for a real connection.", file=sys.stderr)
        return 2

    config = BeamNGConnectionConfig(
        bng_home=args.bng_home,
        launch=not args.no_launch,
    )
    backend = BeamNGBackend(connection=config)
    try:
        observation = backend.reset(load_scenario_config(args.scenario))
        print("reset_observation:", json.dumps(_compact_observation(observation), indent=2))
        for _ in range(max(args.steps, 0)):
            result = backend.step(Action(brake=1.0))
            print("step:", json.dumps({"done": result.done, "info": result.info}, indent=2))
        print("metrics:", json.dumps(backend.get_metrics(), indent=2))
    finally:
        backend.close()
    return 0


def _compact_observation(observation: object) -> dict[str, object]:
    state = getattr(observation, "vehicle_state")
    return {
        "timestamp": getattr(observation, "timestamp"),
        "vehicle_state": {
            "x": state.x,
            "y": state.y,
            "z": state.z,
            "yaw": state.yaw,
            "speed": state.speed,
        },
        "goal": list(getattr(observation, "goal")),
        "info": getattr(observation, "info"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
