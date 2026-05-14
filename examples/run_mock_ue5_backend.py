from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from offroad_sim.backends import MockUE5Server, UE5Backend
from offroad_sim.core import Action
from offroad_sim.scenarios import load_scenario_config


def main() -> None:
    scenario = load_scenario_config(ROOT / "configs" / "scenarios" / "forest_trail_001.yaml")
    console = Console()
    with MockUE5Server() as server:
        backend = UE5Backend(host=server.host, port=server.port)
        obs = backend.reset(scenario)
        table = Table(title="Mock UE5 Backend")
        table.add_column("Step", justify="right")
        table.add_column("X", justify="right")
        table.add_column("Y", justify="right")
        table.add_column("Speed", justify="right")
        for step in range(5):
            table.add_row(str(step), f"{obs.vehicle_state.x:.2f}", f"{obs.vehicle_state.y:.2f}", f"{obs.vehicle_state.speed:.2f}")
            obs = backend.step(Action(throttle=0.7, steer=0.1)).observation
        console.print(table)
        console.print(backend.get_metrics())
        backend.close()


if __name__ == "__main__":
    main()
