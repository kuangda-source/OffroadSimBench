from __future__ import annotations

import subprocess
import sys


def test_cli_list_backends() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "offroad_sim.cli", "list", "--kind", "backends"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "gym_heightmap" in completed.stdout


def test_cli_lists_and_inspects_algorithms() -> None:
    listed = subprocess.run(
        [sys.executable, "-m", "offroad_sim.cli", "algorithms", "list"],
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = subprocess.run(
        [sys.executable, "-m", "offroad_sim.cli", "algorithms", "inspect", "local_lewm_cost", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "local_lewm_cost" in listed.stdout
    assert '"algorithm_id": "local_lewm_cost"' in inspected.stdout


def test_cli_run_json_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "offroad_sim.cli",
            "run",
            "--agent",
            "stop",
            "--max-steps",
            "2",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "episode_id" in completed.stdout
    assert "metrics" in completed.stdout
