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
