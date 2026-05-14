from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table

from offroad_sim.backends import default_backend_registry


def main() -> None:
    registry = default_backend_registry()
    table = Table(title="OffroadSimBench Backends")
    table.add_column("Name")
    table.add_column("Available")
    table.add_column("Message")
    table.add_column("Details")
    for name, status in registry.status().items():
        table.add_row(name, str(status.available), status.message, str(status.details or {}))
    Console().print(table)


if __name__ == "__main__":
    main()
