"""Entry point for the OffroadSimBench desktop GUI."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from desktop_app.qt_main import run
    except ImportError as exc:
        print(
            "PySide6 is required for the desktop GUI. "
            "Install it with: python -m pip install -e .[gui]",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

