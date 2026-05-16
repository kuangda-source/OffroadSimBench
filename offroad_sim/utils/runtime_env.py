"""Runtime environment guards for optional heavy ML stacks."""

from __future__ import annotations

import os


def prepare_stable_worldmodel_runtime() -> None:
    """Set Windows runtime flags before importing torch/stable-worldmodel.

    The local conda environment can contain both OpenMP runtimes through NumPy
    and PyTorch wheels. On Windows this may abort at import time before Python
    can raise an exception. The flag is scoped to optional LE-WM execution paths.
    """

    if os.name == "nt":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

