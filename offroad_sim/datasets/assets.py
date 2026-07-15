"""Small, simulator-neutral helpers for reading dataset assets."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def load_asset_array(asset_ref: str | Path) -> np.ndarray:
    """Load NPY or image data from a path or an ORFD-style zip URI."""

    value = str(asset_ref)
    if value.startswith("zip://"):
        raw = value.removeprefix("zip://")
        zip_path, member = raw.split("!", 1)
        with ZipFile(zip_path) as archive:
            payload = archive.read(member)
        if Path(member).suffix.lower() == ".npy":
            return np.load(BytesIO(payload), allow_pickle=False)
        with Image.open(BytesIO(payload)) as image:
            return np.asarray(image)
    path = Path(value)
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image)
