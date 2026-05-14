"""Base classes for dataset format adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from offroad_sim.datasets.types import DatasetSequence


class DatasetAdapter(ABC):
    """Adapter that maps one physical dataset layout to normalized sequences."""

    name: ClassVar[str]
    priority: ClassVar[int] = 100

    @abstractmethod
    def can_load(self, dataset_root: str | Path) -> bool:
        """Return whether this adapter can read the dataset root."""

    @abstractmethod
    def list_sequences(self, dataset_root: str | Path) -> list[str]:
        """List sequence identifiers available in the dataset."""

    @abstractmethod
    def load_sequence(self, dataset_root: str | Path, sequence_id: str) -> DatasetSequence:
        """Load one sequence into the normalized representation."""
