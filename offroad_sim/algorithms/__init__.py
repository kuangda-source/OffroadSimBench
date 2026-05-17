"""Pluggable algorithm adapter interfaces and registry helpers."""

from offroad_sim.algorithms.base import (
    ActRequest,
    AlgorithmAdapter,
    AlgorithmCapabilities,
    DataPrepRequest,
    DataPrepResult,
    PredictRequest,
    ScoreActionsRequest,
    ScoreActionsResult,
    TrainRequest,
    TrainResult,
    TrajectoryPlanRequest,
    TrajectoryPlanResult,
    UnsupportedCapabilityError,
)
from offroad_sim.algorithms.manifest import AlgorithmManifest
from offroad_sim.algorithms.registry import AlgorithmRegistry, AlgorithmSpec, AlgorithmStatus, default_algorithm_registry, make_algorithm

__all__ = [
    "ActRequest",
    "AlgorithmAdapter",
    "AlgorithmCapabilities",
    "AlgorithmManifest",
    "AlgorithmRegistry",
    "AlgorithmSpec",
    "AlgorithmStatus",
    "DataPrepRequest",
    "DataPrepResult",
    "PredictRequest",
    "ScoreActionsRequest",
    "ScoreActionsResult",
    "TrainRequest",
    "TrainResult",
    "TrajectoryPlanRequest",
    "TrajectoryPlanResult",
    "UnsupportedCapabilityError",
    "default_algorithm_registry",
    "make_algorithm",
]
