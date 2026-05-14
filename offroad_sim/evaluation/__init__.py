"""Evaluation metrics and benchmark helpers."""

from offroad_sim.evaluation.metrics import MetricsTracker

__all__ = ["EpisodeRunResult", "MetricsTracker", "run_episode"]


def __getattr__(name: str):
    if name in {"EpisodeRunResult", "run_episode"}:
        from offroad_sim.evaluation.runner import EpisodeRunResult, run_episode

        return {"EpisodeRunResult": EpisodeRunResult, "run_episode": run_episode}[name]
    raise AttributeError(name)
