"""Evaluation metrics and benchmark helpers."""

from offroad_sim.evaluation.metrics import MetricsTracker

__all__ = ["EpisodeRunResult", "MetricsTracker", "run_episode", "stream_episode_events"]


def __getattr__(name: str):
    if name in {"EpisodeRunResult", "run_episode"}:
        from offroad_sim.evaluation.runner import EpisodeRunResult, run_episode

        return {"EpisodeRunResult": EpisodeRunResult, "run_episode": run_episode}[name]
    if name == "stream_episode_events":
        from offroad_sim.evaluation.streaming import stream_episode_events

        return stream_episode_events
    raise AttributeError(name)
