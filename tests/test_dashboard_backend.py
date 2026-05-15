from __future__ import annotations

import json

from fastapi.testclient import TestClient

from dashboard.backend.main import app


def test_dashboard_health_and_catalogs() -> None:
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    assert any(item["id"] == "forest_trail_001" for item in client.get("/scenarios").json())
    assert any(item["name"] == "gym_heightmap" for item in client.get("/backends").json())


def test_dashboard_run_episode_smoke() -> None:
    client = TestClient(app)
    response = client.post(
        "/run_episode",
        json={
            "backend": "gym_heightmap",
            "scenario": "forest_trail_001",
            "agent": "stop",
            "max_steps": 2,
            "record": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_id"] == "forest_trail_001"
    assert payload["metrics"]["steps"] == 2


def test_dashboard_stream_episode_sse_smoke() -> None:
    client = TestClient(app)
    response = client.get(
        "/stream_episode",
        params={
            "backend": "gym_heightmap",
            "scenario": "forest_trail_001",
            "agent": "stop",
            "max_steps": 2,
            "record": False,
            "delay_ms": 0,
        },
    )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [event["event"] for event in events] == ["start", "step", "step", "end"]
    assert events[0]["data"]["frame"]["observation"]["terrain_map"] is not None
    assert events[1]["data"]["frame"]["observation"]["local_bev"] is not None
    assert events[-1]["data"]["metrics"]["steps"] == 2


def test_dashboard_episode_steps_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/run_episode",
        json={
            "backend": "gym_heightmap",
            "scenario": "forest_trail_001",
            "agent": "stop",
            "max_steps": 2,
            "record": True,
            "record_arrays": True,
        },
    )

    episode_id = response.json()["episode_id"]
    steps_response = client.get(f"/episodes/{episode_id}/steps", params={"limit": 2})

    assert steps_response.status_code == 200
    steps = steps_response.json()
    assert len(steps) == 2
    assert steps[0]["observation"]["local_bev"] is not None


def test_dashboard_beamng_status_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/beamng/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "beamng"
    assert "beamngpy_available" in payload["details"]


def test_dashboard_world_model_and_planner_catalogs() -> None:
    client = TestClient(app)

    world_models = client.get("/world_models").json()
    planners = client.get("/planners").json()

    assert any(item["name"] == "le_wm" for item in world_models)
    assert any(item["name"] == "le_wm_cem" for item in planners)


def _parse_sse(text: str) -> list[dict[str, object]]:
    events = []
    for chunk in text.strip().split("\n\n"):
        event_name = None
        data = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event_name is not None and data is not None:
            events.append({"event": event_name, "data": data})
    return events
