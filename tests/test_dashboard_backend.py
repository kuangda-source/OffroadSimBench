from __future__ import annotations

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
