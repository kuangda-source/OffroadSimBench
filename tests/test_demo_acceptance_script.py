from __future__ import annotations

import json

from desktop_app import services
from scripts import demo_acceptance


def test_demo_acceptance_script_prints_json_and_returns_success(monkeypatch, capsys) -> None:
    def fake_run(request: services.DemoAcceptanceRequest) -> dict[str, object]:
        assert request.runs == 2
        assert request.max_steps == 40
        return {"status": "accepted", "accepted": True, "runs": [{"goal_success": True}]}

    monkeypatch.setattr(services, "run_demo_acceptance", fake_run)

    code = demo_acceptance.main(["--runs", "2", "--max-steps", "40"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "accepted"


def test_demo_acceptance_script_returns_nonzero_when_demo_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(services, "run_demo_acceptance", lambda request: {"status": "failed", "accepted": False})

    code = demo_acceptance.main(["--runs", "1"])

    assert code == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False
