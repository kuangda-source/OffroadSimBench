"""UE5 TCP JSON backend bridge and local mock server."""

from __future__ import annotations

import json
import math
import socket
import socketserver
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from offroad_sim.backends.base import OffroadSimBackend
from offroad_sim.core import Action, Observation, StepResult, VehicleState


class UE5ProtocolError(RuntimeError):
    """Raised when the UE5 bridge returns an invalid response."""


class UE5Backend(OffroadSimBackend):
    """Backend that talks to a UE5 runtime over newline-delimited JSON."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_sec: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)
        self._socket: socket.socket | None = None
        self._file: Any = None
        self._last_observation: Observation | None = None
        self._metrics: dict[str, Any] = {"backend": "ue5", "connected": False, "episode_length": 0}

    def connect(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self._file = self._socket.makefile("rwb")
        self._metrics["connected"] = True

    def reset(self, scenario_config: Any = None) -> Observation:
        self.connect()
        response = self._request("reset", {"scenario": self._to_jsonable(scenario_config)})
        self._last_observation = self._observation_from_response(response)
        self._metrics.update(response.get("metrics", {}))
        self._metrics["episode_length"] = 0
        return self._last_observation

    def step(self, action: Action) -> StepResult:
        response = self._request(
            "step",
            {"action": {"steer": action.steer, "throttle": action.throttle, "brake": action.brake}},
        )
        self._last_observation = self._observation_from_response(response)
        response_metrics = dict(response.get("metrics", {}))
        previous_length = int(self._metrics.get("episode_length", 0))
        self._metrics.update(response_metrics)
        if "episode_length" not in response_metrics:
            self._metrics["episode_length"] = previous_length + 1
        return StepResult(
            observation=self._last_observation,
            reward=float(response.get("reward", 0.0)),
            terminated=bool(response.get("terminated", False)),
            truncated=bool(response.get("truncated", False)),
            info=dict(response.get("info", {})),
        )

    def get_observation(self) -> Observation:
        if self._last_observation is None:
            response = self._request("get_observation", {})
            self._last_observation = self._observation_from_response(response)
        return self._last_observation

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._request("close", {})
            except OSError:
                pass
        if self._file is not None:
            self._file.close()
        if self._socket is not None:
            self._socket.close()
        self._file = None
        self._socket = None
        self._metrics["connected"] = False

    def _request(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._file is None:
            raise RuntimeError("UE5Backend is not connected.")
        message = {"command": command, **payload}
        self._file.write((json.dumps(message) + "\n").encode("utf-8"))
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise UE5ProtocolError("UE5 bridge closed the connection.")
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok", False):
            raise UE5ProtocolError(str(response.get("error", "UE5 bridge command failed.")))
        return response

    def _observation_from_response(self, response: Mapping[str, Any]) -> Observation:
        data = response.get("observation")
        if not isinstance(data, Mapping):
            raise UE5ProtocolError("UE5 response did not include an observation object.")
        pose = data.get("pose", {})
        goal = data.get("goal", (0.0, 0.0))
        return Observation(
            timestamp=float(data.get("timestamp", 0.0)),
            vehicle_state=VehicleState(
                x=float(pose.get("x", 0.0)),
                y=float(pose.get("y", 0.0)),
                z=float(pose.get("z", 0.0)),
                yaw=float(pose.get("yaw", 0.0)),
                pitch=float(pose.get("pitch", 0.0)),
                roll=float(pose.get("roll", 0.0)),
                speed=float(data.get("speed", 0.0)),
            ),
            goal=(float(goal[0]), float(goal[1])),
            info={
                "backend": "ue5",
                "collision": bool(data.get("collision", False)),
                "terrain_type": data.get("terrain_type"),
                "front_rgb_path": data.get("front_rgb_path"),
                "depth_path": data.get("depth_path"),
                "lidar_path": data.get("lidar_path"),
            },
        )

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return value


class _MockUE5Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode("utf-8"))
                response = self.server.runtime.handle(request)  # type: ignore[attr-defined]
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()
            if request.get("command") == "close":
                breakk


class _MockUE5Runtime:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.timestamp = 0.0
        self.goal = (10.0, 0.0)
        self.step_count = 0

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "reset":
            scenario = request.get("scenario") or {}
            task = scenario.get("task", {}) if isinstance(scenario, Mapping) else {}
            start = task.get("start", [0.0, 0.0])
            goal = task.get("goal", [10.0, 0.0])
            self.x = float(start[0])
            self.y = float(start[1])
            self.yaw = 0.0
            self.speed = 0.0
            self.timestamp = 0.0
            self.goal = (float(goal[0]), float(goal[1]))
            self.step_count = 0
            return self._response()
        if command == "step":
            action = request.get("action") or {}
            throttle = float(action.get("throttle", 0.0))
            brake = float(action.get("brake", 0.0))
            steer = float(action.get("steer", 0.0))
            self.speed = max(0.0, min(8.0, self.speed + throttle * 0.5 - brake * 0.8))
            self.yaw += steer * 0.08
            self.x += math.cos(self.yaw) * self.speed * 0.1
            self.y += math.sin(self.yaw) * self.speed * 0.1
            self.timestamp += 0.1
            self.step_count += 1
            return self._response()
        if command == "get_observation":
            return self._response()
        if command == "close":
            return {"ok": True}
        return {"ok": False, "error": f"Unknown command: {command}"}

    def _response(self) -> dict[str, Any]:
        distance = math.hypot(self.goal[0] - self.x, self.goal[1] - self.y)
        return {
            "ok": True,
            "observation": {
                "timestamp": self.timestamp,
                "pose": {"x": self.x, "y": self.y, "z": 0.0, "yaw": self.yaw, "pitch": 0.0, "roll": 0.0},
                "speed": self.speed,
                "collision": False,
                "goal": [self.goal[0], self.goal[1]],
                "terrain_type": "mock_offroad",
            },
            "reward": -distance * 0.01,
            "terminated": distance < 1.0,
            "truncated": False,
            "info": {"backend": "ue5_mock", "distance_to_goal": distance},
            "metrics": {"backend": "ue5", "mock": True, "episode_length": self.step_count},
        }


class MockUE5Server:
    """Threaded mock UE5 JSON server for tests and protocol development."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> "MockUE5Server":
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True

        self._server = Server((self.host, self.port), _MockUE5Handler)
        self._server.runtime = _MockUE5Runtime()  # type: ignore[attr-defined]
        self.host, self.port = self._server.server_address
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def __enter__(self) -> "MockUE5Server":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
