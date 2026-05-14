# UE5 Backend Protocol

`UE5Backend` is a placeholder bridge for a future Unreal runtime. It does not
depend on Unreal Engine. The first protocol uses newline-delimited JSON over TCP
so it can be tested locally with `MockUE5Server`.

## Request Format

Every message is one JSON object followed by `\n`.

```json
{"command": "reset", "scenario": {}}
{"command": "step", "action": {"steer": 0.0, "throttle": 0.5, "brake": 0.0}}
{"command": "get_observation"}
{"command": "close"}
```

## Observation Response

```json
{
  "ok": true,
  "observation": {
    "timestamp": 0.1,
    "pose": {"x": 1.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "speed": 2.0,
    "collision": false,
    "goal": [80.0, 60.0],
    "front_rgb_path": null,
    "depth_path": null,
    "lidar_path": null,
    "terrain_type": "forest"
  },
  "reward": 0.0,
  "terminated": false,
  "truncated": false,
  "info": {},
  "metrics": {}
}
```

Later this bridge can be replaced by gRPC, ZeroMQ, ROS2, or a binary shared-memory
transport without changing `OffroadAgent` or `OffroadSimBackend`.
