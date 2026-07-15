# Training Jobs

Manifest-backed trainers run through a persistent FIFO subprocess queue. The
desktop GUI remains responsive while stdout and stderr are streamed to the run
directory.

Each queued run contains:

- `training_job.json`: live status, progress, PID, timestamps, command, and the
  final result or failure message;
- `stdout.log` and `stderr.log`: incrementally written process output;
- `training_run.json`: the stable experiment record, finalized as `completed`,
  `failed`, or `canceled`.

The queue runs one trainer at a time by default. Progress is read from a
single-line JSON event such as:

```json
{"progress": 0.4, "current_step": 4, "total_steps": 10, "message": "epoch 4"}
```

Text forms such as `Epoch 4/10` are also recognized. A final trainer JSON object
is still used for artifacts, metrics, and history. The GUI can select a task,
inspect its live logs, and cancel queued or running work. Active job files found
after an application restart are shown as `interrupted`; they are not silently
reported as successful.
