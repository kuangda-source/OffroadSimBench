# Training Jobs

Manifest-backed trainers run through a persistent FIFO subprocess queue. The
desktop GUI remains responsive while stdout and stderr are streamed to the run
directory.

Each queued run contains:

- `training_job.json`: live status, progress, PID, timestamps, command, and the
  final result or failure message. It also contains ETA, latest resource values,
  and bounded `resource_history` arrays;
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

Running cancellation is asynchronous: the queue first signals the isolated
process group so a trainer can flush its last checkpoint or state, then forces
the full child process tree only after a three-second grace period. The Training
Results page can enqueue a fresh rerun with the original dataset, split,
parameters, and trainer while leaving the old output directory unchanged.
The manifest and split are also copied into the run directory and hashed;
reruns prefer these portable snapshots for reproducibility.

Generic desktop tasks use detached daemon workers with cancelable GUI
callbacks. Closing the window invalidates the callbacks immediately, so Qt
does not own or destroy a Python worker that is still finishing external work.
Workers may register a cooperative cancellation hook, and shutdown performs a
bounded join after invoking all hooks. A non-cooperative task that outlives the
deadline remains in the window's detached-task registry until its worker emits
the final settled signal; it is never reported as canceled or silently dropped.

With the GUI optional dependency `psutil`, the queue samples trainer and child
process CPU usage and resident memory every 0.5 seconds. If `nvidia-smi` reports
the trainer PID, GPU utilization and process VRAM are sampled as well. Missing
resource tools never fail a job. Resource series are copied into the finalized
run as `resource.*` metrics and use elapsed seconds as their x-axis.
