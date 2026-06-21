from __future__ import annotations

import json
from pathlib import Path

from desktop_app import services
from offroad_sim.utils.yaml_io import load_yaml_file


def _write_echo_trainer(root: Path) -> Path:
    script = root / "echo_trainer.py"
    script.write_text(
        """
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset_root")
parser.add_argument("--output", required=True)
parser.add_argument("--epochs", type=int, required=True)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
checkpoint = output / "echo.ckpt"
checkpoint.write_text("checkpoint", encoding="utf-8")
print(json.dumps({
    "output_dir": str(output.resolve()),
    "checkpoint_path": str(checkpoint.resolve()),
    "artifact_type": "checkpoint",
    "metrics": {"final_loss": 0.25, "epochs": args.epochs},
    "history": {"loss": [1.0, 0.5, 0.25]},
}, indent=2))
""",
        encoding="utf-8",
    )
    manifest = root / "trainer.yaml"
    manifest.write_text(
        """
trainer_id: echo_trainer
display_name: Echo Trainer
runtime: python
entrypoint: echo_trainer.py
input:
  dataset_format: manifest_dataset
parameters:
  epochs:
    type: int
    default: 3
arguments:
  - "{dataset_root}"
  - "--output"
  - "{output_dir}"
  - "--epochs"
  - "{params.epochs}"
outputs:
  artifact_type: checkpoint
""",
        encoding="utf-8",
    )
    return manifest


def test_trainer_manifest_entries_expose_defaults(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)

    entries = services.trainer_manifest_entries(tmp_path)

    assert entries[0]["id"] == "echo_trainer"
    assert entries[0]["label"] == "Echo Trainer"
    assert entries[0]["manifest_path"] == str(manifest.resolve())
    assert entries[0]["parameters"]["epochs"]["default"] == 3


def test_trainer_manifest_entries_accept_named_yaml_files(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)
    named = tmp_path / "custom_trainer.yaml"
    manifest.replace(named)

    entries = services.trainer_manifest_entries(tmp_path)

    assert entries[0]["id"] == "echo_trainer"
    assert entries[0]["manifest_path"] == str(named.resolve())


def test_import_trainer_manifest_copies_external_manifest(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "external"
    source_dir.mkdir()
    source = _write_echo_trainer(source_dir)
    destination_root = tmp_path / "trainers"

    row = services.import_trainer_manifest(source, destination_root=destination_root)

    copied_path = Path(row["manifest_path"])
    assert copied_path.parent == destination_root
    assert copied_path.name == "echo_trainer.yaml"
    copied = load_yaml_file(copied_path)
    assert copied["entrypoint"] == str((source_dir / "echo_trainer.py").resolve())
    assert copied["imported_from"] == str(source.resolve())
    entries = services.trainer_manifest_entries(destination_root)
    assert entries[0]["id"] == "echo_trainer"


def test_run_trainer_manifest_job_executes_command_and_records(tmp_path) -> None:
    manifest = _write_echo_trainer(tmp_path)
    output_dir = tmp_path / "run"

    payload = services.run_trainer_manifest_job(
        str(manifest),
        dataset_root="dataset_root",
        output_dir=str(output_dir),
        parameters={"epochs": 5},
        adapter="manifest_dataset",
        sequence_id="clip_001",
    )

    record = json.loads((output_dir / services.TRAINING_RUN_FILENAME).read_text(encoding="utf-8"))

    assert Path(payload["checkpoint_path"]).name == "echo.ckpt"
    assert payload["training_run_path"] == str((output_dir / services.TRAINING_RUN_FILENAME).resolve())
    assert (output_dir / "stdout.log").exists()
    assert record["preset_id"] == "echo_trainer"
    assert record["dataset_root"] == "dataset_root"
    assert record["adapter"] == "manifest_dataset"
    assert record["sequence_id"] == "clip_001"
    assert record["parameters"]["epochs"] == 5
    assert record["history"]["loss"] == [1.0, 0.5, 0.25]
