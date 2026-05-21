"""Convert a HuggingFace LE-WM mirror into a stable-worldmodel object checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", help="Directory containing HuggingFace weights.pt and config.json.")
    parser.add_argument(
        "output",
        help="Output *_object.ckpt file or run directory. Directories receive lewm_object.ckpt.",
    )
    parser.add_argument("--le-wm-home", default=os.environ.get("LE_WM_HOME", ""), help="Path to lucas-maes/le-wm source checkout.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_path = _resolve_output_path(Path(args.output))
    if args.le_wm_home:
        sys.path.insert(0, str(Path(args.le_wm_home).resolve()))

    try:
        payload = convert_hf_checkpoint(source_dir, output_path)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def convert_hf_checkpoint(source_dir: Path, output_path: Path) -> dict[str, Any]:
    if not (source_dir / "weights.pt").exists():
        raise FileNotFoundError(f"weights.pt not found in {source_dir}")
    if not (source_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {source_dir}")

    import stable_pretraining as spt
    import torch
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    cfg = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(section: str) -> Any:
        values = cfg[section]
        return MLP(
            input_dim=values["input_dim"],
            output_dim=values["output_dim"],
            hidden_dim=values["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**cfg["predictor"]),
        action_encoder=Embedder(**cfg["action_encoder"]),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    state_dict = torch.load(source_dir / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.eval(), output_path)
    return {
        "status": "completed",
        "source_dir": str(source_dir.resolve()),
        "output_checkpoint": str(output_path.resolve()),
        "stablewm_run_name": str(output_path.with_name(output_path.name[: -len("_object.ckpt")])) if output_path.name.endswith("_object.ckpt") else str(output_path),
    }


def _resolve_output_path(output: Path) -> Path:
    if output.suffix == ".ckpt":
        return output
    return output / "lewm_object.ckpt"


if __name__ == "__main__":
    raise SystemExit(main())
