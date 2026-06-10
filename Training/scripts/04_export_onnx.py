"""Exports the fine-tuned model to ONNX for Unity Inference Engine and copies
it into Assets/Models/.

The export uses static shapes and opset 12, which Unity Inference Engine
(Sentis) imports cleanly. Output layout stays [1, 56, anchors] — identical to
the stock yolo11n-pose.onnx — so YoloPoseDetector.cs parses it unchanged.

Run:  python 04_export_onnx.py
      python 04_export_onnx.py --weights ../runs/arm_pose/weights/best.pt --imgsz 320
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # Training/
REPO = ROOT.parent                                   # repo root
DEFAULT_WEIGHTS = ROOT / "runs" / "arm_pose" / "weights" / "best.pt"
ASSETS_MODELS = REPO / "Assets" / "Models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--imgsz", type=int, default=320,
                    help="must match the size used in 03_train.py")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"{weights} not found — run 03_train.py first.")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_path = Path(model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    ))

    dest = ASSETS_MODELS / f"arm-pose-{args.imgsz}.onnx"
    ASSETS_MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest)

    print(f"\nExported and copied to: {dest}")
    print(
        "\nFinal steps in Unity:\n"
        "  1. Let Unity import the new .onnx (appears in Assets/Models/).\n"
        "  2. Select the YoloPoseDetector GameObject in ArmDetectionScene.\n"
        f"  3. Drag arm-pose-{args.imgsz}.onnx into the 'Model Asset' field.\n"
        f"  4. Set 'Input Size' to {args.imgsz}.\n"
        "  5. The fine-tuned model fires on the mannequin arm in NORMAL mode —\n"
        "     you may be able to raise Confidence Threshold (e.g. 0.4+) and rely\n"
        "     less on the arm-only fallback.\n"
    )


if __name__ == "__main__":
    main()
