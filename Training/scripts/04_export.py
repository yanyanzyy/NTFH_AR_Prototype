"""
Step 4 - Export trained pose weights to ONNX for Unity (Sentis).

    python scripts/04_export.py --weights runs/arm_needle_pose/weights/best.pt

Exports a static-shape ONNX (opset 12, 320x320) and copies it into
Assets/Models/ for the CustomArmDetector component to load.

Output channel layout (per anchor), features-first [1, 12, N]:
    0..3   box cx, cy, w, h        (input-pixel scale, 320x320)
    4..5   class scores            (0 = arm, 1 = needle)
    6..11  keypoints               (kx0, ky0, v0, kx1, ky1, v1)
           arm:    kpt0 = proximal, kpt1 = distal
           needle: kpt0 = tip,     kpt1 = hub
"""
import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent   # Training/scripts
ROOT = HERE.parents[1]                    # repo root
ASSETS_MODELS = ROOT / "Assets" / "Models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--name", default="arm-needle-pose-320.onnx")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"{weights} not found.")

    model = YOLO(str(weights))
    out = model.export(format="onnx", opset=args.opset, imgsz=args.imgsz, dynamic=False, simplify=True)

    ASSETS_MODELS.mkdir(parents=True, exist_ok=True)
    dest = ASSETS_MODELS / args.name
    shutil.copy2(out, dest)
    print(f"Exported {out}\nCopied to {dest}")
    print("In Unity: assign this ONNX to CustomArmDetector's Model Asset slot and set Input Size = "
          f"{args.imgsz}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
