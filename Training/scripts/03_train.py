"""
Step 3 - Train / fine-tune the YOLO11n-pose model (arm + needle, 2 keypoints each).

Fine-tunes from yolo11n-pose.pt by default. Outputs land in
Training/runs/arm_needle_pose/weights/best.pt.

    python scripts/03_train.py
    python scripts/03_train.py --epochs 120 --imgsz 320 --batch 16
    python scripts/03_train.py --model /path/to/previous_best.pt   # continue from your own weights
"""
import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent       # Training/scripts
ROOT = HERE.parents[1]                        # repo root (.../NTFH_AR_Prototype)
DATA_YAML = HERE.parent / "data" / "pose" / "data.yaml"
RUNS_DIR = HERE.parent / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="base weights (yolo11n-pose.pt) or a previous best.pt to continue from")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=320, help="match the Quest input size")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU")
    ap.add_argument("--name", default="arm_needle_pose")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} missing - run scripts/02_prepare_dataset.py first.")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
    )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Next:  python scripts/04_export.py --weights {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
