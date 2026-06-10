"""Fine-tunes YOLO11n-pose on the mixed mannequin + COCO dataset.

Starts from the pretrained COCO weights (downloaded automatically by
ultralytics) so human detection is preserved, and trains at 320x320 — the
input size the Quest 3 runs at.

Run:  python 03_train.py
      python 03_train.py --epochs 120 --batch 8
      python 03_train.py --resume            (continue an interrupted run)

A CUDA GPU makes this take ~30-60 min; on CPU expect many hours (consider
Google Colab: upload the Training/ folder, pip install -r requirements.txt,
run the same scripts).

Results land in Training/runs/arm_pose/ — check results.png and
val_batch*_pred.jpg to confirm keypoints land on the arm before exporting.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # Training/
DATA_YAML = ROOT / "dataset" / "yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=320,
                    help="training image size — keep equal to the Unity _inputSize")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="e.g. 0 for first GPU, 'cpu' to force CPU (default: auto)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found — run 02_prepare_dataset.py first.")

    from ultralytics import YOLO

    if args.resume:
        last = ROOT / "runs" / "arm_pose" / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
        return

    model = YOLO("yolo11n-pose.pt")  # pretrained COCO weights, auto-downloaded
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=25,
        cos_lr=True,
        # Mild augmentation; mosaic off for the last epochs so the model sees
        # realistic full frames before convergence.
        close_mosaic=10,
        degrees=10.0,        # the arm can sit at any angle on the table
        translate=0.1,
        scale=0.5,
        fliplr=0.5,          # safe: flip_idx in data.yaml swaps L/R keypoints
        project=str(ROOT / "runs"),
        name="arm_pose",
        exist_ok=True,
    )

    best = ROOT / "runs" / "arm_pose" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print("Inspect runs/arm_pose/results.png and val_batch*_pred.jpg, "
          "then run 04_export_onnx.py")


if __name__ == "__main__":
    main()
