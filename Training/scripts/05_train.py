"""
Step 5 - Train / fine-tune the ARM-ONLY YOLO11n-pose model (1 class, 2 keypoints).

Fine-tunes from yolo11n-pose.pt by default. Outputs land in
Training/runs/arm_pose/weights/best.pt.

    python scripts/05_train.py
    python scripts/05_train.py --epochs 150 --imgsz 320 --batch 16
    python scripts/05_train.py --model /path/to/previous_best.pt   # continue from your own weights
    python scripts/05_train.py --stop-on loss --patience 20        # stop when val loss stalls

The deployment target is a single mannequin RIGHT arm in a fixed position, so
horizontal flip augmentation is disabled (a mirrored frame shows a left arm the
headset will never see). Rotation/translation/scale/colour augs stay on to
cover headset viewpoint and lighting variation.

EARLY STOPPING
--------------
Training stops after --patience epochs without improvement. What counts as
"improvement" is chosen with --stop-on:

  fitness (default) - Ultralytics' own criterion: a weighted blend of box and
                      pose mAP. Stops when the model stops getting *better at
                      the task*.
  loss              - total validation loss (sum of the val/*_loss columns).
                      Stops when the model stops *fitting the data*.

They are not interchangeable. Detection val loss often starts creeping up
while mAP is still improving, because the loss punishes low-confidence
predictions that mAP happily ranks correctly - so --stop-on loss tends to stop
earlier, sometimes before the model peaks. Prefer fitness unless you
specifically want the loss criterion.

Either way `best.pt` is still the highest-*fitness* epoch, not the lowest-loss
one: early stopping decides when to stop, Ultralytics decides which checkpoint
to keep.
"""
import argparse
import math
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

import pose_loss_patch

HERE = Path(__file__).resolve().parent       # Training/scripts
ROOT = HERE.parents[1]                        # repo root (.../NTFH_AR_Prototype)
DATA_YAML = HERE.parent / "data" / "pose" / "data.yaml"
RUNS_DIR = HERE.parent / "runs"


def loss_early_stopper(patience: int, min_delta: float):
    """Callback that halts training when total validation loss stops falling.

    Ultralytics' built-in stopper watches fitness, so this fills the gap. It
    runs on on_fit_epoch_end, where trainer.metrics carries the val/*_loss
    values that also land in results.csv.
    """
    state = {"best": math.inf, "epoch": 0}

    def on_fit_epoch_end(trainer):
        losses = [v for k, v in (trainer.metrics or {}).items()
                  if k.startswith("val/") and k.endswith("_loss")]
        if not losses:
            return                      # nothing to score against yet
        total = float(sum(losses))
        if total < state["best"] - min_delta:
            state["best"], state["epoch"] = total, trainer.epoch
        elif trainer.epoch - state["epoch"] >= patience:
            LOGGER.info(
                f"\nEarly stopping: val loss has not improved for {patience} epochs "
                f"(best {state['best']:.5f} at epoch {state['epoch'] + 1}). "
                f"Best weights are still selected by fitness, not loss."
            )
            trainer.stop = True

    return on_fit_epoch_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="base weights (yolo11n-pose.pt) or a previous best.pt to continue from")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=320, help="match the Quest input size")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu for CPU, mps for Apple Silicon")
    ap.add_argument("--name", default="arm_pose")
    ap.add_argument("--patience", type=int, default=20,
                    help="epochs without improvement before training stops")
    ap.add_argument("--stop-on", choices=("fitness", "loss"), default="fitness",
                    help="what must improve: fitness (box+pose mAP) or total val loss")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="with --stop-on loss, the smallest drop that counts as improvement")
    ap.add_argument("--raw-pose-loss", action="store_true",
                    help="train with stock Ultralytics keypoint losses, letting the ~86%% of "
                         "images with placeholder visibility-0 keypoints suppress the "
                         "visibility head (reproduces arm_pose_v5; see pose_loss_patch.py)")
    args = ap.parse_args()

    if not DATA_YAML.exists():
        raise SystemExit(f"{DATA_YAML} missing - run scripts/02_prepare_dataset.py first.")

    if args.raw_pose_loss:
        LOGGER.warning("Stock keypoint losses: unlabeled instances WILL suppress the "
                       "visibility head (--raw-pose-loss).")
    else:
        pose_loss_patch.apply()

    model = YOLO(args.model)

    if args.stop_on == "loss":
        # Disable the built-in fitness stopper so the two criteria can't race,
        # then let the loss callback own the decision.
        patience = args.epochs + 1
        model.add_callback("on_fit_epoch_end", loss_early_stopper(args.patience, args.min_delta))
        LOGGER.info(f"Early stopping on total val loss (patience={args.patience}, "
                    f"min_delta={args.min_delta}).")
    else:
        patience = args.patience
        LOGGER.info(f"Early stopping on fitness (patience={args.patience}).")

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=patience,
        cos_lr=True,
        close_mosaic=10,
        degrees=15.0,        # headset roll/tilt while looking at the fixed arm
        translate=0.1,
        scale=0.5,           # varying viewing distance
        fliplr=0.0,          # right arm only - never mirror it into a left arm
        flipud=0.0,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
    )

    best = RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Next:  python scripts/06_evaluate.py --weights {best}")
    print(f"       python scripts/07_export.py --weights {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
