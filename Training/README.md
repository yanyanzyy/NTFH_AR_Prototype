# Arm + Needle Pose — Training Kit

Trains the **one and only** model the AR prototype uses: a single **YOLO11n-pose**
model that detects an **arm** and a **needle**, each with **2 keypoints**:

| Class | kpt0 | kpt1 | Used for |
|-------|------|------|----------|
| `0` arm | proximal (near elbow) | distal (wrist) | forearm axis → AR overlay + arm-surface cylinder |
| `1` needle | tip (contact point) | hub (back of needle) | insertion point + needle angle |

This is what lets the app answer **"where does the needle contact the arm?"** — the
needle-tip keypoint, measured against the arm-surface cylinder built from the arm's
two keypoints.

> **Why 2 keypoints each?** YOLO-pose locks every instance to the same keypoint
> count, so 2-each fits cleanly in a single model. One model, one inference pass.

## Setup (once)

```bash
cd Training
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## The pipeline

Run all steps from `Training/`:

| Step | Command | What it does |
|------|---------|--------------|
| 0 | drop images into `data/raw/` | Headset captures or phone photos of the arm, the needle, or both. |
| 1 | `python scripts/01_label.py` | Interactive keypoint labeler. Press `a`, click **proximal** then **distal** to add an arm; press `n`, click **tip** then **hub** to add a needle. `SPACE` saves + next, `x` saves a background frame, `q` quits. Resumable. Writes `data/labels/*.txt` + previews in `data/preview/`. |
| 2 | `python scripts/02_prepare_dataset.py` | Splits labeled images into train/val and writes `data/pose/data.yaml` (`kpt_shape: [2,3]`, classes `arm,needle`). |
| 3 | `python scripts/03_train.py` | Fine-tunes `yolo11n-pose.pt`. Outputs `runs/arm_needle_pose/weights/best.pt`. Use `--model <prev_best.pt>` to continue from your own weights. |
| 4 | `python scripts/04_export.py --weights runs/arm_needle_pose/weights/best.pt` | Exports ONNX (opset 12, 320×320) → `Assets/Models/arm-needle-pose-320.onnx`. |

### Labeling tips
- Images don't need both classes in frame. An arm-only photo gets just an arm;
  a needle-only photo gets just a needle. The model learns each from whichever
  images contain it.
- Add a few **background** frames (`x`, no objects) so the model doesn't
  hallucinate.
- Aim for **300–500 images**, varied angle/distance/lighting. Headset captures
  (via the `TrainingFrameCapture` component) match the deployment camera best.
- The needle tip is small — label it precisely; that point *is* the contact
  estimate.

## ONNX output layout (what Unity parses)

`CustomArmDetector.cs` expects features-first `[1, 12, N]`:

```
0..3   box  cx, cy, w, h        (input-pixel scale, 320×320)
4..5   class scores             (0 = arm, 1 = needle)
6..11  keypoints                (kx0, ky0, v0, kx1, ky1, v1)
       arm:    kpt0 = proximal, kpt1 = distal
       needle: kpt0 = tip,      kpt1 = hub
```

## Deploying to Unity

1. Run step 4 to produce `Assets/Models/arm-needle-pose-320.onnx`.
2. In `ArmDetectionScene`, select the object with **CustomArmDetector** and assign
   the new ONNX to its **Model Asset** slot. Set **Input Size = 320**.
3. Delete the old `Assets/Models/custom-arm-detector.onnx` once the new model is
   assigned (it's the previous box-only model; kept only so the slot isn't broken
   before you swap).

`CustomArmDetector` feeds arm keypoints into the existing overlay/world-tracking
pipeline (proximal→shoulder slot, distal→wrist slot) and exposes detected needles
via `LastNeedles`. `ArmDetectionManager.TryGetNeedle(...)` projects the tip+hub to
world space, and `InjectionSiteDetector` uses that tip to fire the injection-site
events (falling back to the OVR fingertip only if no needle is detected).

## Folder map

```
Training/
  requirements.txt
  scripts/
    01_label.py             (interactive keypoint labeler)
    02_prepare_dataset.py   (train/val split + pose data.yaml)
    03_train.py             (fine-tune yolo11n-pose)
    04_export.py            (export ONNX → Assets/Models/)
  data/
    raw/        <-- DROP IMAGES HERE
    labels/     (pose .txt written by step 1)
    preview/    (label visualisations — spot-check these)
    pose/       (generated train/val split + data.yaml)
  runs/         (training outputs; best.pt)
```
