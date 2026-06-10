# Mannequin-Arm Fine-Tuning Kit

Fine-tunes **YOLO11n-pose** to reliably detect the Limbs & Things venipuncture
arm, while mixing in COCO human images so the model **keeps detecting real
human arms** for the future patient phase. The exported ONNX is drop-in
compatible with the existing Unity `YoloPoseDetector` — same COCO 17-keypoint
schema, same output layout.

## One-time setup

```powershell
cd Training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9+ required. A CUDA GPU makes training take ~30–60 min; on CPU it's
many hours (alternative: upload this `Training/` folder to Google Colab and run
the same scripts there).

## Step 0 — Capture photos (on the headset)

1. In Unity, add the **TrainingFrameCapture** component (in
   `Assets/Scripts/ArmDetection/`) to the `ArmDetectionPrototype` GameObject
   and drag the `PassthroughCameraSource` into its camera field.
2. Build to the Quest 3. It saves a frame every second (max 600).
3. Slowly walk around the mannequin arm for ~5–8 minutes. Vary:
   - **angle** (all sides, above, oblique), **distance** (0.3–1.5 m),
   - **lighting** (lights on/off, blinds open/closed),
   - **background** (different tables/covers if possible),
   - include some frames with **your hands near the arm**, and ~10–15% frames
     **without the arm at all** (label these as background later — they teach
     the model not to hallucinate).
4. Pull the photos to the PC:

```powershell
adb pull /sdcard/Android/data/<your.package.name>/files/ArmCaptures Training/dataset/raw
```

(Find the package name under Edit ▸ Project Settings ▸ Player ▸ Identification.)

Aim for **300–500 images**. Phone photos work in a pinch, but headset captures
match the deployment camera and train a better model.

## Step 1 — Label (≈20–30 min for 400 images)

```powershell
python scripts\01_label_arm.py
```

Click **shoulder → elbow → wrist** (3 clicks), press **SPACE**. Press **x** on
arm-free frames to save them as background negatives. **f** flips arm side
(default RIGHT, matching the Limbs & Things right arm). Progress is saved per
image; quit and resume anytime. Spot-check `dataset/preview/` afterwards.

## Step 2 — Build the dataset

```powershell
python scripts\02_prepare_dataset.py
```

First run downloads COCO val2017 (~1 GB total) into `dataset/coco_cache/` and
mixes ~1500 real-human images with your labeled mannequin images (mannequin
oversampled 3× to balance). Use `--no-coco` only if you never need human
detection from this model.

## Step 3 — Train

```powershell
python scripts\03_train.py
```

Defaults: 80 epochs, 320×320 (matches the Quest input size), early stopping.
When done, open `runs/arm_pose/results.png` and the `val_batch*_pred.jpg`
images — keypoints should sit on the mannequin arm's shoulder/elbow/wrist.

## Step 4 — Export to Unity

```powershell
python scripts\04_export_onnx.py
```

This exports ONNX (opset 12, static 320 shapes) and copies it to
`Assets/Models/arm-pose-320.onnx`. Then in Unity:

1. Select the **YoloPoseDetector** GameObject in `ArmDetectionScene`.
2. Assign `arm-pose-320.onnx` to **Model Asset**.
3. Set **Input Size** to `320`.
4. On-device, the mannequin arm should now fire in **NORMAL mode** with high
   confidence — try raising **Confidence Threshold** to ~0.4 and see if you
   can stop relying on the arm-only fallback entirely.

## Folder map

```
Training/
  requirements.txt
  dataset/
    raw/          ← DROP PHOTOS HERE
    raw_labels/   (written by step 1)
    preview/      (label visualisations — spot-check these)
    coco_cache/   (auto-downloaded COCO subset, ~2 GB; safe to delete after)
    yolo/         (generated train/val dataset)
  scripts/
    01_label_arm.py
    02_prepare_dataset.py
    03_train.py
    04_export_onnx.py
  runs/           (training outputs, best.pt lives here)
```
