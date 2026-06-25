# Arm + Needle Pose — Training Kit

Trains the **one and only** model the AR prototype uses: a single **YOLO11n-pose**
model that detects an **arm** and a **needle**, each with **4 keypoints**
(`kpt_shape: [4,3]`):

| Class | Keypoints | Used for |
|-------|-----------|----------|
| `0` arm | kpt0–3 — **placeholder (visibility 0), not currently supervised** | box only |
| `1` needle | kpt0 = tip (contact point) … kpt3 = hub/plunger; kpt1–2 mid-barrel | insertion point + needle axis |

The needle gets real 4-point supervision (tip = the contact estimate). The arm
class is currently **box-only**: the imported arm data has no real keypoints, so
its 4 keypoint slots are placeholders (`v=0`) that contribute no loss.

> **Why 4 keypoints each?** YOLO-pose locks every instance to the same keypoint
> count. The needle data is labeled with 4 points, so the whole model uses 4 — the
> arm's slots are zero-filled to match. One model, one inference pass.

> ⚠️ **Arm overlay axis is not learned from current data.** The AR forearm overlay
> needs real arm keypoints (proximal/distal). Until arm images are keypoint-labeled,
> the model will emit meaningless arm keypoints — usable for the arm *box*, not the
> axis. See the arm section below.

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
| 2 | `python scripts/02_prepare_dataset.py` | Splits labeled images into train/val and writes `data/pose/data.yaml` (`kpt_shape: [4,3]`, classes `arm,needle`). |
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

> ⚠️ **`01_label.py` is still a 2-keypoint labeler** (proximal/distal, tip/hub) and
> is **out of sync** with the current 4-kpt model — its `data/labels/*.txt` would be
> 11-token, which `02_prepare_dataset.py` mixes badly with the 17-token converter
> output. Don't run it against this model until it's updated to emit 4 keypoints.
> For now the dataset comes entirely from `needle_to_combined.py` +
> `arm_pose_to_combined.py`.

## ONNX output layout (what Unity parses)

With 4 keypoints + 2 classes the tensor is features-first `[1, 18, N]`:

```
0..3    box  cx, cy, w, h        (input-pixel scale, 320×320)
4..5    class scores             (0 = arm, 1 = needle)
6..17   keypoints                (kx0,ky0,v0, kx1,ky1,v1, kx2,ky2,v2, kx3,ky3,v3)
        needle: kpt0 = tip, kpt3 = hub/plunger (kpt1–2 mid-barrel)
        arm:    placeholder (v≈0)
```

> ⚠️ `CustomArmDetector.cs` currently parses the **old `[1, 12, N]`** (2-kpt)
> layout. Moving to this 4-kpt model requires updating that parser to `[1, 18, N]`
> (stride 18, 12 keypoint floats) before the ONNX will work in Unity.

## Importing the needle data (4-kpt, class = needle)

The data in `data/needle/` is labeled as 1 class (`Syringe`) with **4 keypoints**
(points down the syringe axis, k0 = tip … k3 = plunger/hub). Since the combined
model is also 4-kpt, the converter just remaps the class and keeps everything:

```
python scripts/needle_to_combined.py            # add --previews 6 to spot-check
```

| Source (Syringe, 4 kpt) | → | Combined model (needle, 4 kpt) |
|-------------------------|---|--------------------------------|
| class `0` Syringe       | → | class `1` needle |
| box + all 4 keypoints   | → | kept as-is |

It copies each image into `data/raw/` and writes the label into `data/labels/`
(prefixed `v1__`/`v2__` so stems never collide). Current needle data: **1672
images** (578 v1 + 1094 v2); the tip keypoint is labeled in every frame.

## Importing the arm data (4-kpt, class = arm)

Arm data comes from `data/arm-pose-converted/` — a detection dataset already
padded to 4-keypoint pose format. **Its keypoints are placeholders** (all 4 sit on
the box centre with `v=0`); it provides arm *boxes* only. Its class index is `1`,
so the converter remaps it to `0`:

```
python scripts/arm_pose_to_combined.py
```

| Source (arm-pose-converted) | → | Combined model (arm) |
|-----------------------------|---|----------------------|
| class `1`                   | → | class `0` arm |
| box + 4 placeholder kpts (v=0) | → | kept as-is |

It writes class-0 `arm` labels into `data/labels/` and copies images to
`data/raw/`, prefixed `arm__`. Current arm data: **1155 images / 1873 boxes**.

> ⚠️ **No real arm keypoints.** Because every arm keypoint is `v=0`, the model
> gets zero arm-keypoint supervision and the forearm-overlay axis will not work
> from this data alone. For a usable axis, keypoint-label some real mannequin arm
> captures (see the labeling note below) and add them to `data/raw/`+`data/labels/`.

After both imports, run the normal `02 → 03 → 04`.

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
    01_label.py             (interactive keypoint labeler — arm+needle)
    02_prepare_dataset.py   (train/val split + pose data.yaml)
    03_train.py             (fine-tune yolo11n-pose)
    04_export.py            (export ONNX → Assets/Models/)
    needle_to_combined.py     (data/needle/ → 4-kpt needle, class 1)
    arm_pose_to_combined.py   (data/arm-pose-converted/ → 4-kpt arm, class 0)
  data/
    raw/                 <-- DROP IMAGES HERE (also where the converters write)
    labels/              (pose .txt — from the converters)
    preview/             (label visualisations — spot-check these)
    pose/                (generated train/val split + data.yaml)
    needle/              (pre-labeled 4-kpt Syringe source: 'All images/' + 'Pose_2/')
    arm-pose-converted/  (4-kpt arm source: images/ + labels/, box-only placeholders)
  runs/                  (training outputs; best.pt)
```
