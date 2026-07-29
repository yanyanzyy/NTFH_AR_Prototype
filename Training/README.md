# Arm Pose — Training Kit

Trains the **one and only** model the AR prototype uses: a single **YOLO11n-pose**
model that detects the **mannequin arm** (Limbs & Things Advanced Venipuncture
Arm, right arm, fixed position) with **2 keypoints** (`kpt_shape: [2,3]`):

| Class | Keypoints | Used for |
|-------|-----------|----------|
| `0` arm | kpt0 = proximal (near elbow), kpt1 = distal (wrist) | box + forearm axis |

**Why this model?** YOLO11n-pose is the smallest current Ultralytics
architecture (~2.9 M params), exports to a clean static ONNX that Unity's
Inference Engine runs on the Quest 3 GPU, and the pose head gives the
proximal→distal axis the AR overlay needs — one pass, box + axis. Dropping the
needle class halves the output channels ([1,11,N] vs [1,18,N]), removes a
whole class of false positives, and lets every training image supervise the
only thing that matters: the arm.

> The needle is no longer detected by vision. `InjectionSiteDetector` uses the
> OVR hand-tracking index fingertip as the needle-tip proxy.

## Setup (once)

```bash
cd Training
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## The pipeline

Run all steps from `Training/`:

**Script numbers are the run order** — `01` → `07`, no gaps, no jumping back.
Steps 3 and 4 are optional; everything else is required. The `import_*.py`
scripts are deliberately unnumbered: they are optional one-off data sources
that run before step 2 (see [Importing more training data](#importing-more-training-data)).

| Step | Command | What it does |
|------|---------|--------------|
| 0 | drop images into `data/raw/` | Headset captures of the mannequin arm (best) or phone photos. |
| 1 | `python scripts/01_label.py` | Interactive keypoint labeler. Press `a`, click **proximal** then **distal**. `SPACE` saves + next, `x` saves a background frame, `q` quits. Resumable. Writes `data/labels/*.txt` + previews in `data/preview/`. |
| 2 | `python scripts/02_prepare_dataset.py` | Splits labeled images into train/val/test and writes `data/pose/data.yaml` (`kpt_shape: [2,3]`, single class `arm`). **Deletes and rebuilds `data/pose/` — re-run steps 3-4 after it.** |
| 3 *(optional)* | `python scripts/03_augment_closeups.py` | Adds close-up crops (`zoom_*`) to **train only**, covering the framing range the headset sees. Read its docstring first: the crops are much softer than real captures. |
| 4 *(optional)* | `python scripts/04_augment_handview.py` | Adds foreshortened hand-POV views (`hview_*`) to **train only**. Same softness caveat; `--report-only` measures without writing. |
| 5 | `python scripts/05_train.py` | Fine-tunes `yolo11n-pose.pt`. Outputs `runs/<name>/weights/best.pt`. Use `--model <prev_best.pt>` to continue from your own weights. |
| 6 | `python scripts/06_evaluate.py --weights runs/<name>/weights/best.pt` | Scores detection on the full test split and pose on the keypoint-labeled subset, writing `runs/<name>/evaluation/` (`summary.md`, `metrics.csv`, keypoint-error + qualitative plots — mAP, pixel error, PCK, mean IoU/OKS). Ultralytics' own val understates pose badly here (most images carry placeholder keypoints), so use this before trusting a run. |
| 7 | `python scripts/07_export.py --weights runs/<name>/weights/best.pt` | Exports ONNX (opset 12, 320×320) → `Assets/Models/arm-pose-320.onnx`. |

### Capture + labeling tips
- **Capture from the headset** (`TrainingFrameCapture` component) — it matches
  the deployment camera's optics, noise and white balance far better than a
  phone.
- The arm is now **moved and turned during use**, so cover that in the data:
  vary head position/angle (near/far, left/right/above), the **arm's own
  orientation** (rotated, held mid-air, different table spots), lighting, and
  clutter (hands occluding the arm, wipes/tourniquet, the marker band ON the
  arm as it will be at runtime).
- Aim for **300–500 labeled frames**. More viewpoints beat more frames of the
  same viewpoint.
- Add a good handful of **background** frames (`x`, no arm in frame) —
  especially of the table and people's real arms — so the model doesn't
  hallucinate the mannequin everywhere. Real human arms in frame *without* a
  label are the single best negative you can give it.
- Label **proximal/distal precisely** — those two points are the overlay axis.
- Horizontal flip is disabled in training (`fliplr=0.0`): the prop is a right
  arm and mirrored frames would train a left arm that never appears.

## ONNX output layout (what Unity parses)

With 2 keypoints + 1 class the tensor is features-first `[1, 11, N]` (N = 2100
anchors at 320×320):

```
0..3    box  cx, cy, w, h        (input-pixel scale, 320×320)
4       arm score
5..10   keypoints                (kx0,ky0,v0, kx1,ky1,v1)
        kpt0 = proximal (near elbow), kpt1 = distal (wrist)
```

`CustomArmDetector.cs` parses this layout natively and still accepts the legacy
2-class `[1, 12/18, N]` exports (using its serialized arm-class id) so the
scene keeps working until the new ONNX is assigned.

## Importing more training data

Four importers pull external datasets into `data/raw/` + `data/labels/`
(each uses a unique filename prefix, so they can be combined freely):

| Importer | Source | What it contributes |
|---|---|---|
| `import_phlebotomy_zenodo.py --download` | [Zenodo 16924786](https://zenodo.org/records/16924786) — 11,884 images of a blood draw on a **medical training arm** (CC BY 4.0, ~321 MB, auto-downloaded) | The closest public match to our target: training-arm boxes amid gloves/syringe clutter. Box-only (placeholder kpts). Prefix `phleb__`. |
| `import_arm_boxes.py` | `data/arm-pose-converted/` (historical human-arm set) | Generic "arm" box pretraining. Prefix `arm__`. |
| `import_coco_forearms.py --images … --annotations …` | [COCO](https://cocodataset.org) person keypoints (val2017 is plenty) | The only bulk source of **real proximal/distal keypoint supervision** (elbow→wrist). Human forearms — stage-1 only! Prefix `coco__`. |
| `import_negatives.py --src …` | Any armless images, e.g. [EgoHands](http://vision.soic.indiana.edu/projects/egohands/) (headset-view human hands) or your own no-arm captures | Background negatives so the model does **not** fire on real human arms/hands. Prefix `neg__`. |

### Two-stage recipe (best accuracy)

Human-arm positives and human-arm negatives contradict each other, so don't mix
them in one run — pretrain general, then specialize:

```bash
# Stage 1 - learn "arm-ness" + the elbow->wrist axis (generic data)
python scripts/import_phlebotomy_zenodo.py --download
python scripts/import_arm_boxes.py                      # if you have the folder
python scripts/import_coco_forearms.py --images ... --annotations ... --max 800
python scripts/02_prepare_dataset.py && python scripts/05_train.py --name arm_pose_stage1

# Stage 2 - specialize on THE mannequin arm (start fresh raw/labels dirs:
# keep phleb__ + your labeled mannequin captures, drop coco__/arm__, add negatives)
python scripts/import_negatives.py --src <egohands _LABELLEDSAMPLES> --max 400
python scripts/01_label.py                              # your headset captures
python scripts/02_prepare_dataset.py
python scripts/03_augment_closeups.py                   # optional
python scripts/05_train.py --model runs/arm_pose_stage1/weights/best.pt --name arm_pose_v8
python scripts/06_evaluate.py --weights runs/arm_pose_v8/weights/best.pt
python scripts/07_export.py --weights runs/arm_pose_v8/weights/best.pt
```

Your own keypoint-labeled headset captures remain the highest-value data —
the external sets are pretraining bulk and negatives around them.

## Current status

The latest evaluated run is **`arm_pose_v7`** (test split: detection mAP50
0.966, pose mAP50 0.915 on the 263 keypoint-labeled images, PCK@10% of arm
length 67.5%, median keypoint error ~19 px). Full numbers in
`runs/arm_pose_v7/evaluation/summary.md`; earlier runs (`arm_pose_stage1`,
`arm_pose_v2`–`v6`) keep their own `evaluation/` folders for comparison.
Keypoint accuracy is currently bounded by data variety more than by
architecture — new capture sessions move the needle more than retraining.

## Deploying to Unity

1. Run step 7 to produce `Assets/Models/arm-pose-320.onnx`.
2. In `ArmDetectionScene`, select the object with **CustomArmDetector** and assign
   the new ONNX to its **Model Asset** slot. Set **Input Size = 320**.
3. `Assets/Models/` accumulates per-run copies (`arm_pose_v4`–`v7.onnx`) and
   per-person exports (`lucas.onnx`, `zhengliang.onnx`, `best_theothergroup.onnx`)
   alongside the legacy combined models (`arm-needle-pose-320.onnx`,
   `best*.onnx`, `custom-arm-detector.onnx`). Only the ONNX assigned to the
   **Model Asset** slot is used at runtime; anything unassigned can be deleted.
   (`SyringePose*.onnx` belong to the separate syringe-pose scene — leave them.)

### Quest 3 performance settings on the component

- **Backend**: `GPUCompute` (default).
- **Layers Per Frame** (~10–20): the model's layers are spread across frames
  via `ScheduleIterable`, so no single frame takes the whole GPU hit — this is
  what keeps the app at native frame rate. Lower = smoother frame rate, lower
  detection rate; `0` schedules everything in one frame (old behavior).
- **Quantize To FP16** (default on): halves weight memory and speeds up mobile
  GPU inference with no practical accuracy loss for this model.
- Detection results are read back **asynchronously** — the main thread never
  blocks on the GPU; the manager's smoothing/lock carries the overlay between
  completed inferences.

## Folder map

```
Training/
  requirements.txt
  scripts/
    01_label.py             (interactive keypoint labeler — arm proximal/distal)
    02_prepare_dataset.py   (train/val/test split + pose data.yaml)
    03_augment_closeups.py  (optional: close-up crops into train)
    04_augment_handview.py  (optional: foreshortened hand-POV views into train)
    05_train.py             (fine-tune yolo11n-pose, arm-only)
    06_evaluate.py          (detection + pose metrics on the test split)
    07_export.py            (export ONNX → Assets/Models/arm-pose-320.onnx)

    pose_label_io.py        (shared label read/write + crop geometry for 03/04)
    pose_loss_patch.py      (keypoint-loss masking applied by 05_train.py)

    (unnumbered = optional one-off data sources; all run BEFORE step 2)
    import_arm_boxes.py     (data/arm-pose-converted/ → 2-kpt arm labels, class 0)
    import_phlebotomy_zenodo.py  (Zenodo training-arm dataset → arm boxes, auto-download)
    import_coco_forearms.py      (COCO person kpts → forearms with real elbow/wrist kpts)
    import_negatives.py          (any armless images → background negatives)
    import_arm_segmentation_to_pose.py  (one-off Roboflow seg → pose converter)
    make_marker_band.py          (printable ArUco band for the 6-DoF arm tracker)
  data/
    raw/                 <-- DROP IMAGES HERE (also where the importers write)
    labels/              (pose .txt — from 01_label.py / the importers)
    preview/             (label visualisations — spot-check these)
    pose/                (generated train/val split + data.yaml)
    pose_kpt_test/       (generated by 06_evaluate.py: keypoint-only eval subset)
    arm-pose-converted/  (historical arm source: images/ + labels/, 4-kpt padded)
    phlebotomy-src/      (extracted Zenodo dataset; created by --download)
    downloads/           (fetched archives, e.g. the Roboflow blood-extraction zip)
    needle/              (legacy needle dataset — no longer used for training)
  markers/               (created by make_marker_band.py: printable marker band
                          + shoulder disc; not present until you run it)
  runs/                  (training outputs; e.g. runs/arm_pose_v7/weights/best.pt
                          + runs/<name>/evaluation/ from 06_evaluate.py)
```
