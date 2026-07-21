DEPLOYED MODEL: arm_pose_v5.onnx  (arm_pose_v5 training run, 2026-07-21)
Assigned to CustomArmDetector in ArmDetectionScene. Single class, 320x320 input.
Same tensor layout as the previous arm-pose-320.onnx (v2), so it is a drop-in swap.

  Input : [1, 3, 320, 320]   (CustomArmDetector reads the size from the ONNX
                              itself, so the Inspector's Input Size can't break it)
  Output: [1, 11, 2100] features-first
    0..3   box cx, cy, w, h   (input-pixel scale, letterboxed 320x320)
    4      arm score          (single class: 0 = arm)
    5..10  keypoints          (kx0, ky0, v0, kx1, ky1, v1)
           kpt0 = proximal (near elbow), kpt1 = distal (wrist)

  QUALITY (arm_pose_v5, test split - Training/runs/arm_pose_v5/evaluation/summary.md):
    box  mAP50 0.973, mAP50-95 0.905
    pose mAP50 0.926, mAP50-95 0.817   (scored on the 263 keypoint-labeled
                                        images; median error 15 px)
  The keypoint head is now TRAINED - unlike v2, whose pose mAP really was ~0.
  Keypoint POSITIONS are trustworthy, so the manager drives the overlay from
  them (_useKeypointAxis) instead of the old fixed-world-axis fallback.

  IMPORTANT - VISIBILITY SCORES ARE NOT CONFIDENCE:
  Only ~13% of the training images (263 of 2044) carry real keypoints; the rest
  are box-only images whose keypoints are placeholders at visibility 0. The
  visibility head is therefore trained toward zero and reports a small fraction
  even when the keypoint position is spot on. Two consequences:

    * Training-time metrics/mAP50(P) in results.csv reads ~0.002 and looks like
      a failed keypoint head. It is not - it averages in the 1781 placeholder
      images. Trust evaluation/summary.md, which scores labeled images only.
    * NEVER gate detection quality on keypoint visibility. Use the BOX score.
      Thresholding visibility at a "sensible" 0.05-0.4 rejects every detection
      and presents as: model detects fine (white debug boxes appear) but nothing
      locks and no overlay renders. This bug cost a debugging session on
      2026-07-21; the gates now live on the box score
      (ArmDetectionManager._acquireMinConfidence) and _keypointConfidence is
      pinned to 0.

  Preprocessing: frames must be LETTERBOXED (aspect-fit + gray pad), matching
  Ultralytics train/val. CustomArmDetector does this; don't feed stretched frames.

Produced by the Training/ pipeline (see Training/README.md):

  cd Training
  python scripts/01_label.py
  python scripts/02_prepare_dataset.py
  python scripts/03_train.py --imgsz 320
  python scripts/04_export.py --weights runs/<run>/weights/best.pt

DEPLOYED NEEDLE MODEL: best_theothergroup.onnx  (Group 2's "SyringePose" model,
from the Jin-Rui branch). Assigned to NeedleDetector
(Tools > AR Arm Detection > Add Needle Detector). Runs ALONGSIDE the arm model.

  Input : [1, 3, 640, 640]
  Output: [1, 17, 8400] features-first
    0..3   box cx, cy, w, h     (input-pixel scale, letterboxed 640x640)
    4      syringe score        (single class)
    5..16  4 keypoints (kx,ky,v):
           0 = NeedleTip (contact point), 1 = BarrelTop, 2 = BarrelBottom, 3 = Plunger
           NeedleDetector uses kpt0 as tip and kpt3 as hub.

  NeedleDetector config for this model: Num Classes = 1, Needle Class Id = 0,
  confidence threshold ~0.15 (Group 2's tested value).

  KNOWN LIMITATION (Group 2 finding): the Plunger keypoint scores only ~0.03-0.05
  even on correct detections (tip/barrel run ~0.98), so the hub position is noisy.
  NeedleDetection.HubConfidence exposes it; NeedleVisualizer hides the hub marker
  below its threshold.

INCOMPATIBLE (do NOT assign - crashes InferenceEngine with KeyNotFoundException '423'):
  arm-needle-pose-320.onnx / lucas.onnx : 320, [1,18,N], {0: arm, 1: needle}
  NeedleDetector's kill switch disables itself after 8 consecutive failures if one
  of these is assigned; re-export with a compatible opset before trying again.

LEGACY MODELS (kept for reference; class orders CONFLICT — check before use):
  best.onnx / best_v2.onnx              : 640, [1,18,N], {0: Syringe, 1: Arm}  (swapped!)
  custom-arm-detector.onnx              : 640, [1,5,N], box-only

When assigning a legacy 2-class model, set CustomArmDetector._armClassId to the
arm's index in THAT model's convention (see table above).
