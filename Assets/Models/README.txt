DEPLOYED MODEL: arm-pose-320.onnx  (arm_pose_v2 training run, 2026-07-07)
Assigned to CustomArmDetector in ArmDetectionScene. Single class, 320x320 input.

  Input : [1, 3, 320, 320]   (CustomArmDetector reads the size from the ONNX
                              itself, so the Inspector's Input Size can't break it)
  Output: [1, 11, 2100] features-first
    0..3   box cx, cy, w, h   (input-pixel scale, letterboxed 320x320)
    4      arm score          (single class: 0 = arm)
    5..10  keypoints          (kx0, ky0, v0, kx1, ky1, v1)
           kpt0 = proximal (near elbow), kpt1 = distal (wrist)

  KNOWN LIMITATION (arm_pose_v2): box detection is excellent (mAP50-95 ~0.93)
  but the keypoint head is UNTRAINED (pose mAP ~0). Do not trust keypoint
  outputs; the manager uses the fixed-world-axis fallback for arm orientation
  and box size for depth. Fix by labeling real proximal/distal keypoints and
  retraining (Training/README.md).

  Preprocessing: frames must be LETTERBOXED (aspect-fit + gray pad), matching
  Ultralytics train/val. CustomArmDetector does this; don't feed stretched frames.

Produced by the Training/ pipeline (see Training/README.md):

  cd Training
  python scripts/01_label.py
  python scripts/02_prepare_dataset.py
  python scripts/03_train.py --imgsz 320
  python scripts/04_export.py --weights runs/<run>/weights/best.pt

LEGACY MODELS (kept for reference; class orders CONFLICT — check before use):
  arm-needle-pose-320.onnx / lucas.onnx : 320, [1,18,N], {0: arm, 1: needle}
  best.onnx / best_v2.onnx              : 640, [1,18,N], {0: Syringe, 1: Arm}  (swapped!)
  custom-arm-detector.onnx              : 640, [1,5,N], box-only

When assigning a legacy 2-class model, set CustomArmDetector._armClassId to the
arm's index in THAT model's convention (see table above).
