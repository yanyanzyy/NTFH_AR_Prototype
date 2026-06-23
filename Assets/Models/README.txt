This folder holds the one model used by the app: the arm + needle POSE model,
arm-needle-pose-320.onnx (produced by the Training/ pipeline).

How it's produced (see Training/README.md for the full pipeline):

  cd Training
  python scripts/01_label.py            # label arm + needle keypoints
  python scripts/02_prepare_dataset.py
  python scripts/03_train.py
  python scripts/04_export.py --weights runs/arm_needle_pose/weights/best.pt

Unity imports it as a Sentis ModelAsset. Assign it to the "Model Asset" slot on
the CustomArmDetector component (ArmDetectionScene) and set Input Size = 320.

Output layout, features-first [1, 12, N]:
  0..3   box cx, cy, w, h        (input-pixel scale, 320x320)
  4..5   class scores            (0 = arm, 1 = needle)
  6..11  keypoints               (kx0, ky0, v0, kx1, ky1, v1)
         arm:    kpt0 = proximal, kpt1 = distal
         needle: kpt0 = tip,      kpt1 = hub

custom-arm-detector.onnx is the PREVIOUS box-only model. It is kept only so the
scene's Model Asset slot isn't broken before you swap in the pose model. Delete
it once arm-needle-pose-320.onnx is assigned.
