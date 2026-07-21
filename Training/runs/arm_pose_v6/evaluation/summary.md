# Evaluation - arm_pose_v6 (test split)

Weights: `C:\Github Desktop\NTFH_AR_Prototype\Training\runs\arm_pose_v6\weights\best.pt`

## Detection (all images in split)

| images | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| 2044 | 0.9674 | 0.8814 | 0.9487 | 0.921 |

## Pose (keypoint-labeled images only)

Scored on 263 of 2044 images. The rest carry placeholder keypoints
at visibility 0 and cannot be scored; including them understates pose mAP.

| images | mAP50 | mAP50-95 | matched | missed | detection rate |
|---|---|---|---|---|---|
| 263 | 0.8986 | 0.8069 | 169 | 94 | 64.3% |

## Keypoint localization error

| metric | value |
|---|---|
| median error | 17.88 px |
| mean error | 20.62 px |
| 90th percentile | 38.78 px |
| PCK@5% of arm length | 32.2% |
| PCK@10% of arm length | 66.9% |
| PCK@20% of arm length | 94.1% |

### Per keypoint

| keypoint | n | median px | PCK@10% |
|---|---|---|---|
| kpt0 (proximal) | 169 | 24.6 | 53.8% |
| kpt1 (distal) | 169 | 13.5 | 79.9% |

![error](keypoint_error.png)

![qualitative](qualitative.jpg)
