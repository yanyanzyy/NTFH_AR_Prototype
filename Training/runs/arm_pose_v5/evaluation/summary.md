# Evaluation - arm_pose_v5 (test split)

Weights: `C:\Github Desktop\NTFH_AR_Prototype\Training\runs\arm_pose_v5\weights\best.pt`

## Detection (all images in split)

| images | mAP50 | mAP75 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| 2044 | 0.9734 | 0.9375 | 0.9047 | 0.9617 | 0.9328 |

Mean box IoU over matched instances (from the pose subset): **0.7604** (median 0.7744).

## Pose (keypoint-labeled images only)

Scored on 263 of 2044 images. The rest carry placeholder keypoints
at visibility 0 and cannot be scored; including them understates pose mAP.

| images | mAP50 | mAP75 | mAP50-95 | mean OKS | matched | missed | detection rate |
|---|---|---|---|---|---|---|---|
| 263 | 0.9257 | 0.8269 | 0.8173 | 0.5851 | 191 | 72 | 72.6% |

## Keypoint localization error

| metric | value |
|---|---|
| median error | 15.11 px |
| mean error | 17.86 px |
| RMSE | 21.44 px |
| 90th percentile | 32.29 px |
| PCK@5% of arm length | 34.8% |
| PCK@10% of arm length | 73.6% |
| PCK@20% of arm length | 94.5% |

### Per keypoint

| keypoint | n | median px | PCK@10% |
|---|---|---|---|
| kpt0 (proximal) | 191 | 17.0 | 68.1% |
| kpt1 (distal) | 191 | 13.9 | 79.1% |

![error](keypoint_error.png)

![qualitative](qualitative.jpg)
