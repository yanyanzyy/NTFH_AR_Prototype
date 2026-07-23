# Evaluation - arm_pose_v4 (test split)

Weights: `C:\Github Desktop\NTFH_AR_Prototype\Training\runs\arm_pose_v4\weights\best.pt`

## Detection (all images in split)

| images | mAP50 | mAP75 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| 2044 | 0.9919 | 0.9754 | 0.936 | 0.9763 | 0.9721 |

Mean box IoU over matched instances (from the pose subset): **0.857** (median 0.8664).

## Pose (keypoint-labeled images only)

Scored on 263 of 2044 images. The rest carry placeholder keypoints
at visibility 0 and cannot be scored; including them understates pose mAP.

| images | mAP50 | mAP75 | mAP50-95 | mean OKS | matched | missed | detection rate |
|---|---|---|---|---|---|---|---|
| 263 | 0.995 | 0.995 | 0.995 | 0.7492 | 263 | 0 | 100.0% |

## Keypoint localization error

| metric | value |
|---|---|
| median error | 9.62 px |
| mean error | 12.1 px |
| RMSE | 15.1 px |
| 90th percentile | 23.75 px |
| PCK@5% of arm length | 51.9% |
| PCK@10% of arm length | 92.6% |
| PCK@20% of arm length | 99.8% |

### Per keypoint

| keypoint | n | median px | PCK@10% |
|---|---|---|---|
| kpt0 (proximal) | 263 | 10.2 | 90.9% |
| kpt1 (distal) | 263 | 8.9 | 94.3% |

![error](keypoint_error.png)

![qualitative](qualitative.jpg)
