# Evaluation - arm_pose_v7 (test split)

Weights: `C:\Github Desktop\NTFH_AR_Prototype\Training\runs\arm_pose_v7\weights\best.pt`

## Detection (all images in split)

| images | mAP50 | mAP75 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|
| 2044 | 0.9655 | 0.9131 | 0.8825 | 0.9535 | 0.9181 |

Mean box IoU over matched instances (from the pose subset): **0.702** (median 0.7185).

## Pose (keypoint-labeled images only)

Scored on 263 of 2044 images. The rest carry placeholder keypoints
at visibility 0 and cannot be scored; including them understates pose mAP.

| images | mAP50 | mAP75 | mAP50-95 | mean OKS | matched | missed | detection rate |
|---|---|---|---|---|---|---|---|
| 263 | 0.9151 | 0.7859 | 0.7758 | 0.5234 | 163 | 100 | 62.0% |

## Keypoint localization error

| metric | value |
|---|---|
| median error | 19.04 px |
| mean error | 21.22 px |
| RMSE | 24.97 px |
| 90th percentile | 39.61 px |
| PCK@5% of arm length | 28.8% |
| PCK@10% of arm length | 67.5% |
| PCK@20% of arm length | 96.3% |

### Per keypoint

| keypoint | n | median px | PCK@10% |
|---|---|---|---|
| kpt0 (proximal) | 163 | 20.9 | 61.3% |
| kpt1 (distal) | 163 | 17.7 | 73.6% |

![error](keypoint_error.png)

![qualitative](qualitative.jpg)
