"""
Mask UNLABELED instances out of the keypoint losses.

WHY
---
Only ~14% of our images carry real keypoints (the manually-labeled arm_ burst
captures). The other ~86% are box-only supervision (phleb frames, web stills)
and 02_prepare_dataset.py gives them placeholder keypoints at visibility 0.

Ultralytics reads visibility 0 as "this keypoint is ABSENT" - a real training
signal - when for us it means "nobody labeled this". Two things go wrong in
v8PoseLoss.calculate_keypoints_loss:

  1. SUPPRESSION. The keypoint-objectness loss is not masked at all:

         kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

     Every placeholder instance hands the visibility head a target of 0, so
     ~86% of the training signal says "arms have no visible keypoints". The
     head learns exactly that and predicts visibility near 0, which is what
     the runtime confidence gate then rejects.

  2. DILUTION. KeypointLoss.forward ends in .mean() over every row. Unlabeled
     rows contribute 0 to the numerator but still count in the denominator, so
     the localization gradient is scaled down by roughly the labeled fraction.

Both are fixed by dropping instances with NO visible keypoints from BOTH
keypoint losses. Those instances still train the box/cls/dfl heads exactly as
before - we only stop them from voting on keypoints they were never labeled
for.

HOW
---
Rather than reimplement calculate_keypoints_loss (which would have to
replicate its gather logic and would rot on the next Ultralytics bump), we
wrap the two callables it delegates to. Both already receive everything the
mask needs:

  * keypoint_loss(pred, gt, kpt_mask, area) - kpt_mask is per-keypoint
    visibility, so kpt_mask.any(1) is "this instance was labeled".
  * bce_pose(pred_vis, target_vis)          - target_vis IS kpt_mask as float,
    so target.any(1) is the same test.

Per-INSTANCE masking (not per-image filtering) is what mosaic augmentation
requires: one mosaic tile can mix a labeled capture with three box-only
frames, and only the labeled instance should reach the keypoint heads.

USAGE
-----
    import pose_loss_patch; pose_loss_patch.apply()

before model.train(). 05_train.py does this by default; pass --raw-pose-loss
to train with stock Ultralytics behaviour (for an A/B against arm_pose_v5).
"""
import torch
import torch.nn as nn

from ultralytics.utils import LOGGER

_applied = False


def _labeled_rows(target: torch.Tensor) -> torch.Tensor:
    """Per-instance mask: True where the instance has >=1 visible keypoint.

    target is (n_instances, n_keypoints) - either kpt_mask or its float cast.
    """
    return target.reshape(target.shape[0], -1).any(dim=1)


class MaskedKeypointLoss(nn.Module):
    """KeypointLoss restricted to instances that actually carry keypoint labels."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        keep = _labeled_rows(kpt_mask != 0)
        if not bool(keep.any()):
            # No labeled instance in this batch: contribute nothing, but stay
            # attached to the graph so DDP still sees the head's parameters.
            return pred_kpts.sum() * 0.0
        return self.inner(pred_kpts[keep], gt_kpts[keep], kpt_mask[keep], area[keep])


class MaskedBCEPose(nn.Module):
    """bce_pose restricted to instances that actually carry keypoint labels.

    Unlabeled instances would otherwise supply an all-zero visibility target,
    training the head to declare every keypoint invisible.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, pred, target):
        keep = _labeled_rows(target != 0)
        if not bool(keep.any()):
            return pred.sum() * 0.0
        return self.inner(pred[keep], target[keep])


def apply() -> bool:
    """Patch v8PoseLoss so unlabeled instances stop voting on keypoints.

    Idempotent. Returns True if the patch is in effect.
    """
    global _applied
    if _applied:
        return True

    from ultralytics.utils.loss import v8PoseLoss

    original_init = v8PoseLoss.__init__

    def patched_init(self, model):
        original_init(self, model)
        missing = [a for a in ("keypoint_loss", "bce_pose") if not hasattr(self, a)]
        if missing:
            # Ultralytics restructured the pose loss - fail loudly rather than
            # silently training with the suppression bug back in place.
            raise RuntimeError(
                f"pose_loss_patch: v8PoseLoss is missing {missing}. The keypoint-loss "
                f"masking patch no longer applies to this Ultralytics version; review "
                f"calculate_keypoints_loss before training."
            )
        self.keypoint_loss = MaskedKeypointLoss(self.keypoint_loss)
        self.bce_pose = MaskedBCEPose(self.bce_pose)
        LOGGER.info(
            "pose_loss_patch: keypoint + kobj losses restricted to instances with "
            "labeled keypoints (placeholder-visibility-0 instances excluded)."
        )

    v8PoseLoss.__init__ = patched_init
    _applied = True
    return True
