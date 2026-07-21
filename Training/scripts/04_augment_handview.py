"""
Step 4 - Hand-view (foreshortened) augmentation for the TRAIN split.

    python scripts/02_prepare_dataset.py      # build the split first
    python scripts/03_augment_closeups.py     # close-ups
    python scripts/04_augment_handview.py     # then hand-POV foreshortening
    python scripts/05_train.py --name arm_pose_v6

WHY
---
During a real draw the nurse looks down the arm from the HAND end: the wrist is
near and wide, the forearm recedes, the elbow is small and far. Every captured
image instead frames the arm side-on from ~1m, so that viewpoint is absent from
training and the model has never had to place keypoints on a foreshortened arm.

Ultralytics' geometric augs cannot produce it. degrees/translate/scale are
affine - they preserve parallel lines, so they can rotate and resize the arm
but never make one end larger than the other. Foreshortening is projective and
needs a homography.

This script fits a homography per image that pulls the proximal (elbow) end
toward the distal (wrist) end and narrows it, while widening the wrist end -
then crops in, because a hand view is also a close view. Keypoints transform
exactly under the homography; the box is the axis-aligned bound of its four
warped corners.

HONEST LIMITS
-------------
A warp cannot invent a viewpoint. It reproduces the GEOMETRY of foreshortening
(width gradient along the arm, compressed axis, keypoints at very different
scales) but not the APPEARANCE - no palm or fingers entering frame, no changed
self-occlusion or shading. Treat it as teaching the model the shape prior, not
as a substitute for real hand-POV captures.

It also inherits the softness problem measured on arm_pose_v5's zoom_ crops:
synthetic close-ups run ~8x lower Laplacian variance at 320px than real
captures, because cropping in reveals the camera's true detail while
downscaling a full frame sharpens it. That is intrinsic - no interpolation
setting fixes it. If softness ends up correlated with arm size, the model can
use it as a shortcut for scale and then misjudge real, sharp close-ups. This
script prints the sharpness of what it generated against the real captures so
the gap stays visible; --report-only measures without writing anything.

IMPORTANT
---------
Runs AFTER 02_prepare_dataset.py and only touches train, so no synthetic image
reaches val/test. Re-running 02 deletes data/pose entirely - re-run this after.
Running twice is safe: existing hview_ files are cleared first.

Keypoints pushed out of frame are marked visibility 0 rather than clamped, the
same convention 07 uses (see pose_label_io).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from pose_label_io import (format_label, has_keypoints, montage, parse_label,
                           plan_crop, sharpness)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
POSE = DATA / "pose"
PREFIX = "hview_"
SKIP_PREFIXES = (PREFIX,)          # zoom_ crops ARE valid sources: see --from-closeups
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def taper_for_vanish(near_gain, foreshorten, vanish):
    """Far-end width factor that puts the vanishing point `vanish` arm-lengths away.

    The trapezoid's sides converge to zero width at

        x = taper * foreshorten / (near_gain - taper)      [arm lengths]

    beyond the far end. Setting the taper directly is a trap: the obvious-looking
    taper=0.45 / near_gain=1.25 puts x at 0.28 arm-lengths, i.e. the vanishing
    point lands INSIDE the frame and the whole background collapses into a fan of
    radial smears around it. Solving for taper instead keeps that singularity
    safely off-frame:

        taper = near_gain * vanish / (vanish + foreshorten)
    """
    return near_gain * vanish / (vanish + foreshorten)


def handview_homography(prox, dist, half_w, foreshorten, vanish, near_gain):
    """Homography simulating a view down the arm axis from the DISTAL end.

    prox/dist are the two keypoints in pixels; half_w is the arm's half-width
    perpendicular to its axis. The source quad is the rectangle spanning the
    arm; the destination quad pulls the far end in by `foreshorten` and narrows
    it, while widening the near end to `near_gain`. That width gradient IS the
    depth cue. `vanish` (in arm lengths) sets how far off-frame the perspective
    singularity sits - see taper_for_vanish.

    Returns None when the arm is too short to fit a stable quad.
    """
    P = np.asarray(prox, dtype=np.float64)
    D = np.asarray(dist, dtype=np.float64)
    axis = D - P
    length = float(np.linalg.norm(axis))
    if length < 8.0 or half_w < 2.0:
        return None

    u = axis / length
    n = np.array([-u[1], u[0]])          # unit normal to the arm axis
    taper = taper_for_vanish(near_gain, foreshorten, vanish)

    src = np.float32([P - half_w * n, P + half_w * n,
                      D + half_w * n, D - half_w * n])

    far = D - foreshorten * axis         # elbow end, pulled in and narrowed
    dst = np.float32([far - taper * half_w * n, far + taper * half_w * n,
                      D + near_gain * half_w * n, D - near_gain * half_w * n])

    H = cv2.getPerspectiveTransform(src, dst)
    return H if np.all(np.isfinite(H)) else None


def warp_points(H, pts):
    """Apply H to Nx2 points. Returns (Nx2 warped, N bool 'in front of camera').

    A homography can map points behind the virtual camera (w <= 0), which come
    back as nonsense coordinates. Those are reported unusable rather than
    silently written into a label.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2]
    ok = w > 1e-6
    safe = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return hom[:, :2] / safe[:, None], ok


def arm_half_width(box, prox, dist):
    """Half-extent of the box perpendicular to the arm axis."""
    P, D = np.asarray(prox, float), np.asarray(dist, float)
    axis = D - P
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return 0.0
    n = np.array([-axis[1], axis[0]]) / length
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    corners = np.array([[box[0], box[1]], [box[2], box[1]],
                        [box[2], box[3]], [box[0], box[3]]])
    return float(np.max(np.abs((corners - centre) @ n)))


def warp_instances(instances, H, out_w, out_h):
    """Map instances through H. Returns None if the target arm did not survive."""
    kept = []
    for cls, box, kpts in instances:
        corners = np.array([[box[0], box[1]], [box[2], box[1]],
                            [box[2], box[3]], [box[0], box[3]]])
        wc, ok = warp_points(H, corners)
        if not ok.all():
            continue                      # box crossed the horizon - unusable
        x1, y1 = wc.min(axis=0)
        x2, y2 = wc.max(axis=0)
        clipped = np.array([max(0.0, x1), max(0.0, y1),
                            min(out_w, x2), min(out_h, y2)])
        if clipped[2] - clipped[0] <= 2 or clipped[3] - clipped[1] <= 2:
            continue

        new_kpts = []
        for (kx, ky, kv) in kpts:
            if kv == 0:
                new_kpts.append((0.0, 0.0, 0))
                continue
            moved, ok = warp_points(H, [(kx, ky)])
            (nx, ny), good = moved[0], bool(ok[0])
            if not good or not (0 <= nx < out_w and 0 <= ny < out_h):
                new_kpts.append((0.0, 0.0, 0))
            else:
                new_kpts.append((float(nx), float(ny), kv))
        kept.append((cls, clipped, new_kpts))
    return kept or None


def crop_instances(instances, crop, out_w, out_h, min_visible):
    """Map warped instances into a crop rectangle (same contract as 07)."""
    cx1, cy1, cx2, cy2 = crop
    sx, sy = out_w / (cx2 - cx1), out_h / (cy2 - cy1)

    kept = []
    for cls, box, kpts in instances:
        moved = np.array([(box[0] - cx1) * sx, (box[1] - cy1) * sy,
                          (box[2] - cx1) * sx, (box[3] - cy1) * sy])
        clipped = np.array([max(0.0, moved[0]), max(0.0, moved[1]),
                            min(out_w, moved[2]), min(out_h, moved[3])])
        bw, bh = clipped[2] - clipped[0], clipped[3] - clipped[1]
        if bw <= 2 or bh <= 2:
            continue
        full = (moved[2] - moved[0]) * (moved[3] - moved[1])
        if full > 0 and (bw * bh) / full < min_visible:
            continue
        new_kpts = []
        for kx, ky, kv in kpts:
            nx, ny = (kx - cx1) * sx, (ky - cy1) * sy
            if kv == 0 or not (0 <= nx < out_w and 0 <= ny < out_h):
                new_kpts.append((0.0, 0.0, 0))
            else:
                new_kpts.append((nx, ny, kv))
        kept.append((cls, clipped, new_kpts))
    return kept or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=("train", "val", "test"),
                    help="split to augment; val/test are refused unless --force")
    ap.add_argument("--per-image", type=int, default=2, help="views generated per source image")
    ap.add_argument("--min-foreshorten", type=float, default=0.35,
                    help="strongest foreshortening: elbow pulled to this fraction of the arm's length")
    ap.add_argument("--max-foreshorten", type=float, default=0.75,
                    help="mildest foreshortening")
    ap.add_argument("--vanish", type=float, default=3.0,
                    help="distance to the perspective vanishing point, in arm lengths. "
                         "Below ~1.5 the singularity enters the frame and the background "
                         "collapses into radial smears")
    ap.add_argument("--near-gain", type=float, default=1.25,
                    help="width of the near (wrist) end, as a fraction of the true width")
    ap.add_argument("--near-end", default="distal", choices=("distal", "proximal", "both"),
                    help="which end faces the camera; distal = the nurse's view down the hand")
    ap.add_argument("--min-occupancy", type=float, default=0.15,
                    help="lowest fraction of the frame the arm should cover after cropping")
    ap.add_argument("--max-occupancy", type=float, default=0.55, help="highest fraction")
    ap.add_argument("--jitter", type=float, default=0.15, help="crop centre offset")
    ap.add_argument("--min-visible", type=float, default=0.35,
                    help="minimum fraction of the arm box that must remain in frame")
    ap.add_argument("--from-closeups", action="store_true",
                    help="also use step 03's zoom_ crops as sources (compounds their softness)")
    ap.add_argument("--report-only", action="store_true",
                    help="measure and preview without writing into the split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="allow augmenting val/test")
    args = ap.parse_args()

    if args.split != "train" and not args.force:
        print(f"Refusing to augment '{args.split}': synthetic views in val/test would stop "
              f"them measuring real framing. Pass --force to override.")
        return 1

    img_dir = POSE / "images" / args.split
    lbl_dir = POSE / "labels" / args.split
    if not img_dir.is_dir():
        print(f"{img_dir} missing - run scripts/02_prepare_dataset.py first.")
        return 1

    skip = SKIP_PREFIXES if args.from_closeups else SKIP_PREFIXES + ("zoom_",)

    removed = 0
    if not args.report_only:
        for d in (img_dir, lbl_dir):
            for p in list(d.iterdir()):
                if p.name.startswith(PREFIX):
                    p.unlink()
                    removed += 1

    sources = []
    for lbl in sorted(lbl_dir.iterdir()):
        if lbl.suffix != ".txt" or lbl.name.startswith(skip):
            continue
        insts = parse_label(lbl.read_text(), 1, 1)
        if not any(has_keypoints(i) for i in insts):
            continue                      # box-only image: no axis to foreshorten
        img = next((p for p in img_dir.glob(f"{lbl.stem}.*")
                    if p.suffix.lower() in IMG_EXTS), None)
        if img:
            sources.append((img, lbl))

    if not sources:
        print(f"No keypoint-labeled images in {lbl_dir}.")
        return 1

    rng = random.Random(args.seed)
    made = skipped = kpt_dropped = kpt_total = 0
    occupancies, samples, sharp_new, sharp_src = [], [], [], []

    for img_path, lbl_path in sources:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        instances = parse_label(lbl_path.read_text(), w, h)
        target = next((i for i in instances if has_keypoints(i)), None)
        if target is None:
            continue
        sharp_src.append(sharpness(im))

        _, box, kpts = target
        prox, dist = (kpts[0][0], kpts[0][1]), (kpts[1][0], kpts[1][1])
        if not (kpts[0][2] and kpts[1][2]):
            skipped += args.per_image      # need both ends to define the axis
            continue
        half_w = arm_half_width(box, prox, dist)

        for k in range(args.per_image):
            near = args.near_end
            if near == "both":
                near = rng.choice(("distal", "proximal"))
            a, b = (prox, dist) if near == "distal" else (dist, prox)

            H = handview_homography(
                a, b, half_w,
                foreshorten=rng.uniform(args.min_foreshorten, args.max_foreshorten),
                vanish=args.vanish, near_gain=args.near_gain)
            if H is None:
                skipped += 1
                continue

            warped = warp_instances(instances, H, w, h)
            if warped is None:
                skipped += 1
                continue
            tgt = next((i for i in warped if has_keypoints(i)), None)
            if tgt is None:
                skipped += 1
                continue

            occ = rng.uniform(args.min_occupancy, args.max_occupancy)
            crop = plan_crop(w, h, tgt[1], occ, args.jitter, rng)
            if crop is None:
                skipped += 1
                continue
            final = crop_instances(warped, crop, w, h, args.min_visible)
            if final is None or not any(has_keypoints(i) for i in final):
                skipped += 1
                continue

            canvas = cv2.warpPerspective(im, H, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            x1, y1, x2, y2 = crop
            patch = canvas[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                skipped += 1
                continue
            patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)

            stem = f"{PREFIX}{img_path.stem}_{k}"
            if not args.report_only:
                cv2.imwrite(str(img_dir / f"{stem}{img_path.suffix}"), patch)
                (lbl_dir / f"{stem}.txt").write_text(format_label(final, w, h))
            made += 1
            sharp_new.append(sharpness(patch))

            for _, bx, kp in final:
                occupancies.append((bx[2] - bx[0]) * (bx[3] - bx[1]) / (w * h))
                for _, _, kv in kp:
                    kpt_total += 1
                    kpt_dropped += (kv == 0)
            if len(samples) < 8:
                samples.append((patch, final))

    if samples:
        montage(samples, DATA / "preview_handview.jpg")

    occ = np.array(occupancies) if occupancies else np.array([0.0])
    print(f"Source images (with both keypoints): {len(sources)}")
    if removed:
        print(f"Cleared {removed} file(s) from a previous run.")
    print(f"Views written : {made}   skipped: {skipped}"
          f"{'   (REPORT ONLY - nothing written)' if args.report_only else ''}")
    print(f"Arm occupancy : median {100*np.median(occ):.1f}%  "
          f"range {100*occ.min():.1f}%-{100*occ.max():.1f}%   (source images ~3%)")
    print(f"Keypoints out of frame -> visibility 0: {kpt_dropped}/{kpt_total} "
          f"({100*kpt_dropped/max(1,kpt_total):.1f}%)")
    if sharp_new and sharp_src:
        ratio = np.median(sharp_new) / max(1e-6, np.median(sharp_src))
        print(f"Sharpness @320: real sources {np.median(sharp_src):.0f}  ->  "
              f"generated {np.median(sharp_new):.0f}  ({ratio:.2f}x)")
        if ratio < 0.5:
            print("  WARNING: generated views are markedly softer than the real captures.")
            print("  If softness correlates with arm size the model can use it as a scale")
            print("  shortcut and misjudge real close-ups - the suspected v5 regression.")
            print("  Decorrelate it before trusting a retrain (see the module docstring).")
    if samples:
        print(f"Preview: {DATA / 'preview_handview.jpg'}")
    if not args.report_only:
        print(f"\n{args.split} split now has {len(list(img_dir.iterdir()))} images.")
        print("Next: python scripts/05_train.py --name arm_pose_v6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
