"""
Step 6 - Evaluate a trained arm-pose model and write a reportable summary.

    python scripts/06_evaluate.py --weights runs/arm_pose_v5/weights/best.pt
    python scripts/06_evaluate.py --weights ... --split val

The dataset carries two tiers of supervision and they must be scored against
different denominators, or the headline numbers are wrong:

  * DETECTION is supervised by every image (~2000 in the test split), so box
    metrics are computed over the whole split.
  * POSE is supervised only by the manually-labeled arm_ captures. The phleb
    and web images carry placeholder keypoints at visibility 0, and a model
    that predicts keypoints on them scores them as false positives. Computing
    pose mAP over the full split therefore reports a number that is mostly an
    artifact of unlabeled data - arm_pose_v4 scored 0.023 that way versus
    0.672 on the labeled subset. Pose metrics here are computed on a
    keypoint-only subset built from the split.

Alongside mAP this reports the metric that actually matters for an AR overlay:
how far, in pixels, a predicted keypoint sits from the true one. That is
summarized as PCK (percentage of correct keypoints) at several tolerances
expressed as a fraction of the arm's own length, so it is scale-invariant.

Box localization is additionally summarized as mean IoU over matched instances
(the overlap quality behind the detection mAP), and keypoint localization as
mean OKS, the keypoint-space analog of IoU where 1.0 is a perfect prediction.

Outputs land in runs/<run>/evaluation/:
    summary.md          human-readable report
    metrics.csv         one row, machine-readable
    keypoint_error.png  error distribution, overall and per keypoint
    qualitative.jpg     predictions (colour) vs ground truth (hollow circles)
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
POSE = DATA / "pose"
RUNS = HERE.parent / "runs"

PCK_THRESHOLDS = (0.05, 0.10, 0.20)
KPT_NAMES = ("kpt0 (proximal)", "kpt1 (distal)")

# Object Keypoint Similarity falloff. COCO derives a per-keypoint constant from
# hand-annotation variance; there is no such calibration for a 2-point arm, so
# this is a single reasonable stand-in (~mid-range of COCO's k). Larger is more
# forgiving. OKS is the keypoint analog of box IoU: 1.0 is a perfect prediction.
OKS_KAPPA = 0.10


def visible_keypoints(text: str) -> bool:
    for line in text.splitlines():
        p = line.split()
        if len(p) == 11 and (p[7] != "0" or p[10] != "0"):
            return True
    return False


def build_keypoint_subset(split: str, dest: Path) -> int:
    """Copy the keypoint-labeled images of `split` into a standalone dataset."""
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "images" / split).mkdir(parents=True)
    (dest / "labels" / split).mkdir(parents=True)

    n = 0
    for lbl in sorted((POSE / "labels" / split).iterdir()):
        text = lbl.read_text()
        if not visible_keypoints(text):
            continue
        img = next((POSE / "images" / split).glob(f"{lbl.stem}.*"), None)
        if img is None:
            continue
        shutil.copy2(img, dest / "images" / split / img.name)
        shutil.copy2(lbl, dest / "labels" / split / lbl.name)
        n += 1

    rel = f"images/{split}"
    (dest / "data.yaml").write_text(yaml.safe_dump({
        "path": str(dest.resolve()),
        "train": rel, "val": rel, "test": rel,
        "kpt_shape": [2, 3], "flip_idx": [0, 1], "names": {0: "arm"},
    }, sort_keys=False))
    return n


def read_gt(lbl: Path, w: int, h: int):
    """Ground-truth (box_xyxy, kpt0, kpt1) in pixels for each labeled instance."""
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) != 11 or (p[7] == "0" and p[10] == "0"):
            continue
        v = list(map(float, p[1:]))
        cx, cy, bw, bh = v[0] * w, v[1] * h, v[2] * w, v[3] * h
        box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        k0 = np.array([v[4] * w, v[5] * h])
        k1 = np.array([v[7] * w, v[8] * h])
        out.append((box, k0, k1))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def keypoint_errors(model, subset: Path, split: str):
    """Per-keypoint error (normalized by arm length), matched-box IoU and OKS."""
    err_px = [[], []]
    err_rel = [[], []]
    box_iou = []
    oks = []
    matched = missed = 0

    for img_path in sorted((subset / "images" / split).iterdir()):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = read_gt(subset / "labels" / split / f"{img_path.stem}.txt", w, h)
        if not gts:
            continue

        res = model.predict(str(img_path), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.empty((0, 2, 2))

        for gt_box, g0, g1 in gts:
            arm_len = float(np.linalg.norm(g1 - g0))
            if arm_len < 1:
                continue
            best, best_iou = -1, 0.0
            for i, pb in enumerate(boxes):
                s = iou(gt_box, pb)
                if s > best_iou:
                    best, best_iou = i, s
            if best < 0 or best_iou < 0.5 or best >= len(kpts):
                missed += 1
                continue
            matched += 1
            box_iou.append(best_iou)
            # OKS scale is the object area (COCO convention: s^2 = box area).
            scale2 = max(1.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
            terms = []
            for j, gp in enumerate((g0, g1)):
                d = float(np.linalg.norm(kpts[best][j] - gp))
                err_px[j].append(d)
                err_rel[j].append(d / arm_len)
                terms.append(np.exp(-(d * d) / (2 * scale2 * OKS_KAPPA ** 2)))
            oks.append(float(np.mean(terms)))

    return ([np.array(e) for e in err_px], [np.array(e) for e in err_rel],
            np.array(box_iou), np.array(oks), matched, missed)


def plot_errors(err_rel, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allr = np.concatenate([e for e in err_rel if len(e)]) if any(len(e) for e in err_rel) else np.array([0.0])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].hist(allr * 100, bins=40, color="#3b7dd8", edgecolor="white")
    ax[0].set_title("Keypoint error distribution")
    ax[0].set_xlabel("error (% of arm length)")
    ax[0].set_ylabel("keypoints")
    for t in PCK_THRESHOLDS:
        ax[0].axvline(t * 100, color="#d8543b", ls="--", lw=1)
        ax[0].text(t * 100, ax[0].get_ylim()[1] * 0.95, f" {int(t*100)}%",
                   color="#d8543b", fontsize=8, va="top")

    xs = np.linspace(0, 0.5, 200)
    for j, e in enumerate(err_rel):
        if not len(e):
            continue
        ax[1].plot(xs * 100, [(e <= x).mean() * 100 for x in xs], label=KPT_NAMES[j])
    if len(allr):
        ax[1].plot(xs * 100, [(allr <= x).mean() * 100 for x in xs], "k--", label="both", lw=1)
    ax[1].set_title("PCK curve")
    ax[1].set_xlabel("tolerance (% of arm length)")
    ax[1].set_ylabel("correct keypoints (%)")
    ax[1].set_ylim(0, 100)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_qualitative(model, subset: Path, split: str, out: Path, n=12):
    imgs = sorted((subset / "images" / split).iterdir())
    if not imgs:
        return
    pick = [imgs[i] for i in np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)]
    tiles = []
    for p in pick:
        res = model.predict(str(p), verbose=False)[0]
        im = res.plot()
        h, w = im.shape[:2]
        for _, g0, g1 in read_gt(subset / "labels" / split / f"{p.stem}.txt", w, h):
            for g in (g0, g1):
                cv2.circle(im, (int(g[0]), int(g[1])), 9, (255, 128, 0), 2)
        tiles.append(cv2.resize(im, (426, 320)))
    cols = 4
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    rows = [r for r in rows if r.shape[1] == cols * 426]
    if rows:
        cv2.imwrite(str(out), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"{weights} not found.")
        return 1
    run_dir = weights.parent.parent
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    # --- detection, scored over the whole split -------------------------------
    full = model.val(data=str(POSE / "data.yaml"), split=args.split,
                     imgsz=args.imgsz, plots=False, verbose=False)
    n_full = len(list((POSE / "images" / args.split).iterdir()))

    # --- pose, scored only where keypoints exist ------------------------------
    subset = DATA / f"pose_kpt_{args.split}"
    n_kpt = build_keypoint_subset(args.split, subset)
    if n_kpt == 0:
        print(f"No keypoint-labeled images in split '{args.split}'.")
        return 1
    kpt = model.val(data=str(subset / "data.yaml"), split=args.split,
                    imgsz=args.imgsz, plots=False, verbose=False)

    err_px, err_rel, box_iou, oks, matched, missed = keypoint_errors(model, subset, args.split)
    allpx = np.concatenate([e for e in err_px if len(e)]) if any(len(e) for e in err_px) else np.array([])
    allrel = np.concatenate([e for e in err_rel if len(e)]) if any(len(e) for e in err_rel) else np.array([])

    plot_errors(err_rel, out_dir / "keypoint_error.png")
    plot_qualitative(model, subset, args.split, out_dir / "qualitative.jpg")

    pck = {t: float((allrel <= t).mean()) if len(allrel) else 0.0 for t in PCK_THRESHOLDS}
    row = {
        "run": run_dir.name,
        "split": args.split,
        "images_total": n_full,
        "images_with_keypoints": n_kpt,
        "box_mAP50": round(float(full.box.map50), 4),
        "box_mAP75": round(float(full.box.map75), 4),
        "box_mAP50_95": round(float(full.box.map), 4),
        "box_precision": round(float(full.box.mp), 4),
        "box_recall": round(float(full.box.mr), 4),
        "box_iou_mean": round(float(box_iou.mean()), 4) if len(box_iou) else None,
        "box_iou_median": round(float(np.median(box_iou)), 4) if len(box_iou) else None,
        "pose_mAP50": round(float(kpt.pose.map50), 4),
        "pose_mAP75": round(float(kpt.pose.map75), 4),
        "pose_mAP50_95": round(float(kpt.pose.map), 4),
        "instances_matched": matched,
        "instances_missed": missed,
        "kpt_err_median_px": round(float(np.median(allpx)), 2) if len(allpx) else None,
        "kpt_err_mean_px": round(float(allpx.mean()), 2) if len(allpx) else None,
        "kpt_err_rmse_px": round(float(np.sqrt((allpx ** 2).mean())), 2) if len(allpx) else None,
        "kpt_err_p90_px": round(float(np.percentile(allpx, 90)), 2) if len(allpx) else None,
        "oks_mean": round(float(oks.mean()), 4) if len(oks) else None,
        **{f"PCK@{int(t*100)}": round(pck[t], 4) for t in PCK_THRESHOLDS},
    }
    (out_dir / "metrics.csv").write_text(
        ",".join(row) + "\n" + ",".join("" if v is None else str(v) for v in row.values()) + "\n")

    det = 100 * matched / max(1, matched + missed)
    lines = [
        f"# Evaluation - {run_dir.name} ({args.split} split)",
        "",
        f"Weights: `{weights}`",
        "",
        "## Detection (all images in split)",
        "",
        f"| images | mAP50 | mAP75 | mAP50-95 | precision | recall |",
        f"|---|---|---|---|---|---|",
        f"| {n_full} | {row['box_mAP50']} | {row['box_mAP75']} | {row['box_mAP50_95']} | "
        f"{row['box_precision']} | {row['box_recall']} |",
        "",
        "Mean box IoU over matched instances (from the pose subset): "
        f"**{row['box_iou_mean']}** (median {row['box_iou_median']}).",
        "",
        "## Pose (keypoint-labeled images only)",
        "",
        f"Scored on {n_kpt} of {n_full} images. The rest carry placeholder keypoints",
        "at visibility 0 and cannot be scored; including them understates pose mAP.",
        "",
        f"| images | mAP50 | mAP75 | mAP50-95 | mean OKS | matched | missed | detection rate |",
        f"|---|---|---|---|---|---|---|---|",
        f"| {n_kpt} | {row['pose_mAP50']} | {row['pose_mAP75']} | {row['pose_mAP50_95']} | "
        f"{row['oks_mean']} | {matched} | {missed} | {det:.1f}% |",
        "",
        "## Keypoint localization error",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| median error | {row['kpt_err_median_px']} px |",
        f"| mean error | {row['kpt_err_mean_px']} px |",
        f"| RMSE | {row['kpt_err_rmse_px']} px |",
        f"| 90th percentile | {row['kpt_err_p90_px']} px |",
    ]
    for t in PCK_THRESHOLDS:
        lines.append(f"| PCK@{int(t*100)}% of arm length | {100*pck[t]:.1f}% |")
    lines += ["", "### Per keypoint", "", "| keypoint | n | median px | PCK@10% |", "|---|---|---|---|"]
    for j, name in enumerate(KPT_NAMES):
        e_px, e_rel = err_px[j], err_rel[j]
        if len(e_px):
            lines.append(f"| {name} | {len(e_px)} | {np.median(e_px):.1f} | "
                         f"{100*(e_rel <= 0.10).mean():.1f}% |")
    lines += ["", "![error](keypoint_error.png)", "", "![qualitative](qualitative.jpg)", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}")
    print(f"  detection : mAP50={row['box_mAP50']}  mAP50-95={row['box_mAP50_95']}  "
          f"IoU={row['box_iou_mean']}  ({n_full} images)")
    print(f"  pose      : mAP50={row['pose_mAP50']}  mAP50-95={row['pose_mAP50_95']}  "
          f"OKS={row['oks_mean']}  ({n_kpt} images)")
    if len(allpx):
        print(f"  keypoints : median {row['kpt_err_median_px']}px  "
              f"PCK@10%={100*pck[0.10]:.1f}%  detection rate {det:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
