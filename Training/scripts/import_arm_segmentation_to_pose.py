"""
Step 5 (one-off) - Convert a Roboflow segmentation-format arm dataset into
YOLO-pose labels compatible with the syringe pose notebook, so both classes
can be trained in ONE combined model.

INPUT  : a Roboflow YOLOv11 export with polygon segmentation labels
         (class x1 y1 x2 y2 ... xn yn, normalized, variable point count)
OUTPUT : a flat images/ + labels/ folder (zippable) with 4-keypoint pose
         labels matching the syringe notebook's kpt_shape=[4,3]:

            class cx cy w h  px0 py0 v0  px1 py1 v1  px2 py2 v2  px3 py3 v3

For each polygon we run PCA to find the arm's major axis, then take the two
polygon vertices furthest apart along that axis as kpt0/kpt1 (visibility=2).
kpt2/kpt3 are padding (visibility=0, ignored by the pose loss) so this class
can share a 4-keypoint head with "Syringe".

CAVEAT: these stock photos have no visible cue for which axis-end is
anatomically proximal (elbow) vs distal (wrist), so kpt0 is assigned by a
fixed image-space rule (smaller y = "topmost" in the photo) rather than true
anatomy. Treat this dataset as weak/general arm-shape pretraining only -
combine with a smaller batch of correctly hand-labeled mannequin images
(via 01_label.py) for the real proximal/distal semantics.

USAGE
    python scripts/import_arm_segmentation_to_pose.py \
        --input  "C:\\Users\\Xerxes\\Downloads\\Telegram Desktop\\human-arms-detection.yolov11" \
        --output "C:\\Users\\Xerxes\\Downloads\\Telegram Desktop\\arm-pose-converted" \
        --class-id 1

Then zip the output folder and upload it to your Drive alongside the
syringe Key_point_labels.zip files, and add its path to DATA_SOURCES in
the notebook (cell-2). Update cell-6's yaml_data to:
    names: {0: 'Syringe', 1: 'Arm'}
    kpt_shape: [4, 3]
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Filenames containing these substrings are product photos (sleeves, jackets,
# compression gear, watches, etc.) rather than real arms - exclude them so
# the model doesn't learn "fabric/product = arm".
EXCLUDE_KEYWORDS = [
    "sleeve", "cover", "jacket", "glove", "strap", "watch", "tattoo", "jewel",
    "sunscreen", "bracelet", "warmer", "cooling", "uv", "sun-protect",
    "compression", "lymphedema", "denim", "hoodie", "hooded", "puffer",
]


def is_excluded(filename: str) -> bool:
    lower = filename.lower()
    return any(kw in lower for kw in EXCLUDE_KEYWORDS)


def parse_polygon_line(line: str):
    parts = line.strip().split()
    if not parts:
        return None, None
    cls = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2)  # normalized (x, y)
    return cls, pts


def polygon_to_pose(pts_px: np.ndarray):
    """
    pts_px: (N, 2) polygon vertices in pixel space.
    Returns (box_xyxy, kpt0, kpt1) all in pixel space, or None if degenerate.
    """
    if pts_px.shape[0] < 2:
        return None

    centroid = pts_px.mean(axis=0)
    centered = pts_px - centroid

    # PCA via covariance eigen-decomposition; major axis = top eigenvector.
    cov = np.cov(centered.T)
    if np.any(np.isnan(cov)) or np.allclose(cov, 0):
        # Degenerate (all points identical / collinear-zero) - fall back to bbox diag.
        x_min, y_min = pts_px.min(axis=0)
        x_max, y_max = pts_px.max(axis=0)
        kpt_a = np.array([x_min, y_min])
        kpt_b = np.array([x_max, y_max])
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, np.argmax(eigvals)]  # unit vector, longest spread

        proj = centered @ major_axis
        kpt_a = pts_px[np.argmin(proj)]
        kpt_b = pts_px[np.argmax(proj)]

    # Fixed, deterministic rule (no anatomical meaning): smaller image-y = kpt0.
    if kpt_a[1] <= kpt_b[1]:
        kpt0, kpt1 = kpt_a, kpt_b
    else:
        kpt0, kpt1 = kpt_b, kpt_a

    x_min, y_min = pts_px.min(axis=0)
    x_max, y_max = pts_px.max(axis=0)
    return (x_min, y_min, x_max, y_max), kpt0, kpt1


def convert_label_file(label_path: Path, img_w: int, img_h: int, class_id: int,
                        disable_keypoints: bool = False):
    """Returns a list of pose-format label lines (one per instance)."""
    text = label_path.read_text().strip()
    if not text:
        return []  # background negative - keep as empty

    out_lines = []
    for raw_line in text.splitlines():
        cls, pts_norm = parse_polygon_line(raw_line)
        if pts_norm is None:
            continue

        pts_px = pts_norm * np.array([img_w, img_h])
        result = polygon_to_pose(pts_px)
        if result is None:
            continue
        (x_min, y_min, x_max, y_max), kpt0, kpt1 = result

        cx = (x_min + x_max) / 2.0 / img_w
        cy = (y_min + y_max) / 2.0 / img_h
        bw = (x_max - x_min) / img_w
        bh = (y_max - y_min) / img_h

        # Padding keypoints: box centre, visibility 0 (ignored by pose loss).
        pad_x, pad_y = cx, cy

        if disable_keypoints:
            # Box/class supervision only - no orientation signal from this
            # weakly-labeled stock data. All 4 keypoints invisible (v=0).
            k0x, k0y = cx, cy
            k1x, k1y = cx, cy
            v0 = v1 = 0
        else:
            k0x, k0y = kpt0[0] / img_w, kpt0[1] / img_h
            k1x, k1y = kpt1[0] / img_w, kpt1[1] / img_h
            v0 = v1 = 2

        out_lines.append(
            f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{k0x:.6f} {k0y:.6f} {v0} "
            f"{k1x:.6f} {k1y:.6f} {v1} "
            f"{pad_x:.6f} {pad_y:.6f} 0 "
            f"{pad_x:.6f} {pad_y:.6f} 0"
        )
    return out_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Roboflow dataset root (has train/valid/test)")
    ap.add_argument("--output", required=True, help="destination folder for converted pose dataset")
    ap.add_argument("--class-id", type=int, default=1, help="class id to assign (Arm). Syringe is usually 0.")
    ap.add_argument("--preview-count", type=int, default=12, help="number of preview images to render for sanity-checking")
    ap.add_argument("--disable-keypoints", action="store_true",
                     help="Mark all 4 keypoints invisible (v=0) - box/class supervision only, "
                          "no orientation signal. Use this since the auto-derived keypoint "
                          "direction is inconsistent across image orientations.")
    args = ap.parse_args()

    src_root = Path(args.input)
    out_root = Path(args.output)
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_preview = out_root / "preview"
    for d in (out_images, out_labels, out_preview):
        d.mkdir(parents=True, exist_ok=True)

    splits = [d for d in ("train", "valid", "test") if (src_root / d).exists()]
    if not splits:
        raise SystemExit(f"No train/valid/test folders found under {src_root}")

    total_images = 0
    total_instances = 0
    skipped = 0
    excluded = 0
    preview_budget = args.preview_count

    for split in splits:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if not img_dir.exists():
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        before = len(images)
        images = [p for p in images if not is_excluded(p.name)]
        excluded += before - len(images)
        print(f"[{split}] {len(images)} images ({before - len(images)} excluded)")

        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            dest_name = f"{split}_{img_path.name}"
            dest_img = out_images / dest_name
            dest_lbl = out_labels / f"{Path(dest_name).stem}.txt"

            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as ex:
                print(f"  skip unreadable {img_path.name}: {ex}")
                skipped += 1
                continue

            if lbl_path.exists():
                lines = convert_label_file(lbl_path, w, h, args.class_id, args.disable_keypoints)
            else:
                lines = []  # no label file at all -> background negative

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

            total_images += 1
            total_instances += len(lines)

            if preview_budget > 0 and lines:
                preview_budget -= 1
                _write_preview(img_path, lines, out_preview / dest_name)

    print("\n" + "=" * 50)
    print(f"Converted {total_images} images ({total_instances} arm instances), skipped {skipped}, excluded {excluded} (product/clothing keywords).")
    print(f"Output: {out_root}")
    print(f"Spot-check {args.preview_count} previews in: {out_preview}")
    print("=" * 50)
    print("\nNext steps:")
    print(f"  1. Spot-check images in {out_preview} - keypoints should sit at the arm's two ends.")
    print(f"  2. Zip the '{out_root.name}' folder and upload it to your Drive (e.g. ITP-Venipuncture/).")
    print("  3. Add its path to DATA_SOURCES in the notebook's cell-2.")
    print("  4. In cell-6, set names: {0: 'Syringe', 1: 'Arm'}, kpt_shape: [4, 3], flip_idx: [0, 1, 2, 3].")
    return 0


def _write_preview(img_path: Path, lines, dest: Path):
    import cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for line in lines:
        parts = [float(x) for x in line.split()[1:]]
        cx, cy, bw, bh = parts[0:4]
        k0x, k0y, v0 = parts[4:7]
        k1x, k1y, v1 = parts[7:10]

        x_min = int((cx - bw / 2) * w)
        y_min = int((cy - bh / 2) * h)
        x_max = int((cx + bw / 2) * w)
        y_max = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)

        p0 = (int(k0x * w), int(k0y * h))
        p1 = (int(k1x * w), int(k1y * h))
        cv2.line(img, p0, p1, (0, 200, 0), 2)
        cv2.circle(img, p0, 6, (0, 0, 255), -1)   # kpt0 = red (top end)
        cv2.circle(img, p1, 6, (255, 0, 0), -1)   # kpt1 = blue (bottom end)

    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)


if __name__ == "__main__":
    sys.exit(main())
