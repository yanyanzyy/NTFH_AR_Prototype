"""
Import the pre-labeled "Syringe" data (data/needle/) into the combined model as
the `needle` class, KEEPING all 4 keypoints.

The combined model is 4-keypoint (kpt_shape [4,3]): arm and needle share the same
keypoint slot count. The needle source is already 4-kpt (4 points down the syringe
axis, k0 = tip ... k3 = plunger/hub), single class 0 = Syringe. The only change
needed is the class index:

    class 0 (Syringe)  ->  class 1 (needle)     box + all 4 keypoints kept as-is

It copies each image into data/raw/ and writes the label into data/labels/
(prefixed v1__/v2__ so stems never collide), so the normal pipeline picks them up
with no changes. Run 02_prepare_dataset.py afterwards.

    python scripts/needle_to_combined.py            # add --previews 6 to spot-check
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
NEEDLE = DATA / "needle"
RAW = DATA / "raw"
LABELS = DATA / "labels"
PREVIEW = DATA / "preview"

NEEDLE_CLASS = 1          # 'needle' in the combined model (arm=0, needle=1)
NUM_TOKENS = 17           # cls + 4 box + 4 keypoints * 3
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

# (tag, images_dir, labels_dir) for each labeled export under data/needle/.
SOURCES = [
    ("v1", NEEDLE / "All images", NEEDLE / "All images" / "Yolo_pose_v1" / "labels" / "train"),
    ("v2", NEEDLE / "Pose_2", NEEDLE / "Pose_2" / "Yolo_pose_v2" / "labels" / "Train"),
]


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def convert_line(tokens):
    """One 17-token Syringe line -> same line with class remapped to needle, or None."""
    if len(tokens) != NUM_TOKENS:
        return None
    return " ".join([str(NEEDLE_CLASS), *tokens[1:]])


def maybe_preview(img_path, lines, dest):
    try:
        import cv2
    except ImportError:
        return
    im = cv2.imread(str(img_path))
    if im is None:
        return
    h, w = im.shape[:2]
    for line in lines:
        t = line.split()
        pts = [(int(float(t[5 + i * 3]) * w), int(float(t[6 + i * 3]) * h)) for i in range(4)]
        for i in range(3):
            cv2.line(im, pts[i], pts[i + 1], (0, 128, 255), 2)
        for i, p in enumerate(pts):
            cv2.circle(im, p, 6, (0, 128, 255), -1)
            cv2.putText(im, f"k{i}", (p[0] + 8, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
    cv2.imwrite(str(dest), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--previews", type=int, default=0,
                    help="write this many spot-check overlays to data/preview/ (needs opencv)")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    if args.previews:
        PREVIEW.mkdir(parents=True, exist_ok=True)

    converted = skipped_no_img = skipped_bad = previews_done = 0
    for tag, images_dir, labels_dir in SOURCES:
        if not labels_dir.exists():
            print(f"  [warn] labels dir not found, skipping: {labels_dir}")
            continue
        for lbl in sorted(labels_dir.glob("*.txt")):
            img = find_image(images_dir, lbl.stem)
            if img is None:
                skipped_no_img += 1
                continue
            out_lines = []
            for raw_line in lbl.read_text().splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                conv = convert_line(raw_line.split())
                if conv is None:
                    skipped_bad += 1
                    continue
                out_lines.append(conv)

            name = f"{tag}__{img.stem}"
            shutil.copy2(img, RAW / f"{name}{img.suffix.lower()}")
            (LABELS / f"{name}.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
            converted += 1

            if args.previews and previews_done < args.previews and out_lines:
                maybe_preview(img, out_lines, PREVIEW / f"{name}.jpg")
                previews_done += 1

    print(f"Converted {converted} image(s) -> {RAW} (+labels in {LABELS}), 4 keypoints, class=needle.")
    if skipped_no_img:
        print(f"  {skipped_no_img} label(s) had no matching image and were skipped.")
    if skipped_bad:
        print(f"  {skipped_bad} malformed label line(s) skipped (expected {NUM_TOKENS} tokens).")
    if previews_done:
        print(f"  Wrote {previews_done} spot-check overlay(s) to {PREVIEW}.")
    print("Next: python scripts/02_prepare_dataset.py  (then 03_train.py, 04_export.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
