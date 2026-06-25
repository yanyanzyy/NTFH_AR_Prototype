"""
Import the pre-converted 4-keypoint arm data (data/arm-pose-converted/) into the
combined model as the `arm` class.

That folder is a detection dataset padded to 4-keypoint pose format: every arm has
a box and 4 keypoint slots, but the keypoints are placeholders (all sit on the box
centre with visibility 0 — i.e. NO real arm keypoints). Its class index is 1; the
combined model uses arm=0, so we remap:

    class 1 (arm-pose-converted)  ->  class 0 (arm)    box + 4 keypoint slots kept as-is

The placeholder keypoints (v=0) contribute no keypoint loss in Ultralytics pose
training, so the arm class is learned as box-only here. The needle class (from
needle_to_combined.py) supplies the real 4-keypoint supervision.

    python scripts/arm_pose_to_combined.py
    python scripts/arm_pose_to_combined.py --src data/arm-pose-converted

Images go to data/raw/, labels to data/labels/, prefixed arm__ so stems never
collide with the needle frames. Run 02_prepare_dataset.py afterwards.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"

ARM_CLASS = 0             # 'arm' in the combined model (arm=0, needle=1)
NUM_TOKENS = 17           # cls + 4 box + 4 keypoints * 3
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def convert_line(tokens):
    """One 17-token line -> same line with class remapped to arm, or None."""
    if len(tokens) != NUM_TOKENS:
        return None
    return " ".join([str(ARM_CLASS), *tokens[1:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "arm-pose-converted"),
                    help="folder with images/ and labels/ (4-kpt pose txt)")
    args = ap.parse_args()

    src = Path(args.src)
    img_dir, lbl_dir = src / "images", src / "labels"
    if not (img_dir.is_dir() and lbl_dir.is_dir()):
        raise SystemExit(f"Expected images/ and labels/ under {src}.")

    RAW.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    imported = boxes = skipped_no_img = skipped_bad = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = find_image(img_dir, lbl.stem)
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

        name = f"arm__{img.stem}"
        shutil.copy2(img, RAW / f"{name}{img.suffix.lower()}")
        (LABELS / f"{name}.txt").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
        imported += 1
        boxes += len(out_lines)

    print(f"Imported {imported} image(s), {boxes} arm box(es) -> {RAW} (+labels in {LABELS}), class=arm.")
    if skipped_no_img:
        print(f"  {skipped_no_img} label(s) had no matching image and were skipped.")
    if skipped_bad:
        print(f"  {skipped_bad} malformed label line(s) skipped (expected {NUM_TOKENS} tokens).")
    print("Next: python scripts/02_prepare_dataset.py  (then 03_train.py, 04_export.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
