"""
Import the pre-converted arm data (data/arm-pose-converted/) into the arm-only
model's label format.

That folder is a detection dataset padded to 4-keypoint pose format: every arm
has a box and 4 keypoint slots. Depending on how it was generated the first two
keypoints are either real PCA-derived axis endpoints (v=2) or placeholders on
the box centre (v=0). The arm-only model uses 2 keypoints, so each line is
truncated to its first two keypoint slots and the class is remapped to 0:

    cls cx cy w h  k0 k0 v0  k1 k1 v1  k2 k2 v2  k3 k3 v3   (17 tokens, class 1)
 -> 0   cx cy w h  k0 k0 v0  k1 k1 v1                       (11 tokens, class 0=arm)

Lines that are already 11 tokens (2-kpt) are kept as-is apart from the class
remap. Keypoints with v=0 contribute no keypoint loss in Ultralytics pose
training, so purely box-labeled arms are learned as box-only - fine for
detection, but keypoint-label some real mannequin captures with 01_label.py
to get a usable proximal/distal axis.

    python scripts/import_arm_boxes.py
    python scripts/import_arm_boxes.py --src data/arm-pose-converted

Images go to data/raw/, labels to data/labels/, prefixed arm__ so stems never
collide with other frames. Run 02_prepare_dataset.py afterwards.
"""
import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"

ARM_CLASS = 0
TOKENS_2KPT = 11          # cls + 4 box + 2 keypoints * 3
TOKENS_4KPT = 17          # cls + 4 box + 4 keypoints * 3
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(images_dir: Path, stem: str):
    for ext in IMG_EXTS:
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def convert_line(tokens):
    """One 11- or 17-token line -> 11-token arm-only line, or None if malformed."""
    if len(tokens) not in (TOKENS_2KPT, TOKENS_4KPT):
        return None
    return " ".join([str(ARM_CLASS), *tokens[1:TOKENS_2KPT]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "arm-pose-converted"),
                    help="folder with images/ and labels/ (pose txt, 2 or 4 kpts)")
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
        print(f"  {skipped_bad} malformed label line(s) skipped (expected {TOKENS_2KPT} or {TOKENS_4KPT} tokens).")
    print("Next: python scripts/02_prepare_dataset.py  (then 05_train.py, 07_export.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
