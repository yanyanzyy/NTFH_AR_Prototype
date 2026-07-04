"""
Import any folder of images as BACKGROUND NEGATIVES (empty label files).

Use for frames that contain NO mannequin arm — especially egocentric footage
full of real human hands/arms, which teaches the model not to fire on people.
Good sources:

  EgoHands (head-mounted camera, human hands/arms):
      http://vision.soic.indiana.edu/projects/egohands/  (egohands_data.zip)
      -> unzip, then point --src at the _LABELLEDSAMPLES folder
  Your own headset captures of the room/table WITHOUT the mannequin arm.

    python scripts/import_negatives.py --src ~/Downloads/egohands/_LABELLEDSAMPLES --max 400
    python scripts/import_negatives.py --src ~/captures/no-arm --prefix bg

Images -> data/raw/ (prefixed), empty labels -> data/labels/. Negatives should
stay a minority of the dataset (~10-20%); the --max default keeps that sane.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder to scan recursively for images")
    ap.add_argument("--max", type=int, default=400, help="cap imported images (0 = all)")
    ap.add_argument("--prefix", default="neg", help="filename prefix (avoids stem collisions)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not src.is_dir():
        raise SystemExit(f"{src} is not a directory.")

    images = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit(f"No images found under {src}.")

    random.seed(args.seed)
    random.shuffle(images)
    if args.max:
        images = images[: args.max]

    RAW.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    imported = 0
    for img in images:
        # Parent folder in the stem keeps EgoHands' repeated frame names unique.
        name = f"{args.prefix}__{img.parent.name}__{img.stem}"
        dest = RAW / f"{name}{img.suffix.lower()}"
        if dest.exists():
            continue
        shutil.copy2(img, dest)
        (LABELS / f"{name}.txt").write_text("")
        imported += 1

    print(f"Imported {imported} background negative(s) -> {RAW} (empty labels in {LABELS}).")
    print("Next: python scripts/02_prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
