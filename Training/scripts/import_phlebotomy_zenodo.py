"""
Import the Zenodo "simulated phlebotomy procedures" dataset as arm-only training
data — 11,884 third-person images of a blood draw performed on a MEDICAL
TRAINING ARM (the closest public match to our deployment target), CC BY 4.0.

    https://zenodo.org/records/16924786   (~321 MB zip)

Only the "Training Arm" class is kept; its boxes are converted to our 2-keypoint
arm-only pose format with placeholder keypoints (v=0, box-only supervision):

    <cls> cx cy w h [poly...]  ->  0 cx cy w h  cx cy 0  cx cy 0

Both plain YOLO boxes (5 tokens) and polygon lines (converted to their bounding
box) are handled. Frames whose labels contain no training arm are skipped by
default (the arm may still be visible unannotated — unsafe as negatives).

    python scripts/import_phlebotomy_zenodo.py --download          # fetch + import
    python scripts/import_phlebotomy_zenodo.py --src data/phlebotomy-src
    python scripts/import_phlebotomy_zenodo.py --download --max 3000

Images -> data/raw/ prefixed phleb__, labels -> data/labels/.
Then run 02_prepare_dataset.py as usual.
"""
import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
SRC_DEFAULT = DATA / "phlebotomy-src"
DOWNLOADS = DATA / "downloads"

ZENODO_API = "https://zenodo.org/api/records/16924786"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download_and_extract() -> Path:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    print(f"Querying {ZENODO_API} ...")
    with urllib.request.urlopen(ZENODO_API) as r:
        record = json.load(r)

    files = record.get("files", [])
    if not files:
        raise SystemExit("Zenodo record lists no files.")
    entry = max(files, key=lambda f: f.get("size", 0))
    url = entry["links"]["self"]
    dest = DOWNLOADS / entry["key"]

    if dest.exists() and dest.stat().st_size == entry.get("size", -1):
        print(f"Already downloaded: {dest}")
    else:
        print(f"Downloading {entry['key']} ({entry.get('size', 0) / 1e6:.0f} MB) ...")

        def hook(blocks, bs, total):
            done = blocks * bs
            if total > 0 and blocks % 200 == 0:
                print(f"  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="\r")

        urllib.request.urlretrieve(url, dest, reporthook=hook)
        print(f"\nSaved {dest}")

    if SRC_DEFAULT.exists() and any(SRC_DEFAULT.iterdir()):
        print(f"Already extracted: {SRC_DEFAULT}")
    else:
        print(f"Extracting to {SRC_DEFAULT} ...")
        SRC_DEFAULT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest) as z:
            z.extractall(SRC_DEFAULT)
    return SRC_DEFAULT


def find_arm_class_id(src: Path, explicit_id, name_hint: str):
    if explicit_id is not None:
        return explicit_id, f"(--arm-class-id {explicit_id})"
    for yml in sorted(src.rglob("*.yaml")) + sorted(src.rglob("*.yml")):
        try:
            doc = yaml.safe_load(yml.read_text())
        except Exception:
            continue
        names = (doc or {}).get("names")
        if names is None:
            continue
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for idx, name in items:
            if name_hint in str(name).lower().replace("_", " "):
                return int(idx), f"('{name}' in {yml.relative_to(src)})"
    raise SystemExit(
        f"Could not find a class containing '{name_hint}' in any data.yaml under {src}. "
        "Pass --arm-class-id explicitly (check the dataset's data.yaml).")


def line_to_arm_box(tokens, arm_id):
    """Returns (cx, cy, w, h) if this line is an arm instance, else None."""
    try:
        cls = int(float(tokens[0]))
    except ValueError:
        return None
    if cls != arm_id:
        return None
    vals = [float(t) for t in tokens[1:]]
    if len(vals) == 4:                      # plain YOLO box
        return tuple(vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> bounding box
        xs, ys = vals[0::2], vals[1::2]
        x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
        y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
            return None
        return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="extracted dataset root (skips download)")
    ap.add_argument("--download", action="store_true", help="download + extract from Zenodo first")
    ap.add_argument("--arm-class-id", type=int, default=None,
                    help="class index of the training arm (default: auto from data.yaml)")
    ap.add_argument("--arm-class-name", default="arm", help="substring to find the class by name")
    ap.add_argument("--max", type=int, default=0, help="cap imported images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.download:
        src = download_and_extract()
    elif args.src:
        src = Path(args.src)
    elif SRC_DEFAULT.exists():
        src = SRC_DEFAULT
    else:
        raise SystemExit("Pass --download (fetches ~321 MB from Zenodo) or --src <extracted folder>.")
    if not src.exists():
        raise SystemExit(f"{src} not found.")

    arm_id, how = find_arm_class_id(src, args.arm_class_id, args.arm_class_name.lower())
    print(f"Training-arm class id: {arm_id} {how}")

    # Map image stems -> paths (per parent dir to survive duplicate stems across splits).
    image_index = {}
    for img in src.rglob("*"):
        if img.suffix.lower() in IMG_EXTS:
            image_index.setdefault(img.stem, []).append(img)

    label_files = [p for p in src.rglob("*.txt")
                   if p.parent.name != "downloads" and p.stem in image_index]
    random.seed(args.seed)
    random.shuffle(label_files)

    RAW.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)

    imported = boxes = skipped_no_arm = 0
    for lbl in label_files:
        if args.max and imported >= args.max:
            break

        arm_lines = []
        for raw_line in lbl.read_text().splitlines():
            tokens = raw_line.split()
            if not tokens:
                continue
            box = line_to_arm_box(tokens, arm_id)
            if box is None:
                continue
            cx, cy, w, h = box
            arm_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
                             f"{cx:.6f} {cy:.6f} 0 {cx:.6f} {cy:.6f} 0")
        if not arm_lines:
            skipped_no_arm += 1
            continue

        # Prefer the image sitting next to this label's split (labels/ <-> images/).
        candidates = image_index[lbl.stem]
        img = candidates[0]
        for cand in candidates:
            if cand.parent.name == lbl.parent.name or \
               cand.parent.parent == lbl.parent.parent:
                img = cand
                break

        split_tag = lbl.parent.parent.name if lbl.parent.name == "labels" else lbl.parent.name
        name = f"phleb__{split_tag}__{img.stem}"
        shutil.copy2(img, RAW / f"{name}{img.suffix.lower()}")
        (LABELS / f"{name}.txt").write_text("\n".join(arm_lines) + "\n")
        imported += 1
        boxes += len(arm_lines)

    print(f"Imported {imported} image(s), {boxes} training-arm box(es) -> {RAW}")
    print(f"Skipped {skipped_no_arm} frame(s) without an annotated training arm.")
    print("Next: python scripts/02_prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
