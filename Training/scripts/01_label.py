"""
Step 1 - Label keypoints for the arm + needle POSE model.

This is a pose (keypoint) pipeline, NOT plain detection. Each instance has a
box AND 2 keypoints. The keypoint MEANING depends on the class:

  class 0 = arm    -> kpt0 = proximal (near elbow / insertion zone), kpt1 = distal (wrist)
  class 1 = needle -> kpt0 = tip (the contact point),               kpt1 = hub (back of needle)

Two keypoints per class is deliberate: YOLO-pose locks every instance to the
same keypoint count, and 2-each fits perfectly. tip->hub gives the needle axis
(insertion angle); proximal->distal gives the forearm axis for the AR overlay.

WORKFLOW
  1. Drop images into Training/data/raw/
  2. python scripts/01_label.py
  3. For each image:
       press  a  then click PROXIMAL, then DISTAL   -> adds one arm
       press  n  then click TIP,      then HUB      -> adds one needle
       (repeat to add more instances of either class)
       u = undo last instance      r = redo current image from scratch
       SPACE / s = save + next      x = save as background (no objects) + next
       b = previous image           q = save + quit
  4. Labels are written to Training/data/labels/<name>.txt (Ultralytics pose
     format) and a preview to Training/data/preview/<name>.jpg.

Resumable: images that already have a label file are skipped unless you pass
--relabel. Background frames (empty label) are also remembered as "done".
"""
import argparse
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RAW = DATA / "raw"
LABELS = DATA / "labels"
PREVIEW = DATA / "preview"

CLASSES = ["arm", "needle"]
# Per-class keypoint names, in label order (kpt0, kpt1).
KPT_NAMES = {
    0: ("proximal", "distal"),
    1: ("tip", "hub"),
}
CLASS_COLOR = {0: (0, 200, 0), 1: (0, 128, 255)}  # BGR: arm=green, needle=orange
BOX_PAD_FRAC = 0.10  # pad the box this fraction of its diagonal around the 2 kpts
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class Instance:
    __slots__ = ("cls", "p0", "p1")

    def __init__(self, cls, p0, p1):
        self.cls = cls
        self.p0 = p0  # (x, y) pixels
        self.p1 = p1


def list_images():
    return sorted(p for p in RAW.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path(img: Path) -> Path:
    return LABELS / f"{img.stem}.txt"


def box_from_points(p0, p1, w, h):
    xs = [p0[0], p1[0]]
    ys = [p0[1], p1[1]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    diag = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5
    pad = max(6.0, diag * BOX_PAD_FRAC)
    x_min = max(0.0, x_min - pad)
    y_min = max(0.0, y_min - pad)
    x_max = min(w - 1.0, x_max + pad)
    y_max = min(h - 1.0, y_max + pad)
    cx = (x_min + x_max) * 0.5 / w
    cy = (y_min + y_max) * 0.5 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return cx, cy, bw, bh


def write_label(img: Path, instances, w, h):
    lines = []
    for inst in instances:
        cx, cy, bw, bh = box_from_points(inst.p0, inst.p1, w, h)
        # Ultralytics pose: cls cx cy w h  px0 py0 v0  px1 py1 v1   (all normalized, v=2 visible)
        px0, py0 = inst.p0[0] / w, inst.p0[1] / h
        px1, py1 = inst.p1[0] / w, inst.p1[1] / h
        lines.append(
            f"{inst.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{px0:.6f} {py0:.6f} 2 {px1:.6f} {py1:.6f} 2"
        )
    label_path(img).write_text("\n".join(lines) + ("\n" if lines else ""))


def write_preview(img_bgr, instances, dest: Path):
    vis = img_bgr.copy()
    for inst in instances:
        color = CLASS_COLOR[inst.cls]
        p0 = (int(inst.p0[0]), int(inst.p0[1]))
        p1 = (int(inst.p1[0]), int(inst.p1[1]))
        cv2.line(vis, p0, p1, color, 2)
        cv2.circle(vis, p0, 6, color, -1)          # kpt0 filled
        cv2.circle(vis, p1, 6, color, 2)           # kpt1 ring
        n0, n1 = KPT_NAMES[inst.cls]
        cv2.putText(vis, n0, (p0[0] + 8, p0[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(vis, n1, (p1[0] + 8, p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(str(dest), vis)


def draw_hud(canvas, img_name, idx, total, mode_cls, pending, instances):
    h = canvas.shape[0]
    bar = canvas.copy()
    cv2.rectangle(bar, (0, h - 64), (canvas.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, canvas, 0.45, 0, canvas)

    counts = {0: 0, 1: 0}
    for inst in instances:
        counts[inst.cls] += 1
    mode_txt = "-" if mode_cls is None else CLASSES[mode_cls]
    nxt = ""
    if mode_cls is not None:
        names = KPT_NAMES[mode_cls]
        nxt = f" click {names[len(pending)]}" if len(pending) < 2 else ""
    cv2.putText(canvas, f"[{idx + 1}/{total}] {img_name}", (8, h - 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas,
                f"mode={mode_txt}{nxt}   arms={counts[0]} needles={counts[1]}",
                (8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas,
                "a=arm n=needle u=undo r=reset SPACE=save x=bg b=back q=quit",
                (8, h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)


def render(img_bgr, instances, mode_cls, pending, img_name, idx, total):
    canvas = img_bgr.copy()
    for inst in instances:
        color = CLASS_COLOR[inst.cls]
        p0 = (int(inst.p0[0]), int(inst.p0[1]))
        p1 = (int(inst.p1[0]), int(inst.p1[1]))
        cv2.line(canvas, p0, p1, color, 2)
        cv2.circle(canvas, p0, 6, color, -1)
        cv2.circle(canvas, p1, 6, color, 2)
    if mode_cls is not None and pending:
        cv2.circle(canvas, (int(pending[0][0]), int(pending[0][1])), 6, CLASS_COLOR[mode_cls], -1)
    draw_hud(canvas, img_name, idx, total, mode_cls, pending, instances)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel", action="store_true", help="re-open images that already have labels")
    ap.add_argument("--max-width", type=int, default=1280, help="downscale display if wider than this")
    args = ap.parse_args()

    images = list_images()
    if not images:
        print(f"No images in {RAW}. Drop .jpg/.png files there first.")
        return 1

    if not args.relabel:
        todo = [im for im in images if not label_path(im).exists()]
        if not todo:
            print("All images already labeled. Use --relabel to edit them.")
            return 0
        images = todo

    print(f"Labeling {len(images)} image(s). a=arm n=needle, click 2 points each. SPACE=save.")

    state = {"mode_cls": None, "pending": [], "instances": []}

    def on_mouse(event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN or state["mode_cls"] is None:
            return
        state["pending"].append((float(x) / state["scale"], float(y) / state["scale"]))
        if len(state["pending"]) == 2:
            state["instances"].append(Instance(state["mode_cls"], state["pending"][0], state["pending"][1]))
            state["pending"] = []

    win = "label  (arm + needle keypoints)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    idx = 0
    while 0 <= idx < len(images):
        img = images[idx]
        img_bgr = cv2.imread(str(img))
        if img_bgr is None:
            print(f"  skip unreadable {img.name}")
            idx += 1
            continue

        h, w = img_bgr.shape[:2]
        scale = min(1.0, args.max_width / float(w))
        disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr.copy()
        state.update(mode_cls=None, pending=[], instances=[], scale=scale)

        while True:
            frame = render(disp, state["instances"], state["mode_cls"], state["pending"],
                           img.name, idx, len(images))
            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("a"):
                state["mode_cls"], state["pending"] = 0, []
            elif key == ord("n"):
                state["mode_cls"], state["pending"] = 1, []
            elif key == ord("u"):
                if state["pending"]:
                    state["pending"] = []
                elif state["instances"]:
                    state["instances"].pop()
            elif key == ord("r"):
                state.update(mode_cls=None, pending=[], instances=[])
            elif key in (ord(" "), ord("s")):
                write_label(img, state["instances"], w, h)
                write_preview(img_bgr, state["instances"], PREVIEW / f"{img.stem}.jpg")
                idx += 1
                break
            elif key == ord("x"):
                write_label(img, [], w, h)  # empty = background negative
                idx += 1
                break
            elif key == ord("b"):
                idx = max(0, idx - 1)
                break
            elif key == ord("q"):
                write_label(img, state["instances"], w, h)
                write_preview(img_bgr, state["instances"], PREVIEW / f"{img.stem}.jpg")
                cv2.destroyAllWindows()
                print("Quit. Progress saved.")
                return 0

    cv2.destroyAllWindows()
    print(f"Done. Labels in {LABELS}, previews in {PREVIEW}. Next: python scripts/02_prepare_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
