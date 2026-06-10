"""Click-to-label tool for mannequin-arm keypoints.

Labels images in Training/dataset/raw/ with 3 keypoints (shoulder, elbow, wrist)
in YOLO-pose format using the standard COCO 17-keypoint schema, so the exported
model stays drop-in compatible with the Unity detector (CocoKeypoint indices).

Controls
--------
  Left-click   place next keypoint (1=shoulder, 2=elbow, 3=wrist)
  u            undo last point
  f            flip arm side (LEFT <-> RIGHT keypoint slots; default RIGHT)
  SPACE/ENTER  save label and go to next image (needs all 3 points)
  x            save EMPTY label = background/negative image (no arm visible)
  k            skip image (no label written)
  b            go back one image
  q / ESC      quit

Run:  python 01_label_arm.py            (labels only unlabeled images)
      python 01_label_arm.py --relabel  (revisit already-labeled images too)
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]          # Training/
RAW_DIR = ROOT / "dataset" / "raw"
LABEL_DIR = ROOT / "dataset" / "raw_labels"
PREVIEW_DIR = ROOT / "dataset" / "preview"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_VIEW_W, MAX_VIEW_H = 1500, 900

# COCO keypoint slot indices (17-keypoint schema).
SLOTS = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}
POINT_NAMES = ["shoulder", "elbow", "wrist"]
POINT_COLORS = [(0, 255, 255), (255, 255, 0), (0, 0, 255)]  # BGR: yellow, cyan, red
BBOX_PAD_FRAC = 0.15  # padding around keypoints as a fraction of arm pixel length


class LabelSession:
    def __init__(self):
        self.points = []      # [(x, y)] in ORIGINAL image pixels, up to 3
        self.side = "right"   # the Limbs & Things venipuncture arm is a right arm

    def reset(self):
        self.points.clear()


def yolo_label_line(points, side, img_w, img_h):
    """Builds one YOLO-pose label line: class cx cy w h  (x y v) * 17."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    arm_len = max(((xs[0] - xs[2]) ** 2 + (ys[0] - ys[2]) ** 2) ** 0.5, 20.0)
    pad = arm_len * BBOX_PAD_FRAC

    x1 = max(0.0, min(xs) - pad)
    y1 = max(0.0, min(ys) - pad)
    x2 = min(img_w - 1.0, max(xs) + pad)
    y2 = min(img_h - 1.0, max(ys) + pad)

    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h

    kps = [0.0] * (17 * 3)
    for name, (px, py) in zip(POINT_NAMES, points):
        slot = SLOTS[side][name]
        kps[slot * 3 + 0] = px / img_w
        kps[slot * 3 + 1] = py / img_h
        kps[slot * 3 + 2] = 2.0  # visible

    vals = [0, cx, cy, bw, bh] + kps
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def draw_overlay(img, session, scale, msg):
    view = img.copy()
    for i, (px, py) in enumerate(session.points):
        p = (int(px * scale), int(py * scale))
        cv2.circle(view, p, 7, POINT_COLORS[i], -1)
        cv2.putText(view, POINT_NAMES[i], (p[0] + 10, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, POINT_COLORS[i], 2)
        if i > 0:
            q = (int(session.points[i - 1][0] * scale), int(session.points[i - 1][1] * scale))
            cv2.line(view, q, p, (255, 255, 255), 2)

    next_pt = POINT_NAMES[len(session.points)] if len(session.points) < 3 else "DONE - SPACE to save"
    bar = (f"side={session.side.upper()}  next={next_pt}  |  "
           "u=undo f=side x=background k=skip b=back q=quit")
    cv2.rectangle(view, (0, 0), (view.shape[1], 28), (32, 32, 32), -1)
    cv2.putText(view, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    if msg:
        cv2.putText(view, msg, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
    return view


def save_preview(img, session, name):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    prev = draw_overlay(img, session, 1.0, "")
    cv2.imwrite(str(PREVIEW_DIR / f"{name}.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel", action="store_true",
                    help="also revisit images that already have labels")
    args = ap.parse_args()

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in RAW_DIR.iterdir()
                    if p.suffix.lower() in IMG_EXTS) if RAW_DIR.exists() else []
    if not images:
        sys.exit(f"No images found in {RAW_DIR}\n"
                 "Capture frames on the headset (TrainingFrameCapture) and pull them here.")

    if not args.relabel:
        images = [p for p in images if not (LABEL_DIR / f"{p.stem}.txt").exists()]
        if not images:
            sys.exit("All images already labeled. Use --relabel to revisit them.")

    print(f"{len(images)} image(s) to label.")
    session = LabelSession()
    idx = 0
    msg = ""

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    state = {"scale": 1.0}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(session.points) < 3:
            s = state["scale"]
            session.points.append((x / s, y / s))

    cv2.setMouseCallback("label", on_mouse)

    while 0 <= idx < len(images):
        path = images[idx]
        img = cv2.imread(str(path))
        if img is None:
            print(f"Unreadable image, skipping: {path.name}")
            idx += 1
            continue

        h, w = img.shape[:2]
        scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h, 1.0)
        state["scale"] = scale
        disp = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

        while True:
            view = draw_overlay(disp, session, scale, msg)
            cv2.setWindowTitle("label", f"[{idx + 1}/{len(images)}] {path.name}")
            cv2.imshow("label", view)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord("u") and session.points:
                session.points.pop()
            elif key == ord("f"):
                session.side = "left" if session.side == "right" else "right"
            elif key == ord("x"):
                (LABEL_DIR / f"{path.stem}.txt").write_text("")
                msg = f"saved BACKGROUND: {path.name}"
                session.reset()
                idx += 1
                break
            elif key in (ord("k"), ord("n")):
                session.reset()
                idx += 1
                msg = ""
                break
            elif key == ord("b"):
                session.reset()
                idx = max(0, idx - 1)
                msg = ""
                break
            elif key in (32, 13):  # SPACE / ENTER
                if len(session.points) != 3:
                    msg = "need all 3 points (shoulder, elbow, wrist)"
                    continue
                line = yolo_label_line(session.points, session.side, w, h)
                (LABEL_DIR / f"{path.stem}.txt").write_text(line + "\n")
                save_preview(img, session, path.stem)
                msg = f"saved: {path.stem}.txt"
                session.reset()
                idx += 1
                break

    cv2.destroyAllWindows()
    n_labels = len(list(LABEL_DIR.glob("*.txt")))
    print(f"Done. {n_labels} label file(s) in {LABEL_DIR}")
    print("Check the drawn keypoints in dataset/preview/, then run 02_prepare_dataset.py")


if __name__ == "__main__":
    main()
