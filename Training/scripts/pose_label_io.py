"""
Shared label I/O for the offline augmentation steps (07, 08).

The dataset's on-disk format is the 11-column YOLO-pose line:

    class  cx cy bw bh  k0x k0y v0  k1x k1y v1        (all normalized 0..1)

kpt0 = proximal (near elbow), kpt1 = distal (wrist).

CONVENTION: a keypoint that leaves the frame is written back at visibility 0,
NOT clamped to the border. A wrist that is out of frame has no true position,
and inventing one on the edge would teach the model to park keypoints on frame
edges. Both augmentation scripts rely on this.
"""
import cv2
import numpy as np


def parse_label(text, w, h):
    """11-column normalized lines -> [(cls, box_xyxy_px, [(x_px, y_px, v), ...])].

    Pass w=h=1 to keep normalized units.
    """
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) != 11:
            continue
        cls = int(float(p[0]))
        v = list(map(float, p[1:]))
        cx, cy, bw, bh = v[0] * w, v[1] * h, v[2] * w, v[3] * h
        box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
        kpts = [(v[4] * w, v[5] * h, int(v[6])),
                (v[7] * w, v[8] * h, int(v[9]))]
        out.append((cls, box, kpts))
    return out


def format_label(instances, w, h):
    """Inverse of parse_label. Invisible keypoints are written as '0 0 0'."""
    lines = []
    for cls, box, kpts in instances:
        cx = (box[0] + box[2]) / 2 / w
        cy = (box[1] + box[3]) / 2 / h
        bw = (box[2] - box[0]) / w
        bh = (box[3] - box[1]) / h
        parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for kx, ky, kv in kpts:
            if kv == 0:
                parts += ["0.000000", "0.000000", "0"]
            else:
                parts += [f"{min(max(kx / w, 0.0), 1.0):.6f}",
                          f"{min(max(ky / h, 0.0), 1.0):.6f}", str(kv)]
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def has_keypoints(instance):
    """True if any keypoint on this instance is visible."""
    return any(k[2] for k in instance[2])


def plan_crop(img_w, img_h, box, occupancy, jitter, rng):
    """Crop rectangle that makes `box` cover `occupancy` of the frame.

    Keeps the source aspect ratio, offsets the centre by up to `jitter` of the
    crop size so the arm is not always centred, and returns None when the
    required crop would not fit inside the image.
    """
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    aspect = img_w / img_h
    crop_h = np.sqrt(box_area / occupancy / aspect)
    crop_w = crop_h * aspect
    if crop_w > img_w or crop_h > img_h:
        return None                      # already closer than the target

    cx = (box[0] + box[2]) / 2 + rng.uniform(-jitter, jitter) * crop_w
    cy = (box[1] + box[3]) / 2 + rng.uniform(-jitter, jitter) * crop_h
    x1 = float(np.clip(cx - crop_w / 2, 0, img_w - crop_w))
    y1 = float(np.clip(cy - crop_h / 2, 0, img_h - crop_h))
    return x1, y1, x1 + crop_w, y1 + crop_h


def sharpness(img_bgr, at=320):
    """Laplacian variance at the training resolution.

    Synthetic close-ups are softer than real ones because cropping in reveals
    the camera's true detail at that magnification while downscaling a full
    frame averages 4x4 blocks and sharpens. If that softness correlates with
    arm size, the model can use it as a shortcut for scale and then misjudge
    real (sharp) close-ups - the suspected arm_pose_v5 regression. Both
    augmentation scripts report it so the gap stays visible.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cv2.resize(g, (at, at), interpolation=cv2.INTER_AREA),
                               cv2.CV_64F).var())


def montage(samples, out_path, cols=4, tile=(426, 320)):
    """Grid preview. samples: [(image_bgr, instances)]. Draws box + keypoints.

    kpt0 (proximal) is drawn amber, kpt1 (distal/wrist) red, with a line
    between them so a wrong axis direction is obvious at a glance.
    """
    tiles = []
    for im, insts in samples:
        t = im.copy()
        for _, box, kpts in insts:
            cv2.rectangle(t, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (60, 200, 60), 2)
            vis = [(int(kx), int(ky)) for kx, ky, kv in kpts if kv]
            if len(vis) == 2:
                cv2.line(t, vis[0], vis[1], (200, 200, 60), 2)
            for i, (kx, ky, kv) in enumerate(kpts):
                if kv:
                    cv2.circle(t, (int(kx), int(ky)), 7,
                               (40, 170, 240) if i == 0 else (40, 40, 230), -1)
        tiles.append(cv2.resize(t, tile))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    rows = [r for r in rows if r.shape[1] == cols * tile[0]]
    if rows:
        cv2.imwrite(str(out_path), np.vstack(rows))
