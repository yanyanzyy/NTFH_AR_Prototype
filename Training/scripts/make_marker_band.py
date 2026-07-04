"""
Generate the printable ArUco marker band (+ shoulder disc marker) for the
mannequin-arm 6-DoF tracker, and print the matching Unity inspector values.

The band is a strip of DICT_4X4_50 markers that folds into a regular N-gon prism
around the arm: each marker sits on one FLAT facet (curved markers decode badly
and report wrong poses). Fold on the dashed lines, wrap with the labeled edge
toward the SHOULDER, and fix it so it can't rotate on the arm.

    python scripts/make_marker_band.py --arm-circumference-mm 280 --facets 6

Outputs (print at 100% scale — verify with the calibration bar):
    markers/arm_band_4x4.png      the fold-up band
    markers/shoulder_disc.png     one extra marker for the shoulder cross-section

Then copy the printed "UNITY SETTINGS" block into the MarkerArmTracker's
Marker Band Layout in the inspector.
"""
import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "markers"

QUIET_MM = 6          # white quiet zone around each marker (>= 1 marker bit)
EDGE_MM = 12          # strip margin above/below markers (fold stability + label)
CAL_BAR_MM = 100      # calibration bar length


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def marker_image(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-circumference-mm", type=float, default=280.0,
                    help="circumference of the arm where the band sits")
    ap.add_argument("--facets", type=int, default=6, help="flat faces the band folds into")
    ap.add_argument("--marker-mm", type=float, default=38.0, help="marker side length (black border incl.)")
    ap.add_argument("--start-id", type=int, default=0, help="facet 0 marker id (DICT_4X4_50)")
    ap.add_argument("--disc-id", type=int, default=10, help="shoulder disc marker id")
    ap.add_argument("--disc-marker-mm", type=float, default=50.0)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    n = args.facets
    if args.start_id + n > 50 or args.disc_id >= 50:
        raise SystemExit("DICT_4X4_50 has ids 0..49 - lower --start-id/--disc-id.")
    if args.start_id <= args.disc_id < args.start_id + n:
        raise SystemExit("--disc-id collides with the band's id range.")

    # Faceted prism circumscribing the arm cylinder: apothem = arm radius.
    arm_radius_mm = args.arm_circumference_mm / (2 * math.pi)
    facet_mm = 2 * arm_radius_mm * math.tan(math.pi / n)
    apothem_mm = arm_radius_mm

    usable_mm = facet_mm - 2 * QUIET_MM
    if usable_mm < args.marker_mm:
        max_marker = usable_mm
        raise SystemExit(
            f"Facet width {facet_mm:.1f} mm leaves only {max_marker:.1f} mm for a marker "
            f"(need {args.marker_mm:.1f}). Use fewer facets or a smaller --marker-mm.")

    dpi = args.dpi
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # ── Band strip ────────────────────────────────────────────────────────────────
    strip_w = mm_to_px(facet_mm * n, dpi)
    strip_h = mm_to_px(args.marker_mm + 2 * EDGE_MM, dpi)
    cal_h = mm_to_px(18, dpi)
    canvas = np.full((strip_h + cal_h, max(strip_w, mm_to_px(CAL_BAR_MM + 20, dpi))), 255, np.uint8)

    marker_px = mm_to_px(args.marker_mm, dpi)
    top_px = mm_to_px(EDGE_MM, dpi)

    for i in range(n):
        facet_x0 = mm_to_px(facet_mm * i, dpi)
        facet_x1 = mm_to_px(facet_mm * (i + 1), dpi)
        cx = (facet_x0 + facet_x1) // 2

        marker = marker_image(dictionary, args.start_id + i, marker_px)
        x0 = cx - marker_px // 2
        canvas[top_px:top_px + marker_px, x0:x0 + marker_px] = marker

        # Dashed fold line at the right edge of every facet except the last.
        if i < n - 1:
            for y in range(0, strip_h, mm_to_px(4, dpi)):
                y1 = min(y + mm_to_px(2, dpi), strip_h)
                canvas[y:y1, facet_x1 - 1:facet_x1 + 1] = 120

        cv2.putText(canvas, f"id {args.start_id + i}",
                    (facet_x0 + mm_to_px(3, dpi), strip_h - mm_to_px(3, dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.6, 0, 2)

    cv2.putText(canvas, "<< SHOULDER SIDE (this edge toward the shoulder) >>",
                (mm_to_px(3, dpi), mm_to_px(8, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.6, 0, 2)

    # Calibration bar under the strip.
    bar_y = strip_h + cal_h // 2
    bar_x0 = mm_to_px(10, dpi)
    bar_x1 = bar_x0 + mm_to_px(CAL_BAR_MM, dpi)
    canvas[bar_y - 2:bar_y + 2, bar_x0:bar_x1] = 0
    canvas[bar_y - mm_to_px(2, dpi):bar_y + mm_to_px(2, dpi), bar_x0:bar_x0 + 2] = 0
    canvas[bar_y - mm_to_px(2, dpi):bar_y + mm_to_px(2, dpi), bar_x1 - 2:bar_x1] = 0
    cv2.putText(canvas, f"this bar must measure exactly {CAL_BAR_MM} mm when printed",
                (bar_x1 + mm_to_px(4, dpi), bar_y + mm_to_px(1, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, dpi / 300.0 * 0.5, 0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    band_path = OUT_DIR / "arm_band_4x4.png"
    cv2.imwrite(str(band_path), canvas)

    # ── Shoulder disc marker ─────────────────────────────────────────────────────
    disc_px = mm_to_px(args.disc_marker_mm, dpi)
    quiet_px = mm_to_px(QUIET_MM, dpi)
    disc = np.full((disc_px + 2 * quiet_px, disc_px + 2 * quiet_px), 255, np.uint8)
    disc[quiet_px:quiet_px + disc_px, quiet_px:quiet_px + disc_px] = \
        marker_image(dictionary, args.disc_id, disc_px)
    disc_path = OUT_DIR / "shoulder_disc.png"
    cv2.imwrite(str(disc_path), disc)

    print(f"Wrote {band_path}")
    print(f"Wrote {disc_path}")
    print(f"\nPrint BOTH at 100% scale ({dpi} dpi). Verify the {CAL_BAR_MM} mm bar.\n")
    print("=" * 60)
    print("UNITY SETTINGS  (MarkerArmTracker > Marker Band Layout)")
    print("=" * 60)
    print(f"  Facets                = {n}")
    print(f"  Facet Apothem Meters  = {apothem_mm / 1000.0:.4f}")
    print(f"  Marker Size Meters    = {args.marker_mm / 1000.0:.4f}")
    print(f"  First Marker Id       = {args.start_id}")
    print(f"  Extra marker (shoulder disc): Id={args.disc_id}, "
          f"Size={args.disc_marker_mm / 1000.0:.4f}")
    print("    - stick it on the flat shoulder cross-section, set its Local Position")
    print("      to roughly (0, 0, -<band-to-disc distance in m>) and Local Euler to")
    print("      (0, 180, 0); nudge until the overlay agrees from all angles.")
    print(f"\nBand geometry: facet width {facet_mm:.1f} mm, strip length "
          f"{facet_mm * n:.1f} mm, fits circumference {args.arm_circumference_mm:.0f} mm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
