# Marker-based arm tracking — setup

Tracks the mannequin arm's full 6-DoF pose (position + rotation incl. roll) from
an ArUco marker band, so the 3D arm model follows the physical arm as it is
moved and turned. The YOLO detector remains as fallback + detection UX.

## 1. Print the marker band

```bash
cd Training
source .venv/bin/activate
python scripts/make_marker_band.py --arm-circumference-mm 280 --facets 6
```

Measure the mannequin's circumference where the band will sit (upper-arm end,
away from the venipuncture zone) and pass it in. The script writes
`Training/markers/arm_band_4x4.png` (+ a shoulder-disc marker) and prints the
exact **Facet Apothem** value to enter in Unity. Print at **100% scale** (the
sheet has a calibration bar to verify), fold at the dashed lines so each facet
is FLAT, wrap with the labeled edge toward the shoulder, and fix it so it
cannot rotate on the arm.

## 2. Install the ArUco backend (one-time)

1. Import **OpenCV for Unity** (Enox Software) from the Asset Store.
2. Project Settings → Player → Scripting Define Symbols → add `OPENCV_FOR_UNITY`.

Everything compiles without it; the tracker just reports
"provider not ready" until the define is set.

## 3. Scene wiring

On a new GameObject (e.g. `ArmMarkerTracking`):

- **ArUcoCornerProvider** — leave defaults (Downscale 1).
- **MarkerArmTracker** — assign the `PassthroughCameraSource`; fill the
  *Marker Band Layout* with the values `make_marker_band.py` printed
  (Facets, Facet Apothem Meters, Marker Size Meters, First Marker Id).
- **TrackedArmModel** — assign the tracker, the `ArmDetectionManager`
  (fallback), and your 3D arm model root.

## 4. Calibrate the model offset (once)

Enter Play mode with the band tracked. Move/rotate the 3D model until it visually
covers the physical arm, then run the **"Capture Model Offset From Current
Transform"** context-menu item on TrackedArmModel and copy the component values
out of Play mode (right-click component → Copy Component → Paste Component Values).

## 5. Tuning

| Symptom | Fix |
|---|---|
| Overlay vertically mirrored / pose nonsense | Toggle **Flip Y** on ArUcoCornerProvider |
| Overlay orbits the wrong way when the arm rotates | Toggle **Reverse Wrap Direction** on the band layout |
| Jitter at rest | Lower One-Euro **Min Cutoff** (e.g. 0.8); freeze already removes most of it |
| Lag during motion | Raise One-Euro **Beta** (e.g. 0.15) |
| Solves rejected (HUD shows `rms=` high) | Verify printed size = configured size, band facets are flat, circumference/apothem correct |
| CPU cost too high | Set provider **Downscale** to 2, or tracker **Detect Every N Frames** to 2 |

Marker detection costs a few ms of CPU per frame and nothing on the GPU, so it
coexists with the YOLO detector without touching the frame rate strategy.
