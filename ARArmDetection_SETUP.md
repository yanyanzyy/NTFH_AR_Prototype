# AR Arm Detection — Setup

A Meta Quest 3 prototype that detects other people's arms via the Passthrough
Camera API and overlays a red 3D quad on each detected arm. Inference runs
locally with **Unity Sentis** using **YOLOv11n-pose** (~6 MB).

## What's in place

**Code (`Assets/Scripts/ArmDetection/`)**

| Script | Role |
| --- | --- |
| `ArmDetectionManager.cs` | Per-frame orchestrator. Pulls camera frame, runs detector, filters, drives overlay. |
| `PassthroughCameraSource.cs` | Wraps Meta's `WebCamTextureManager`; exposes texture + image↔world projection. |
| `YoloPoseDetector.cs` | Sentis runtime, NMS, COCO-17 keypoint parsing. |
| `WearerHandFilter.cs` | Excludes detections that match the wearer's own wrists (via OVR hand transforms). |
| `ArmOverlay.cs` | Pool of red unlit 3D quads, oriented along shoulder→wrist in world space. |
| `DetectionTypes.cs` | Plain data structs. |

**Manifest changes**

- `Packages/manifest.json` — added `com.unity.sentis: 2.1.2`
- `Assets/Plugins/Android/AndroidManifest.xml` — added `horizonos.permission.HEADSET_CAMERA` and `android.permission.CAMERA`

## One-time setup

### 1. Reopen Unity so Sentis installs

Sentis was added to `manifest.json`. Reopen the project (or hit "Refresh" in
Package Manager) and let Unity import it.

If the version `2.1.2` is unavailable, switch to whatever stable 2.1.x version
shows up in **Window → Package Manager → Unity Sentis**.

### 2. Export the YOLOv11n-pose ONNX

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', opset=15, imgsz=640, dynamic=False, simplify=True)"
```

Drop `yolo11n-pose.onnx` into `Assets/Models/`. Unity will import it as a
Sentis `ModelAsset`.

### 3. Enable the Passthrough Camera API in your Meta SDK

This project already targets the Meta XR SDK v201 which supports the
Passthrough Camera API (PCA). You still need to:

1. **Meta XR → Project Setup Tool** (or Tools menu) — apply the recommended fixes.
2. **`OculusProjectConfig`** (`Assets/Oculus/OculusProjectConfig.asset`):
   - Passthrough Support = **Required**
   - Hand Tracking Support = **Hands Only** or **Controllers + Hands**
3. Drag the **`WebCamTextureManager`** prefab (from Meta XR's
   `PassthroughCameraSamples` package or copy it from the
   `Meta-XR-Samples/PassthroughCameraAPI` GitHub repo) into your scene. It
   needs exactly one in the scene; auto-discovery in
   `PassthroughCameraSource.Awake()` will find it by component name.

### 4. Build the scene

Open `Assets/Scenes/MR_TestScene.unity` (or `Assets/test.unity`). It should
already contain the Meta OVR rig + passthrough.

Add a new GameObject called **`ArmDetectionPrototype`** and attach (in order):

1. `PassthroughCameraSource`
2. `YoloPoseDetector` — drag `Assets/Models/yolo11n-pose.onnx` into **Model Asset**.
3. `WearerHandFilter` — drag the left and right OVR wrist bones (e.g.
   `OVRCameraRigInteraction → ... → b_l_wrist` and `b_r_wrist`) into
   **Wearer Wrist Transforms**.
4. `ArmOverlay`
5. `ArmDetectionManager` — its `Reset()` auto-fills the four references if the
   above scripts are children. Otherwise drag them in by hand.

### 5. Build settings

- Platform: **Android**
- Texture compression: **ASTC**
- XR Plug-in Management → Android → Oculus / OpenXR with Meta XR feature group enabled
- Player Settings → Other Settings → Minimum API Level: **Android 10 (29)** or higher
- Color Space: Linear, Graphics API: Vulkan or OpenGLES3

Build and deploy to Quest 3 / 3S. On first launch the headset will prompt for
the Headset Camera permission — accept it.

## How it works

1. **Camera frame** — `PassthroughCameraSource` reads the left RGB sensor as a
   `WebCamTexture` via Meta's API. ~1280×960 @ 30 fps depending on SDK config.

2. **Inference** — `YoloPoseDetector` blits to 640×640, normalizes to a NCHW
   float tensor, runs YOLOv11n-pose on the GPU compute backend (~20–35 ms on
   Quest 3). Output is `(1, 56, 8400)`. We confidence-threshold,
   sort, and NMS down to a handful of people.

3. **Per-person arm rects** — From COCO-17 keypoints we take
   shoulder→elbow→wrist for each side (keypoints 5/7/9 and 6/8/10) and only
   keep arms whose shoulder and wrist exceed the keypoint confidence floor.

4. **Wearer filter** — Each candidate arm's wrist is compared against the
   image-space projection of the wearer's actual wrist bones (Meta hand
   tracking). If they're within ~120 px, the arm is the wearer's and is
   discarded. A 0.7 m proximity fallback covers the case where hand tracking
   is missing.

5. **Depth estimate** — Person bbox height in pixels and an assumed 1.7 m
   stature give a rough metric depth. Good enough for placing the overlay
   in front of the person.

6. **3D quad** — Shoulder and wrist are back-projected to world space at that
   depth. A red unlit quad is placed at the midpoint, rotated so its long
   axis follows the arm and its forward axis faces the camera.

## Real occlusion (arms in front of the quad)

The current overlay uses an Unlit shader. The Quest 3's real-world hand will
*not* occlude the red quad out of the box because the standard scene depth
buffer doesn't know where the user's hand is in space.

For real depth occlusion, install **Meta XR Depth API** and switch the
overlay's shader to one that samples Meta's environment depth texture (e.g.
`Meta/Depth/Occlusion/URP/Unlit`). The depth-aware shader will reject pixels
where the real world is closer than the virtual quad — making the arm look
like it sits in front of the red rectangle.

Edit `ArmOverlay._preferredShader` to point at that shader once you have it.

## Tuning knobs

- `YoloPoseDetector.ConfidenceThreshold` (default 0.35) — raise to suppress
  false positives at the cost of recall.
- `YoloPoseDetector.NmsIoUThreshold` (default 0.45) — raise to keep more
  overlapping detections.
- `ArmDetectionManager.InferEveryNFrames` (default 2) — at 72 Hz this caps
  inference around 36 Hz. Increase to lower thermals at the cost of lag.
- `ArmDetectionManager.AssumedPersonHeightMeters` (1.7) — adjust for the
  target population.
- `WearerHandFilter.ImageRadiusPixels` (120) — widen if your own arms still
  get overlaid; narrow if other people's arms next to yours get rejected.
- `ArmOverlay.ThicknessRatio` (0.22) — wider/thinner red bar.

## Known limitations

- **No letterboxing.** Camera frames are stretched to 640×640. For 4:3 sensors
  this introduces a small aspect distortion; precision is fine for a prototype.
  Add letterboxing in `YoloPoseDetector` if needed.
- **Depth is a heuristic.** Bbox-height to depth assumes adult stature and
  full-body framing. Children, seated subjects, or partial-body framing will
  place the quad too close or too far. Swap in Meta's depth API later for
  true per-pixel depth.
- **One inference per frame, synchronous readback.** `ReadbackAndClone()`
  blocks until the GPU is done. For a smoother pipeline, switch to
  `ReadbackAndCloneAsync()` and consume results next frame.
- **WebCamTextureManager name match.** Auto-discovery looks for a component
  named `WebCamTextureManager` or `PassthroughCameraManager`. If Meta renames
  it, set the reference manually on `PassthroughCameraSource`.
