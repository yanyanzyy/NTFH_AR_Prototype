# AR Arm Detection — Setup

A Meta Quest 3 venipuncture-training prototype. The headset's passthrough
camera feeds a **custom single-class YOLO11n-pose model** (2 keypoints:
proximal/elbow + distal/wrist) that detects the **mannequin training arm**
(Limbs & Things Advanced Venipuncture Arm). A vein-mapped 3D arm model is
overlaid on the detection, and a staged injection evaluator (contact → vein
spot → angle → depth) critiques blood-draw attempts, narrated by a floating
facilitator chat panel.

Inference runs on-device with **Unity Inference Engine**
(`com.unity.ai.inference`, the successor of Sentis). Training the model is a
separate pipeline — see [Training/README.md](Training/README.md).

> Historical note: the project started as a COCO-17 person-pose prototype
> (`YoloPoseDetector.cs`, red quad overlays). That code is gone; this doc
> describes the current arm-only system.

## Project facts

- **Unity**: 6000.4.7f1 (Unity 6.4)
- **Packages** (`Packages/manifest.json`): `com.unity.ai.inference 2.6.1`,
  Meta XR SDK Core + MRUK + Platform `201.0.0`, OpenXR `1.16.1`,
  XR Interaction Toolkit `3.4.1`
- **Working scene**: `Assets/Scenes/ArmDetectionScene.unity` — the only scene
  in Build Settings. (`MR_TestScene.unity` is the Meta-configured base it was
  created from; `SyringePoseDetectionScene*.unity` are Group 2's syringe test
  scenes.)
- **Android manifest** (`Assets/Plugins/Android/AndroidManifest.xml`) already
  declares: `horizonos.permission.HEADSET_CAMERA`, `android.permission.CAMERA`,
  `com.oculus.permission.HAND_TRACKING`, `com.oculus.permission.USE_ANCHOR_API`,
  `com.oculus.permission.USE_SCENE`, and the passthrough/hand-tracking features.

## Code map

**`Assets/Scripts/ArmDetection/`** — the main pipeline

| Script | Role |
| --- | --- |
| `ArmDetectionManager.cs` | Orchestrator. Detector selection, depth estimation, target lock/refine/freeze, spatial anchoring, needle routing. |
| `PassthroughCameraSource.cs` | Camera texture + calibrated image↔world projection. Prefers Meta's `PassthroughCameraAccess` (MRUK 201+, real intrinsics + timestamp-locked pose); falls back to `WebCamTextureManager` / raw `WebCamTexture` / editor webcam. |
| `CustomArmDetector.cs` | Inference Engine runtime for the arm-pose ONNX. 320×320 input, GPUCompute backend, FP16 weights, inference sliced across frames (`Layers Per Frame`), async GPU readback. Parses `[1,11,N]` (and legacy 2-class layouts). |
| `MediaPipeHandArmDetector.cs` + `MediaPipeHomulerBridge.cs` | Optional fallback: infers an arm from MediaPipe hand landmarks when the custom detector is starved. Lifecycle auto-managed (paused while the custom detector delivers). |
| `WearerHandFilter.cs` / `WearerArmOccluder.cs` | Reject the wearer's own arm from detections / depth-mask it so it renders in front of the overlay. |
| `ArmOverlay.cs` | Poses the 3D arm prefab on the detection. Preferred: `OverlayAnchor_Proximal` / `OverlayAnchor_Distal` child transforms in the prefab land exactly on the two detected keypoints. Falls back to a debug quad if no prefab. |
| `ArmLockButton.cs` / `ArmOverlayControlPanel.cs` / `DetectionModeButton.cs` | Runtime UI: RE-DETECT ARM (unlock, also controller **B**), overlay on/off, status panel. |
| `InjectionSiteDetector.cs` | Needle-tip vs arm-surface contact + dwell events. Falls back to the OVR **right-hand index fingertip** when no vision needle is available. |
| `InjectionSequenceEvaluator.cs` | Staged assessment: contact → vein spot → insertion angle → depth. |
| `VeinMap.cs` / `VeinPathVisualizer.cs` / `VeinFeedbackController.cs` / `VeinFeedbackUI.cs` / `VeinTrainerHUD.cs` | Prefab-authored vein paths, on-device line rendering, feedback + trainer summary HUD. |
| `NeedleDetector.cs` / `NeedleAngleEstimator.cs` / `NeedleVisualizer.cs` | Legacy vision-needle path (SyringePose model). Used only when no `CustomSyringeDetector` is assigned. |
| `SimulatedNeedleProvider.cs` | Test rig: tracked fingertip (or pen in pinch grip) stands in for the syringe. While active it **overrides** the vision needle (the syringe model is currently unreliable). |
| `ArmBoundingBoxDebug.cs` / `ArmDetectionDebugHUD.cs` | Debug visuals for the selected detection + pipeline stats. |
| `TrainingFrameCapture.cs` | Saves passthrough frames as PNGs for training-data capture (see below). |

**`Assets/Scripts/SyringePose/`** — Group 2's syringe-pose engine
(`CustomSyringeDetector` on the `SyringePosePrototype` object, 640×640,
4 keypoints). When assigned on the manager it is the **preferred** vision
needle source.

**`Assets/Scripts/ArmTracking/`** — ArUco marker-band 6-DoF arm tracker
(alternative/supplement to the vision model). Own doc:
[MARKER_TRACKING_SETUP.md](Assets/Scripts/ArmTracking/MARKER_TRACKING_SETUP.md).

**`Assets/Scripts/Facilitator/`** — facilitator chat popup. **No scene setup
needed**: `FacilitatorModeBootstrap` auto-creates it at runtime whenever an
`ArmDetectionManager` or `CustomSyringeDetector` is in the scene. It requires
`Assets/Resources/Facilitator/VenepunctureProcedure.asset` (present).

## The model

The detector consumes an arm-pose ONNX produced by the training pipeline
(`Training/scripts/07_export.py` → `Assets/Models/arm-pose-320.onnx`).

To swap models: select **CustomArmDetector** (child of
`ArmDetectionPrototype` in `ArmDetectionScene`), drag the ONNX into **Model
Asset**, keep **Input Size = 320**. `Assets/Models/` holds several per-run and
per-person exports — only the assigned one is used at runtime. Leave
`SyringePose*.onnx` alone (they belong to the syringe pipeline).

## Scene setup (already done — for reference / rebuilds)

`ArmDetectionScene.unity` already contains the OVR rig, the
`ArmDetectionPrototype` hierarchy (CameraSource, CustomArmDetector,
MediaPipeHandDetector, WearerFilter, WearerOccluder, ArmOverlay,
DepthRaycaster, BoundingBoxDebug, ModeButton, debug HUD), the `VeinSystem`
object, `VeinTrainerHUD`, and Group 2's `SyringePosePrototype`.

To rebuild or extend a scene, use the editor menu **Tools → AR Arm
Detection**:

- **Create Scene From MR_TestScene** / **Add Prototype to Open Scene** —
  builds and wires the `ArmDetectionPrototype` hierarchy automatically
  (`CustomArmDetector` must then be added and assigned by hand).
- **Add / Remove Bounding Box Debug**, **Add Arm Unlock Button**
- **Add Needle Detector** — legacy vision-needle pipeline (detector + angle
  estimator + visualizer).
- **Add Injection Sequence Evaluator**, **Add Vein Path Visualizer**
- **Add Vein Test Rig (Finger As Syringe)** — wires the
  `SimulatedNeedleProvider` + angle estimator + trainer HUD so the right
  index fingertip drives the whole injection assessment.
- **Add MediaPipe Hand Detector**

Each action is idempotent (safe to re-run; re-wires instead of duplicating).

## Build settings (current)

- Platform: **Android**; scene list: `ArmDetectionScene` only
- Min SDK **32**, target SDK **34**
- Color space: **Linear**; Graphics APIs: **Vulkan, OpenGLES3**
- XR Plug-in Management → Android → OpenXR with the Meta XR feature group

On first launch the headset prompts for the **Headset Camera** permission —
accept it, or the camera source (and everything downstream) stays black.

## How it works

1. **Camera frame** — `PassthroughCameraSource` provides the passthrough
   texture plus per-frame camera intrinsics/pose for accurate image→world
   projection.
2. **Inference** — `CustomArmDetector` letterbox-blits to 320×320 and runs the
   arm-pose model on GPUCompute with FP16 weights. The model's layers are
   spread across frames (`Layers Per Frame`, default ~14) and the output is
   read back **asynchronously**, so the main thread never blocks on the GPU.
3. **Fallback** — if the custom detector is starved for ~2 s, the MediaPipe
   hand-landmark fallback is started; it pauses again once custom detections
   resume.
4. **Depth** — the detected box is placed in the world via Meta's
   `EnvironmentRaycastManager` (Depth API) with an arm-length heuristic
   (~0.5 m forearm) as sanity bound/fallback.
5. **Target lock** — after N consistent frames inside an acquire radius, the
   manager locks the target, runs a short multi-viewpoint refinement, then
   **freezes** the overlay and pins it with a **spatial anchor**. While
   locked, other detections are ignored; unlock via RE-DETECT ARM /
   controller B. Detection is suspended while frozen to free the GPU.
6. **Overlay** — `ArmOverlay` poses `Assets/Prefabs/3DScanArm_Veins.prefab`
   so its two authored anchors land on the detected proximal/distal points.
   `WearerArmOccluder` (depth-write-only, Geometry−10) + the overlay's
   `ArmOverlayUnlit` shader (Geometry+1) make the wearer's real arm render in
   front of the overlay without the Depth-API occlusion shaders.
7. **Injection assessment** — the needle tip comes from (in priority order)
   the simulated fingertip provider, the `CustomSyringeDetector`, the legacy
   `NeedleDetector`, or the OVR right-index fingertip. `InjectionSiteDetector`
   + `InjectionSequenceEvaluator` score contact → vein spot → angle → depth
   against the prefab-authored `VeinMap`, feeding the trainer HUD and the
   facilitator chat.

## Capturing training data on-device

1. Add `TrainingFrameCapture` to `ArmDetectionPrototype` and drag the
   `PassthroughCameraSource` into its slot.
2. Build, then slowly move around the mannequin arm (it captures ~1 frame/s).
3. Pull frames to the PC and continue in the training pipeline:

```bash
adb pull /sdcard/Android/data/<your.package.name>/files/ArmCaptures Training/data/raw
```

4. Remove/disable the component when not capturing.

## Tuning

Every serialized field on the components carries an Inspector tooltip — that
is the authoritative reference. The ones that matter most:

- `CustomArmDetector` → **Confidence Threshold** (box score gate) and
  **Layers Per Frame** (lower = smoother frame rate, slower detection).
- `ArmDetectionManager` → **Keypoint Confidence must stay at 0.** The
  arm-pose models supervise keypoints on only ~13% of training images, so the
  visibility head under-reports even when positions are accurate; any real
  threshold silently rejects every detection. Quality is gated by box score.
- `ArmDetectionManager` → the **Target lock** block (acquire frames/radius,
  gate radius, refine time, freeze/anchor behavior).
- `ArmOverlay` → anchor names, axis/pivot fallback, lateral offset.
- `SimulatedNeedleProvider` → which finger, pen-tip offset;
  `ArmDetectionManager._simulatedNeedleOverridesVision` to restore
  vision-first needle priority once the syringe model is reliable.

## Known limitations

- **Keypoint visibility is untrustworthy** (see Tuning) — gate on box score.
- **The syringe vision model is unreliable**; the fingertip test rig is the
  default needle source (`_simulatedNeedleOverridesVision = true`).
- **Pose accuracy is data-bound** — current model (`arm_pose_v7`) has ~19 px
  median keypoint error; more capture variety, not architecture, is the lever
  (see Training/README.md).
- **Depth heuristics** assume the fixed ~0.5 m mannequin forearm; other props
  need `_assumedArmLengthMeters` adjusted.
