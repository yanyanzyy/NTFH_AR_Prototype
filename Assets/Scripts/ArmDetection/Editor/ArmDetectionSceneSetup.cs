using System.IO;
using System.Linq;
using Meta.XR;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace ARArmDetection.EditorTools
{
    public static class ArmDetectionSceneSetup
    {
        private const string NewScenePath = "Assets/Scenes/ArmDetectionScene.unity";

        [MenuItem("Tools/AR Arm Detection/Create Scene From MR_TestScene")]
        public static void CreateSceneFromMRScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

            string basePath = FindBaseScene();
            if (string.IsNullOrEmpty(basePath))
            {
                EditorUtility.DisplayDialog(
                    "Base scene not found",
                    "Could not find MR_TestScene.unity. Open your Meta-configured scene manually, " +
                    "then run 'Tools > AR Arm Detection > Add Prototype to Open Scene' instead.",
                    "OK");
                return;
            }

            var scene = EditorSceneManager.OpenScene(basePath, OpenSceneMode.Single);
            var root  = AddPrototypeHierarchy();

            Directory.CreateDirectory(Path.GetDirectoryName(NewScenePath)!);
            if (!EditorSceneManager.SaveScene(scene, NewScenePath))
            {
                Debug.LogError("[ArmDetection] Failed to save scene to " + NewScenePath);
                return;
            }
            AddSceneToBuildSettings(NewScenePath);

            Selection.activeGameObject = root;
            EditorGUIUtility.PingObject(root);
            Debug.Log("[ArmDetection] Created scene at " + NewScenePath);
        }

        [MenuItem("Tools/AR Arm Detection/Add Prototype to Open Scene")]
        public static void AddToOpenScene()
        {
            var active = EditorSceneManager.GetActiveScene();
            if (!active.IsValid())
            {
                Debug.LogError("[ArmDetection] No active scene. Open a scene first.");
                return;
            }
            var root = AddPrototypeHierarchy();
            Selection.activeGameObject = root;
            EditorGUIUtility.PingObject(root);
            EditorSceneManager.MarkSceneDirty(active);
        }

        /// <summary>
        /// Removes the per-detection bounding-box visualiser from the open scene's
        /// ArmDetectionPrototype. Use this when the floating coloured rectangles
        /// clutter the view while you're testing detection.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Remove Bounding Box Debug")]
        public static void RemoveBoundingBoxDebug()
        {
            var debugs = Object.FindObjectsByType<ArmBoundingBoxDebug>(FindObjectsSortMode.None);
            if (debugs.Length == 0)
            {
                Debug.Log("[ArmDetection] No ArmBoundingBoxDebug found in scene.");
                return;
            }
            foreach (var d in debugs)
            {
                if (d == null) continue;
                Debug.Log($"[ArmDetection] Removing ArmBoundingBoxDebug from '{d.gameObject.name}'.");
                // If the GO only exists to host this component (the default 'BoundingBoxDebug' child),
                // remove the whole GO; otherwise just strip the component.
                if (d.gameObject.name == "BoundingBoxDebug" && d.gameObject.GetComponents<Component>().Length <= 2)
                    Undo.DestroyObjectImmediate(d.gameObject);
                else
                    Undo.DestroyObjectImmediate(d);
            }
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        /// <summary>
        /// Re-adds the bounding-box visualiser to the open scene under ArmDetectionPrototype.
        /// Wires its _manager and _cameraSource references automatically.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Add Bounding Box Debug")]
        public static void AddBoundingBoxDebug()
        {
            var prototype = GameObject.Find("ArmDetectionPrototype");
            if (prototype == null)
            {
                Debug.LogError("[ArmDetection] ArmDetectionPrototype not found. Add the prototype first.");
                return;
            }
            if (prototype.GetComponentInChildren<ArmBoundingBoxDebug>() != null)
            {
                Debug.Log("[ArmDetection] ArmBoundingBoxDebug already present.");
                return;
            }
            var manager      = prototype.GetComponent<ArmDetectionManager>();
            var cameraSource = prototype.GetComponentInChildren<PassthroughCameraSource>();

            var go    = new GameObject("BoundingBoxDebug");
            go.transform.SetParent(prototype.transform, false);
            var debug = go.AddComponent<ArmBoundingBoxDebug>();
            Undo.RegisterCreatedObjectUndo(go, "Add BoundingBoxDebug");

            if (manager != null)      WireReference(debug, "_manager",      manager);
            if (cameraSource != null) WireReference(debug, "_cameraSource", cameraSource);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = go;
        }

        /// <summary>
        /// Adds the UNLOCK ARM button to an existing ArmDetectionPrototype scene.
        /// Use this on scenes created before the target-lock feature existed.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Add Arm Unlock Button")]
        public static void AddArmUnlockButton()
        {
            var prototype = GameObject.Find("ArmDetectionPrototype");
            if (prototype == null)
            {
                Debug.LogError("[ArmDetection] ArmDetectionPrototype not found. Add the prototype first.");
                return;
            }
            if (prototype.GetComponentInChildren<ArmLockButton>() != null)
            {
                Debug.Log("[ArmDetection] ArmLockButton already present.");
                return;
            }

            var manager = prototype.GetComponent<ArmDetectionManager>();
            var go = new GameObject("ArmLockButton");
            go.transform.SetParent(prototype.transform, false);
            var lockButton = go.AddComponent<ArmLockButton>();
            Undo.RegisterCreatedObjectUndo(go, "Add Arm Unlock Button");

            if (manager != null) WireReference(lockButton, "_manager", manager);
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = go;
            Debug.Log("[ArmDetection] Added ArmLockButton (UNLOCK ARM panel + controller B shortcut).");
        }

        private const string NeedleModelPath = "Assets/Models/best_theothergroup.onnx";

        /// <summary>
        /// Adds (or fixes) the vision needle pipeline in an existing ArmDetectionPrototype
        /// scene: the NeedleDetector (SyringePose model from the Jin-Rui branch, single
        /// class, 4 keypoints), the NeedleAngleEstimator (insertion angle vs horizontal),
        /// and the NeedleVisualizer (tip/hub markers + angle-coloured line). Safe to run
        /// again: existing components are re-wired and their model/layout config updated,
        /// not duplicated.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Add Needle Detector")]
        public static void AddNeedleDetector()
        {
            var prototype = GameObject.Find("ArmDetectionPrototype");
            if (prototype == null)
            {
                Debug.LogError("[ArmDetection] ArmDetectionPrototype not found. Add the prototype first.");
                return;
            }

            var manager = prototype.GetComponent<ArmDetectionManager>();

            // ── Detector (create or update) ─────────────────────────────────────────
            var detector = prototype.GetComponentInChildren<NeedleDetector>();
            if (detector == null)
            {
                var detectorGO = new GameObject("NeedleDetector");
                detectorGO.transform.SetParent(prototype.transform, false);
                detector = detectorGO.AddComponent<NeedleDetector>();
                Undo.RegisterCreatedObjectUndo(detectorGO, "Add Needle Detector");
            }

            var model = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(NeedleModelPath);
            if (model != null)
                WireReference(detector, "_modelAsset", model);
            else
                Debug.LogWarning($"[ArmDetection] {NeedleModelPath} not found — assign your needle ONNX " +
                                 "to the NeedleDetector's Model Asset slot manually.");

            // SyringePose layout: 640 input, single class, needle class 0. Group 2 ran it at 0.15.
            WireInt(detector, "_inputSize", 640);
            WireInt(detector, "_numClasses", 1);
            WireInt(detector, "_needleClassId", 0);
            WireFloat(detector, "_confidenceThreshold", 0.15f);

            if (manager != null) WireReference(manager, "_needleDetector", detector);

            // ── Angle estimator + visualizer (ported from the Jin-Rui branch) ───────
            var estimator = prototype.GetComponentInChildren<NeedleAngleEstimator>();
            if (estimator == null)
            {
                var estimatorGO = new GameObject("NeedleAngleEstimator");
                estimatorGO.transform.SetParent(prototype.transform, false);
                estimator = estimatorGO.AddComponent<NeedleAngleEstimator>();
                Undo.RegisterCreatedObjectUndo(estimatorGO, "Add Needle Angle Estimator");
            }
            if (manager != null) WireReference(estimator, "_armManager", manager);

            var visualizer = prototype.GetComponentInChildren<NeedleVisualizer>();
            if (visualizer == null)
            {
                var visualizerGO = new GameObject("NeedleVisualizer");
                visualizerGO.transform.SetParent(prototype.transform, false);
                visualizer = visualizerGO.AddComponent<NeedleVisualizer>();
                Undo.RegisterCreatedObjectUndo(visualizerGO, "Add Needle Visualizer");
            }
            if (manager != null) WireReference(visualizer, "_manager", manager);
            WireReference(visualizer, "_detector", detector);
            WireReference(visualizer, "_angleEstimator", estimator);

            var hud = prototype.GetComponentInChildren<ArmDetectionDebugHUD>();
            if (hud != null) WireReference(hud, "_angleEstimator", estimator);

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = detector.gameObject;
            Debug.Log("[ArmDetection] Needle pipeline wired: NeedleDetector (SyringePose model) + " +
                      "NeedleAngleEstimator + NeedleVisualizer.");
        }

        /// <summary>
        /// Adds (or re-wires) the InjectionSequenceEvaluator — the staged
        /// contact → vein spot → angle → depth assessment — on the VeinSystem object,
        /// next to the VeinMap it queries. Safe to run again.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Add Injection Sequence Evaluator")]
        public static void AddInjectionSequenceEvaluator()
        {
            var veinMap = Object.FindFirstObjectByType<VeinMap>();
            if (veinMap == null)
            {
                Debug.LogError("[ArmDetection] No VeinMap found in the scene. Set up the VeinSystem " +
                               "object (VeinMap + InjectionSiteDetector + feedback components) first.");
                return;
            }

            var host = veinMap.gameObject;
            var evaluator = host.GetComponent<InjectionSequenceEvaluator>();
            if (evaluator == null)
            {
                evaluator = Undo.AddComponent<InjectionSequenceEvaluator>(host);
            }

            var manager = Object.FindFirstObjectByType<ArmDetectionManager>();
            var angleEstimator = Object.FindFirstObjectByType<NeedleAngleEstimator>();

            if (manager != null) WireReference(evaluator, "_armManager", manager);
            else Debug.LogWarning("[ArmDetection] No ArmDetectionManager in scene — wire _armManager manually.");
            WireReference(evaluator, "_veinMap", veinMap);

            // VeinMap needs the overlay to resolve prefab-authored vein paths (Option A).
            var overlay = Object.FindFirstObjectByType<ArmOverlay>();
            if (overlay != null) WireReference(veinMap, "_overlay", overlay);
            if (angleEstimator != null) WireReference(evaluator, "_angleEstimator", angleEstimator);
            else Debug.LogWarning("[ArmDetection] No NeedleAngleEstimator in scene — run " +
                                  "'Add Needle Detector' first, then re-run this.");

            var hud = Object.FindFirstObjectByType<ArmDetectionDebugHUD>();
            if (hud != null) WireReference(hud, "_injectionEvaluator", evaluator);

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = host;
            Debug.Log("[ArmDetection] InjectionSequenceEvaluator wired on '" + host.name + "' " +
                      "(contact -> vein spot -> angle -> depth).");
        }

        /// <summary>
        /// Adds the runtime vein path visualizer (LineRenderers in the headset) so the
        /// prefab-authored vein waypoints can be verified on device. VeinMap's gizmos are
        /// editor-only. Safe to run again; also (re)wires VeinMap._overlay.
        /// </summary>
        [MenuItem("Tools/AR Arm Detection/Add Vein Path Visualizer")]
        public static void AddVeinPathVisualizer()
        {
            var veinMap = Object.FindFirstObjectByType<VeinMap>();
            if (veinMap == null)
            {
                Debug.LogError("[ArmDetection] No VeinMap found in the scene. Set up the VeinSystem object first.");
                return;
            }

            var host = veinMap.gameObject;
            var visualizer = host.GetComponent<VeinPathVisualizer>();
            if (visualizer == null) visualizer = Undo.AddComponent<VeinPathVisualizer>(host);
            WireReference(visualizer, "_veinMap", veinMap);

            var overlay = Object.FindFirstObjectByType<ArmOverlay>();
            if (overlay != null) WireReference(veinMap, "_overlay", overlay);
            else Debug.LogWarning("[ArmDetection] No ArmOverlay found — prefab vein paths won't resolve.");

            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = host;
            Debug.Log("[ArmDetection] VeinPathVisualizer added — vein paths render as coloured lines once the arm locks.");
        }

        [MenuItem("Tools/AR Arm Detection/Add MediaPipe Hand Detector")]
        public static void AddMediaPipeHandDetector()
        {
            var prototype = GameObject.Find("ArmDetectionPrototype");
            if (prototype == null)
            {
                Debug.LogError("[ArmDetection] ArmDetectionPrototype not found. Add the prototype first.");
                return;
            }

            var manager = prototype.GetComponent<ArmDetectionManager>();
            if (manager == null)
            {
                Debug.LogError("[ArmDetection] ArmDetectionManager not found on ArmDetectionPrototype.");
                return;
            }

            var detector = prototype.GetComponentInChildren<MediaPipeHandArmDetector>();
            GameObject go;
            if (detector == null)
            {
                go = new GameObject("MediaPipeHandDetector");
                go.transform.SetParent(prototype.transform, false);
                detector = go.AddComponent<MediaPipeHandArmDetector>();
                Undo.RegisterCreatedObjectUndo(go, "Add MediaPipe Hand Detector");
            }
            else
            {
                go = detector.gameObject;
            }

            var bridge = go.GetComponent<MediaPipeHomulerBridge>() ?? go.AddComponent<MediaPipeHomulerBridge>();
            var cameraSource = prototype.GetComponentInChildren<PassthroughCameraSource>();

            WireReference(manager, "_mediaPipeDetector", detector);
            if (cameraSource != null) WireReference(detector, "_cameraSource", cameraSource);
            WireReference(bridge, "_target", detector);
            var hud = prototype.GetComponentInChildren<ArmDetectionDebugHUD>();
            if (hud != null) WireReference(hud, "_mediaPipeDetector", detector);

            Debug.Log("[ArmDetection] Added MediaPipeHandDetector. After importing MediaPipeUnityPlugin, " +
                      "drag its HandLandmarkerRunner into MediaPipeHomulerBridge._handLandmarkerRunner.");
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
            Selection.activeGameObject = go;
        }

        // ── Private helpers ────────────────────────────────────────────────────────────

        private static string FindBaseScene()
        {
            string[] candidates =
            {
                "Assets/Scenes/MR_TestScene.unity",
                "Assets/MR_TestScene.unity",
            };
            foreach (var p in candidates)
                if (File.Exists(p)) return p;

            var guids = AssetDatabase.FindAssets("MR_TestScene t:Scene");
            if (guids.Length > 0) return AssetDatabase.GUIDToAssetPath(guids[0]);
            return null;
        }

        private static GameObject AddPrototypeHierarchy()
        {
            // Only skip if already fully set up (has the manager component).
            var existing = GameObject.Find("ArmDetectionPrototype");
            if (existing != null && existing.GetComponent<ArmDetectionManager>() != null)
            {
                Debug.LogWarning("[ArmDetection] ArmDetectionPrototype already fully set up. Skipping.");
                return existing;
            }

            GameObject root;
            if (existing != null)
            {
                Debug.Log("[ArmDetection] Found empty ArmDetectionPrototype — populating it.");
                root = existing;
            }
            else
            {
                root = new GameObject("ArmDetectionPrototype");
                Undo.RegisterCreatedObjectUndo(root, "Create ArmDetectionPrototype");
            }

            // ── Child components ────────────────────────────────────────────────────
            var cameraSourceGO = CreateChild(root, "CameraSource");
            var cameraSource   = cameraSourceGO.AddComponent<PassthroughCameraSource>();

            var mediaPipeGO = CreateChild(root, "MediaPipeHandDetector");
            var mediaPipeDetector = mediaPipeGO.AddComponent<MediaPipeHandArmDetector>();
            var mediaPipeBridge = mediaPipeGO.AddComponent<MediaPipeHomulerBridge>();

            var filterGO = CreateChild(root, "WearerFilter");
            var filter   = filterGO.AddComponent<WearerHandFilter>();

            var occluderGO = CreateChild(root, "WearerOccluder");
            var occluder   = occluderGO.AddComponent<WearerArmOccluder>();

            var overlayGO = CreateChild(root, "Overlay");
            var overlay   = overlayGO.AddComponent<ArmOverlay>();

            var needleGO = CreateChild(root, "NeedleDetector");
            var needleDetector = needleGO.AddComponent<NeedleDetector>();
            var needleModel = AssetDatabase.LoadAssetAtPath<Unity.InferenceEngine.ModelAsset>(NeedleModelPath);
            if (needleModel != null) WireReference(needleDetector, "_modelAsset", needleModel);

            var modeButtonGO = CreateChild(root, "ModeButton");
            var modeButton   = modeButtonGO.AddComponent<DetectionModeButton>();

            var lockButtonGO = CreateChild(root, "ArmLockButton");
            var lockButton   = lockButtonGO.AddComponent<ArmLockButton>();

            // ArmBoundingBoxDebug only draws geometry for the
            // *selected* arm (the one the manager picked and ArmOverlay renders).
            // While searching, no boxes appear — see ArmBoundingBoxDebug.cs.
            var bboxDebugGO = CreateChild(root, "BoundingBoxDebug");
            var bboxDebug   = bboxDebugGO.AddComponent<ArmBoundingBoxDebug>();

            // ── Depth-API raycaster (Meta Depth API for accurate arm placement) ─────
            // Adds Meta.XR.EnvironmentRaycastManager as a sibling of CameraSource so the
            // ArmDetectionManager can raycast viewport rays against real-world geometry.
            var depthRaycasterGO = CreateChild(root, "DepthRaycaster");
            var depthRaycaster   = depthRaycasterGO.AddComponent<EnvironmentRaycastManager>();

            // ── Root manager ────────────────────────────────────────────────────────
            var manager = root.AddComponent<ArmDetectionManager>();

            WireReference(manager,    "_cameraSource",  cameraSource);
            WireReference(manager,    "_needleDetector", needleDetector);
            WireReference(manager,    "_mediaPipeDetector", mediaPipeDetector);
            WireReference(manager,    "_wearerFilter",  filter);
            WireReference(manager,    "_overlay",       overlay);
            WireReference(manager,    "_depthRaycaster", depthRaycaster);
            WireReference(mediaPipeDetector, "_cameraSource", cameraSource);
            WireReference(mediaPipeBridge, "_target", mediaPipeDetector);
            WireReference(modeButton, "_manager",       manager);
            WireReference(lockButton, "_manager",       manager);
            WireReference(bboxDebug,  "_manager",       manager);
            WireReference(bboxDebug,  "_cameraSource",  cameraSource);

            // ── Auto-assign from scene objects ──────────────────────────────────────
            TryAutoAssignWebCamManager(cameraSource);
            TryAutoAssignWristTransforms(filter, occluder);
            TryAutoAssignCameraTransform(cameraSource);

            EditorUtility.SetDirty(root);
            return root;
        }

        private static GameObject CreateChild(GameObject parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            return go;
        }

        private static void WireReference(Object owner, string propertyName, Object value)
        {
            var so   = new SerializedObject(owner);
            var prop = so.FindProperty(propertyName);
            if (prop == null)
            {
                Debug.LogWarning($"[ArmDetection] Property '{propertyName}' not found on {owner.GetType().Name}");
                return;
            }
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void WireInt(Object owner, string propertyName, int value)
        {
            var so   = new SerializedObject(owner);
            var prop = so.FindProperty(propertyName);
            if (prop == null)
            {
                Debug.LogWarning($"[ArmDetection] Property '{propertyName}' not found on {owner.GetType().Name}");
                return;
            }
            prop.intValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void WireFloat(Object owner, string propertyName, float value)
        {
            var so   = new SerializedObject(owner);
            var prop = so.FindProperty(propertyName);
            if (prop == null)
            {
                Debug.LogWarning($"[ArmDetection] Property '{propertyName}' not found on {owner.GetType().Name}");
                return;
            }
            prop.floatValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void TryAutoAssignWebCamManager(PassthroughCameraSource source)
        {
            // Priority 1: Meta.XR.PassthroughCameraAccess (MRUK 201+, the official PCA API).
            // This is what the [BuildingBlock] Passthrough Camera Access drops into the scene.
            // Priority 2: legacy WebCamTextureManager / PassthroughCameraManager (older SDKs).
            MonoBehaviour pca = null, legacy = null;
            foreach (var mb in Object.FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None))
            {
                if (mb == null) continue;
                var typeName = mb.GetType().Name;
                if (typeName == "PassthroughCameraAccess" && pca == null) pca = mb;
                else if ((typeName == "WebCamTextureManager" || typeName == "PassthroughCameraManager")
                         && legacy == null) legacy = mb;
            }

            var chosen = pca ?? legacy;
            if (chosen != null)
            {
                WireReference(source, "_webCamTextureManager", chosen);
                Debug.Log($"[ArmDetection] Auto-assigned camera manager: {chosen.GetType().Name} on '{chosen.gameObject.name}'");
                return;
            }

            Debug.LogWarning("[ArmDetection] No PassthroughCameraAccess / WebCamTextureManager found. " +
                             "Add the '[BuildingBlock] Passthrough Camera Access' prefab to the scene " +
                             "(or assign your camera manager manually to CameraSource._webCamTextureManager).");
        }

        /// <summary>
        /// Finds wrist bone transforms by name and assigns them to both
        /// WearerHandFilter (for wearer-arm rejection) and WearerArmOccluder
        /// (for depth-buffer occlusion).
        /// </summary>
        private static void TryAutoAssignWristTransforms(WearerHandFilter filter,
                                                         WearerArmOccluder occluder)
        {
            Transform left = null, right = null;
            foreach (var t in Object.FindObjectsByType<Transform>(FindObjectsSortMode.None))
            {
                var n = t.name.ToLowerInvariant();
                bool isLeft  = left  == null &&
                    (n == "b_l_wrist" || n == "wrist_l" || n == "leftwrist"  ||
                     n == "left_wrist"  || n.EndsWith("_l_wrist"));
                bool isRight = right == null &&
                    (n == "b_r_wrist" || n == "wrist_r" || n == "rightwrist" ||
                     n == "right_wrist" || n.EndsWith("_r_wrist"));
                if (isLeft)  left  = t;
                if (isRight) right = t;
                if (left != null && right != null) break;
            }

            if (left == null && right == null) return;

            var transforms = new[] { left, right }.Where(t => t != null).ToArray();

            // Assign to WearerHandFilter
            {
                var so   = new SerializedObject(filter);
                var prop = so.FindProperty("_wearerWristTransforms");
                if (prop != null)
                {
                    prop.arraySize = transforms.Length;
                    for (int i = 0; i < transforms.Length; i++)
                        prop.GetArrayElementAtIndex(i).objectReferenceValue = transforms[i];
                    so.ApplyModifiedPropertiesWithoutUndo();
                }
            }

            // Assign the same transforms to WearerArmOccluder
            {
                var so   = new SerializedObject(occluder);
                var prop = so.FindProperty("_wearerArmTransforms");
                if (prop != null)
                {
                    prop.arraySize = transforms.Length;
                    for (int i = 0; i < transforms.Length; i++)
                        prop.GetArrayElementAtIndex(i).objectReferenceValue = transforms[i];
                    so.ApplyModifiedPropertiesWithoutUndo();
                }
            }

            Debug.Log($"[ArmDetection] Auto-assigned wrist transforms: " +
                      $"L={left?.name ?? "none"}  R={right?.name ?? "none"}");
        }

        private static void TryAutoAssignCameraTransform(PassthroughCameraSource source)
        {
            if (Camera.main != null)
                WireReference(source, "_cameraReferenceTransform", Camera.main.transform);
        }

        private static void AddSceneToBuildSettings(string scenePath)
        {
            var scenes = EditorBuildSettings.scenes.ToList();
            if (scenes.Any(s => s.path == scenePath)) return;
            scenes.Add(new EditorBuildSettingsScene(scenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
        }
    }
}
