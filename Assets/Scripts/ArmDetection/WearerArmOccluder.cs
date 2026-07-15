using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Creates invisible depth-only sphere proxies at the wearer's tracked arm joints.
    ///
    /// HOW OCCLUSION WORKS
    /// -------------------
    /// Passthrough video is composited by Meta's runtime at a layer below Unity's renderer;
    /// it never writes to Unity's depth buffer. Because of this, any Unity mesh placed in
    /// world space will ordinarily appear in front of passthrough objects regardless of real
    /// depth — including on top of the wearer's own arms.
    ///
    /// To fix this we use a "depth pre-pass trick":
    ///   1. This script renders a sphere at each tracked wrist position using the
    ///      ArmDetection/DepthOccluder shader (RenderQueue = Geometry-10, ColorMask 0).
    ///      The sphere is invisible but writes its clip-space depth to the depth buffer.
    ///   2. The ArmOverlay quad uses ArmDetection/ArmOverlayUnlit (RenderQueue = Geometry+1,
    ///      ZTest LEqual). Where the occluder sphere wrote a smaller depth value (closer to
    ///      camera) the overlay's ZTest fails → overlay pixel discarded → wearer's arm is
    ///      visually in front of the overlay. ✓
    ///   3. The target arm in passthrough has no occluder sphere → overlay renders over it. ✓
    ///
    /// SETUP
    /// -----
    /// Drag the OVRSkeleton components from the Hand Tracking building blocks (left + right)
    /// into _wearerSkeletons. The wrist bone is found automatically at runtime.
    /// Alternatively assign wrist bone Transforms directly to _wearerArmTransforms.
    /// </summary>
    public class WearerArmOccluder : MonoBehaviour
    {
        [Tooltip("World-space transforms of the wearer's arm joints (wrists, elbows, etc).\n" +
                 "Leave empty when using _wearerSkeletons instead.")]
        [SerializeField] private Transform[] _wearerArmTransforms;

        [Tooltip("Drag the OVRSkeleton components from the Hand Tracking building blocks here " +
                 "(one per hand). The wrist bone is located automatically at runtime — " +
                 "no need to navigate the bone list in the Inspector.")]
        [SerializeField] private OVRSkeleton[] _wearerSkeletons;

        [Tooltip("Radius of each hand-bone occluder in metres. Keep this close to wrist size so " +
                 "hands provide depth feedback without cutting a large circle from world UI.")]
        [SerializeField] private float _occluderRadius = 0.055f;

        [Header("Gripping hand")]
        [Tooltip("Which bones on EACH wearer skeleton get a depth occluder. The wrist alone isn't " +
                 "enough for the hand holding the syringe — its fingers reach across the target arm, " +
                 "so the overlay would paint over them. One sphere is created per skeleton per bone.")]
        [SerializeField] private OVRSkeleton.BoneId[] _occludedHandBones =
        {
            OVRSkeleton.BoneId.Hand_WristRoot,
            OVRSkeleton.BoneId.Hand_IndexTip,
            OVRSkeleton.BoneId.Hand_Index1,
            OVRSkeleton.BoneId.Hand_MiddleTip,
        };

        [Header("Needle / syringe")]
        [Tooltip("Punch the held syringe out of the overlay using the vision-detected needle " +
                 "(tip→hub) from the ArmDetectionManager, so the syringe renders in FRONT of the arm.")]
        [SerializeField] private bool _occludeNeedle = true;

        [Tooltip("ArmDetectionManager supplying TryGetNeedle(tip, hub). Required for needle occlusion.")]
        [SerializeField] private ArmDetectionManager _armManager;

        [Tooltip("Number of depth spheres spread along the needle tip→hub axis.")]
        [SerializeField, Range(2, 12)] private int _needleOccluderCount = 6;

        [Tooltip("Radius (m) of each needle occluder sphere. ~3 cm covers a gripped syringe barrel.")]
        [SerializeField] private float _needleOccluderRadius = 0.03f;

        [Tooltip("Push each needle occluder this far toward the headset camera. The needle world " +
                 "position sits on the ARM's depth plane, and equal depth loses the overlay's " +
                 "ZTest LEqual tie; the real syringe is physically in front of the arm, so a few cm " +
                 "of bias makes the occluder reliably win and the overlay gets discarded there.")]
        [SerializeField] private float _needleCameraBiasMeters = 0.04f;

        [Header("Overlay gating")]
        [Tooltip("When set, ALL occluders only run while this overlay's mesh is actually shown " +
                 "(ArmOverlay.IsModelRevealed). With the answer-key overlay hidden during poking " +
                 "there is nothing to sit in front of, so every occluder switches OFF and the view " +
                 "is completely clear; they re-arm automatically during the 5-second reveal. Leave " +
                 "empty to run the occluders unconditionally.")]
        [SerializeField] private ArmOverlay _overlay;

        private GameObject[] _transformOccluders;  // one per _wearerArmTransforms entry
        private GameObject[] _skeletonOccluders;   // _wearerSkeletons.Length * _occludedHandBones.Length
        private GameObject[] _needleOccluders;     // spread along the detected needle axis
        private Material     _depthOnlyMaterial;
        private Transform    _mainCamTransform;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake()
        {
            _depthOnlyMaterial = CreateDepthOnlyMaterial();
            RebuildOccluders();
        }

        private void OnDestroy()
        {
            DestroyOccluders();
            if (_depthOnlyMaterial != null) Destroy(_depthOnlyMaterial);
        }

        private void LateUpdate()
        {
            // LateUpdate so the occluder moves AFTER OVR skeleton updates transforms.

            // Gate: with the answer-key overlay hidden (normal poking) there is no overlay to
            // occlude, so switch every occluder off — the trainee gets a clear, sphere-free view.
            if (_overlay != null && !_overlay.IsModelRevealed)
            {
                DeactivateAll();
                return;
            }

            // 1. Transform-based occluders.
            if (_wearerArmTransforms != null && _transformOccluders != null)
            {
                int count = Mathf.Min(_wearerArmTransforms.Length, _transformOccluders.Length);
                for (int i = 0; i < count; i++)
                {
                    bool valid = _wearerArmTransforms[i] != null;
                    _transformOccluders[i].SetActive(valid);
                    if (valid)
                        _transformOccluders[i].transform.position = _wearerArmTransforms[i].position;
                }
                for (int i = count; i < _transformOccluders.Length; i++)
                    _transformOccluders[i].SetActive(false);
            }

            // 2. Skeleton-based occluders — one sphere per (skeleton × tracked bone), resolved each frame.
            if (_wearerSkeletons != null && _skeletonOccluders != null && _occludedHandBones != null)
            {
                int bonesPerSkel = _occludedHandBones.Length;
                for (int s = 0; s < _wearerSkeletons.Length; s++)
                {
                    bool ready = IsSkeletonReady(_wearerSkeletons[s]);
                    for (int b = 0; b < bonesPerSkel; b++)
                    {
                        int idx = s * bonesPerSkel + b;
                        if (idx >= _skeletonOccluders.Length) break;

                        Vector3? pos = ready
                            ? FindBonePosition(_wearerSkeletons[s], _occludedHandBones[b]) : null;
                        _skeletonOccluders[idx].SetActive(pos.HasValue);
                        if (pos.HasValue)
                            _skeletonOccluders[idx].transform.position = pos.Value;
                    }
                }
            }

            // 3. Needle occluders — spread along the vision-detected syringe, biased toward the camera.
            UpdateNeedleOccluders();
        }

        private void DeactivateAll()
        {
            SetAllInactive(_transformOccluders);
            SetAllInactive(_skeletonOccluders);
            SetAllInactive(_needleOccluders);
        }

        private static void SetAllInactive(GameObject[] arr)
        {
            if (arr == null) return;
            foreach (var o in arr)
                if (o != null && o.activeSelf) o.SetActive(false);
        }

        private void UpdateNeedleOccluders()
        {
            if (_needleOccluders == null || _needleOccluders.Length == 0) return;

            Vector3 tip = default, hub = default;
            bool has = _occludeNeedle && _armManager != null && _armManager.TryGetNeedle(out tip, out hub);
            if (!has)
            {
                foreach (var o in _needleOccluders) if (o != null) o.SetActive(false);
                return;
            }

            if (_mainCamTransform == null && Camera.main != null) _mainCamTransform = Camera.main.transform;
            Vector3 camPos = _mainCamTransform != null ? _mainCamTransform.position : tip;

            int n = _needleOccluders.Length;
            for (int i = 0; i < n; i++)
            {
                float f = n == 1 ? 0f : (float)i / (n - 1);
                Vector3 p = Vector3.Lerp(tip, hub, f);

                // Bias toward the headset camera so the depth-only sphere sits in front of the
                // overlay surface (both are otherwise on the arm's depth plane).
                Vector3 toCam = camPos - p;
                if (toCam.sqrMagnitude > 1e-6f) p += toCam.normalized * _needleCameraBiasMeters;

                _needleOccluders[i].transform.position = p;
                _needleOccluders[i].SetActive(true);
            }
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (!Application.isPlaying) return;
            DestroyOccluders();
            RebuildOccluders();
        }
#endif

        // ── Private helpers ────────────────────────────────────────────────────────────

        private void RebuildOccluders()
        {
            int transformCount = _wearerArmTransforms?.Length ?? 0;
            int skeletonCount  = _wearerSkeletons?.Length     ?? 0;
            int bonesPerSkel   = _occludedHandBones?.Length   ?? 0;

            _transformOccluders = new GameObject[transformCount];
            _skeletonOccluders  = new GameObject[skeletonCount * bonesPerSkel];

            for (int i = 0; i < transformCount; i++)
                _transformOccluders[i] = CreateOccluderSphere($"TransformOccluder_{i}", _occluderRadius);

            for (int i = 0; i < _skeletonOccluders.Length; i++)
                _skeletonOccluders[i]  = CreateOccluderSphere($"SkeletonOccluder_{i}", _occluderRadius);

            int needleCount = _occludeNeedle ? Mathf.Max(2, _needleOccluderCount) : 0;
            _needleOccluders = new GameObject[needleCount];
            for (int i = 0; i < needleCount; i++)
                _needleOccluders[i] = CreateOccluderSphere($"NeedleOccluder_{i}", _needleOccluderRadius);
        }

        private GameObject CreateOccluderSphere(string goName, float radius)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = goName;
            go.transform.SetParent(transform, false);
            go.transform.localScale = Vector3.one * (radius * 2f);

            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);

            var mr = go.GetComponent<MeshRenderer>();
            mr.sharedMaterial       = _depthOnlyMaterial;
            mr.shadowCastingMode    = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = UnityEngine.Rendering.LightProbeUsage.Off;
            mr.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;

            go.SetActive(false);
            return go;
        }

        private void DestroyOccluders()
        {
            if (_transformOccluders != null)
            {
                foreach (var o in _transformOccluders) if (o != null) Destroy(o);
                _transformOccluders = null;
            }
            if (_skeletonOccluders != null)
            {
                foreach (var o in _skeletonOccluders) if (o != null) Destroy(o);
                _skeletonOccluders = null;
            }
            if (_needleOccluders != null)
            {
                foreach (var o in _needleOccluders) if (o != null) Destroy(o);
                _needleOccluders = null;
            }
        }

        private static Material CreateDepthOnlyMaterial()
        {
            var shader = Shader.Find("ArmDetection/DepthOccluder");
            if (shader != null) return new Material(shader);

            Debug.LogError("[WearerArmOccluder] Shader 'ArmDetection/DepthOccluder' not found. " +
                           "Ensure DepthOccluder.shader is inside the Assets folder.");
            return new Material(Shader.Find("Universal Render Pipeline/Unlit")
                                ?? Shader.Find("Unlit/Color"));
        }

        // ── Static helpers ─────────────────────────────────────────────────────────────

        private static bool IsSkeletonReady(OVRSkeleton skel)
            => skel != null && skel.IsInitialized && skel.Bones != null && skel.Bones.Count > 0;

        private static Vector3? FindBonePosition(OVRSkeleton skel, OVRSkeleton.BoneId boneId)
        {
            foreach (var bone in skel.Bones)
            {
                if (bone == null || bone.Transform == null) continue;
                if (bone.Id == boneId)
                    return bone.Transform.position;
            }
            return null;
        }
    }
}
