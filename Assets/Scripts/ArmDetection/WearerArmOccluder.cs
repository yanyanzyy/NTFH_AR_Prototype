using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Creates invisible depth-only proxies over the wearer's tracked hands (and the
    /// detected needle) so they punch a correctly-SHAPED hole in the arm overlay.
    ///
    /// HOW OCCLUSION WORKS
    /// -------------------
    /// Passthrough video is composited by Meta's runtime at a layer below Unity's renderer;
    /// it never writes to Unity's depth buffer. Because of this, any Unity mesh placed in
    /// world space will ordinarily appear in front of passthrough objects regardless of real
    /// depth — including on top of the wearer's own hands.
    ///
    /// To fix this we use a "depth pre-pass trick":
    ///   1. This script renders depth-only geometry over each tracked hand using the
    ///      ArmDetection/DepthOccluder shader (RenderQueue = Geometry-10, ColorMask 0).
    ///      The geometry is invisible but writes its clip-space depth to the depth buffer.
    ///   2. The ArmOverlay quad uses ArmDetection/ArmOverlayUnlit (RenderQueue = Geometry+1,
    ///      ZTest LEqual). Where the occluder wrote a smaller depth value (closer to camera)
    ///      the overlay's ZTest fails → overlay pixel discarded → wearer's hand is
    ///      visually in front of the overlay. ✓
    ///   3. The target arm in passthrough has no occluder → overlay renders over it. ✓
    ///
    /// SHAPE (why capsules, not spheres)
    /// ---------------------------------
    /// The old implementation dropped a discrete SPHERE on a handful of bones, which cut
    /// many separate visible CIRCLES out of the overlay wherever the hand crossed it. Now
    /// one thin CAPSULE is stretched along every bone segment of the tracked skeleton
    /// (finger phalanges thin, palm-spanning segments thick), so the punched-out hole
    /// follows the actual silhouette of the hand — one continuous hand-shaped cutout
    /// instead of a cluster of circles. The needle likewise uses a single capsule along
    /// its tip→hub axis instead of a row of spheres.
    ///
    /// SETUP
    /// -----
    /// Drag the OVRSkeleton components from the Hand Tracking building blocks (left + right)
    /// into _wearerSkeletons. Bone segments are discovered automatically at runtime (works
    /// with both legacy Hand_* and OpenXR XRHand_* skeletons — no bone ids to configure).
    /// Alternatively assign wrist bone Transforms directly to _wearerArmTransforms.
    /// </summary>
    public class WearerArmOccluder : MonoBehaviour
    {
        [Tooltip("World-space transforms of the wearer's arm joints (wrists, elbows, etc).\n" +
                 "Each gets a simple sphere occluder. Leave empty when using _wearerSkeletons.")]
        [SerializeField] private Transform[] _wearerArmTransforms;

        [Tooltip("Drag the OVRSkeleton components from the Hand Tracking building blocks here " +
                 "(one per hand). A capsule occluder is stretched along every bone segment " +
                 "automatically — no bone list to maintain.")]
        [SerializeField] private OVRSkeleton[] _wearerSkeletons;

        [Tooltip("Radius (m) of the sphere occluders used for _wearerArmTransforms entries.")]
        [SerializeField] private float _occluderRadius = 0.055f;

        [Header("Hand capsule shape")]
        [Tooltip("Radius (m) of the capsule around each FINGER phalanx segment.")]
        [SerializeField] private float _fingerRadiusMeters = 0.011f;
        [Tooltip("Radius (m) of the capsules spanning the PALM (wrist→knuckle segments). " +
                 "Together the five thick palm capsules fill the palm silhouette.")]
        [SerializeField] private float _palmRadiusMeters = 0.024f;

        [Header("Needle / syringe")]
        [Tooltip("Punch the held syringe out of the overlay using the needle (tip→hub) from the " +
                 "ArmDetectionManager — vision-detected OR simulated — so it renders in FRONT of the arm.")]
        [SerializeField] private bool _occludeNeedle = true;

        [Tooltip("ArmDetectionManager supplying TryGetNeedle(tip, hub). Required for needle occlusion.")]
        [SerializeField] private ArmDetectionManager _armManager;

        [Tooltip("Radius (m) of the needle occluder capsule. ~3 cm covers a gripped syringe barrel.")]
        [SerializeField] private float _needleOccluderRadius = 0.03f;

        [Tooltip("Extends the needle capsule this far (m) past the hub end, covering the syringe " +
                 "barrel/plunger behind the detected hub keypoint.")]
        [SerializeField] private float _needleHubOverhangMeters = 0.05f;

        [Tooltip("Push the needle occluder this far toward the headset camera. The needle world " +
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

        /// <summary>One capsule stretched between two bone transforms each frame.</summary>
        private struct BoneSegment
        {
            public Transform Parent;
            public Transform Child;
            public GameObject Capsule;
            public float Radius;
        }

        private GameObject[] _transformOccluders;          // one sphere per _wearerArmTransforms entry
        private List<BoneSegment>[] _skeletonSegments;     // per skeleton, built lazily once initialized
        private GameObject _needleOccluder;                // single capsule along the needle axis
        private Material   _depthOnlyMaterial;
        private Transform  _mainCamTransform;

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
            // LateUpdate so the occluders move AFTER OVR skeleton updates transforms.

            // Gate: with the answer-key overlay hidden (normal poking) there is no overlay to
            // occlude, so switch every occluder off — the trainee gets a clear view.
            if (_overlay != null && !_overlay.IsModelRevealed)
            {
                DeactivateAll();
                return;
            }

            // 1. Transform-based sphere occluders.
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

            // 2. Skeleton capsule chains — built lazily (bones only exist once the skeleton
            //    initializes at runtime), then just repositioned every frame.
            if (_wearerSkeletons != null && _skeletonSegments != null)
            {
                for (int s = 0; s < _wearerSkeletons.Length; s++)
                {
                    var skel = _wearerSkeletons[s];
                    bool ready = IsSkeletonReady(skel);

                    if (ready && _skeletonSegments[s] == null)
                        _skeletonSegments[s] = BuildSegmentsForSkeleton(skel);

                    var segments = _skeletonSegments[s];
                    if (segments == null) continue;

                    // A skeleton that lost tracking reports IsDataValid false via its bones
                    // still being driven; hide the chain when the skeleton stops being ready.
                    foreach (var seg in segments)
                        UpdateSegmentCapsule(seg, ready);
                }
            }

            // 3. Needle occluder — one capsule along the needle axis, biased toward the camera.
            UpdateNeedleOccluder();
        }

        private void DeactivateAll()
        {
            SetAllInactive(_transformOccluders);
            if (_skeletonSegments != null)
                foreach (var segments in _skeletonSegments)
                    if (segments != null)
                        foreach (var seg in segments)
                            if (seg.Capsule != null && seg.Capsule.activeSelf) seg.Capsule.SetActive(false);
            if (_needleOccluder != null && _needleOccluder.activeSelf) _needleOccluder.SetActive(false);
        }

        private static void SetAllInactive(GameObject[] arr)
        {
            if (arr == null) return;
            foreach (var o in arr)
                if (o != null && o.activeSelf) o.SetActive(false);
        }

        // ── Hand capsule chain ─────────────────────────────────────────────────────────

        /// <summary>
        /// Walks the skeleton's bone list and creates one capsule per parent→child segment.
        /// Works for legacy and XR skeletons alike because it uses the rig's own
        /// ParentBoneIndex topology instead of hard-coded bone ids.
        /// </summary>
        private List<BoneSegment> BuildSegmentsForSkeleton(OVRSkeleton skel)
        {
            var segments = new List<BoneSegment>();
            var bones = skel.Bones;
            var type = skel.GetSkeletonType();

            for (int i = 0; i < bones.Count; i++)
            {
                var bone = bones[i];
                if (bone == null || bone.Transform == null) continue;

                int parentIdx = bone.ParentBoneIndex;
                if (parentIdx < 0 || parentIdx >= bones.Count) continue;
                var parent = bones[parentIdx];
                if (parent == null || parent.Transform == null || parent.Transform == bone.Transform) continue;

                float radius = IsPalmSegment(type, bone.Id) ? _palmRadiusMeters : _fingerRadiusMeters;

                var capsule = CreateOccluderPrimitive(PrimitiveType.Capsule,
                    $"HandOccluder_{skel.name}_{bone.Id}");
                segments.Add(new BoneSegment
                {
                    Parent = parent.Transform,
                    Child = bone.Transform,
                    Capsule = capsule,
                    Radius = radius,
                });
            }

            Debug.Log($"[WearerArmOccluder] Built {segments.Count} capsule segments for " +
                      $"'{skel.name}' ({type}).");
            return segments;
        }

        /// <summary>Segments that span the palm (wrist→knuckle) get the thick palm radius;
        /// everything further out is a finger phalanx and stays thin.</summary>
        private static bool IsPalmSegment(OVRSkeleton.SkeletonType type, OVRSkeleton.BoneId childId)
        {
            if (type == OVRSkeleton.SkeletonType.XRHandLeft ||
                type == OVRSkeleton.SkeletonType.XRHandRight)
            {
                switch (childId)
                {
                    case OVRSkeleton.BoneId.XRHand_Palm:
                    case OVRSkeleton.BoneId.XRHand_ThumbMetacarpal:
                    case OVRSkeleton.BoneId.XRHand_IndexMetacarpal:
                    case OVRSkeleton.BoneId.XRHand_MiddleMetacarpal:
                    case OVRSkeleton.BoneId.XRHand_RingMetacarpal:
                    case OVRSkeleton.BoneId.XRHand_LittleMetacarpal:
                    // metacarpal→proximal spans the palm to the knuckle
                    case OVRSkeleton.BoneId.XRHand_ThumbProximal:
                    case OVRSkeleton.BoneId.XRHand_IndexProximal:
                    case OVRSkeleton.BoneId.XRHand_MiddleProximal:
                    case OVRSkeleton.BoneId.XRHand_RingProximal:
                    case OVRSkeleton.BoneId.XRHand_LittleProximal:
                        return true;
                    default:
                        return false;
                }
            }

            // Legacy rig: the knuckle bones (Index1/Middle1/Ring1, thumb/pinky metacarpals)
            // parent straight to the wrist root, so those segments span the palm.
            switch (childId)
            {
                case OVRSkeleton.BoneId.Hand_ForearmStub:
                case OVRSkeleton.BoneId.Hand_Thumb0:
                case OVRSkeleton.BoneId.Hand_Thumb1:
                case OVRSkeleton.BoneId.Hand_Index1:
                case OVRSkeleton.BoneId.Hand_Middle1:
                case OVRSkeleton.BoneId.Hand_Ring1:
                case OVRSkeleton.BoneId.Hand_Pinky0:
                case OVRSkeleton.BoneId.Hand_Pinky1:
                    return true;
                default:
                    return false;
            }
        }

        private void UpdateSegmentCapsule(in BoneSegment seg, bool ready)
        {
            if (seg.Capsule == null) return;

            if (!ready)
            {
                if (seg.Capsule.activeSelf) seg.Capsule.SetActive(false);
                return;
            }

            Vector3 a = seg.Parent.position;
            Vector3 b = seg.Child.position;
            Vector3 axis = b - a;
            float len = axis.magnitude;

            // Degenerate segment this frame (e.g. co-located helper bones): hide it.
            if (len < 1e-4f)
            {
                if (seg.Capsule.activeSelf) seg.Capsule.SetActive(false);
                return;
            }

            var t = seg.Capsule.transform;
            t.position = (a + b) * 0.5f;
            t.rotation = Quaternion.FromToRotation(Vector3.up, axis / len);
            // Unity's capsule mesh spans y −1..+1 at radius 0.5, so scale y to half the
            // segment length plus the radius (the hemispheres round off past each joint,
            // which keeps consecutive segments seamlessly connected at the knuckles).
            t.localScale = new Vector3(seg.Radius * 2f, len * 0.5f + seg.Radius, seg.Radius * 2f);

            if (!seg.Capsule.activeSelf) seg.Capsule.SetActive(true);
        }

        // ── Needle occluder ────────────────────────────────────────────────────────────

        private void UpdateNeedleOccluder()
        {
            if (_needleOccluder == null) return;

            Vector3 tip = default, hub = default;
            bool has = _occludeNeedle && _armManager != null && _armManager.TryGetNeedle(out tip, out hub);
            if (!has)
            {
                if (_needleOccluder.activeSelf) _needleOccluder.SetActive(false);
                return;
            }

            if (_mainCamTransform == null && Camera.main != null) _mainCamTransform = Camera.main.transform;
            Vector3 camPos = _mainCamTransform != null ? _mainCamTransform.position : tip;

            // Overhang past the hub covers the barrel/plunger behind the detected keypoint.
            Vector3 axis = hub - tip;
            float len = axis.magnitude;
            Vector3 dir = len > 1e-4f ? axis / len : Vector3.up;
            Vector3 hubEnd = hub + dir * _needleHubOverhangMeters;

            Vector3 mid = (tip + hubEnd) * 0.5f;
            // Bias toward the headset camera so the depth-only capsule sits in front of the
            // overlay surface (both are otherwise on the arm's depth plane).
            Vector3 toCam = camPos - mid;
            if (toCam.sqrMagnitude > 1e-6f) mid += toCam.normalized * _needleCameraBiasMeters;

            float fullLen = Vector3.Distance(tip, hubEnd);
            var t = _needleOccluder.transform;
            t.position = mid;
            t.rotation = Quaternion.FromToRotation(Vector3.up, dir);
            t.localScale = new Vector3(_needleOccluderRadius * 2f,
                                       fullLen * 0.5f + _needleOccluderRadius,
                                       _needleOccluderRadius * 2f);

            if (!_needleOccluder.activeSelf) _needleOccluder.SetActive(true);
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

            _transformOccluders = new GameObject[transformCount];
            for (int i = 0; i < transformCount; i++)
            {
                _transformOccluders[i] = CreateOccluderPrimitive(PrimitiveType.Sphere, $"TransformOccluder_{i}");
                _transformOccluders[i].transform.localScale = Vector3.one * (_occluderRadius * 2f);
            }

            // Segments are built lazily in LateUpdate once each skeleton initializes.
            _skeletonSegments = new List<BoneSegment>[skeletonCount];

            _needleOccluder = _occludeNeedle
                ? CreateOccluderPrimitive(PrimitiveType.Capsule, "NeedleOccluder")
                : null;
        }

        private GameObject CreateOccluderPrimitive(PrimitiveType type, string goName)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = goName;
            go.transform.SetParent(transform, false);

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
            if (_skeletonSegments != null)
            {
                foreach (var segments in _skeletonSegments)
                    if (segments != null)
                        foreach (var seg in segments)
                            if (seg.Capsule != null) Destroy(seg.Capsule);
                _skeletonSegments = null;
            }
            if (_needleOccluder != null)
            {
                Destroy(_needleOccluder);
                _needleOccluder = null;
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
    }
}
