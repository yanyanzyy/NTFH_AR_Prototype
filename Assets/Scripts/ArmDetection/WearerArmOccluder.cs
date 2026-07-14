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

        [Tooltip("Radius of each wrist occluder in metres. Keep this close to wrist size so " +
                 "hands provide depth feedback without cutting a large circle from world UI.")]
        [SerializeField] private float _occluderRadius = 0.055f;

        private GameObject[] _transformOccluders;  // one per _wearerArmTransforms entry
        private GameObject[] _skeletonOccluders;   // one per _wearerSkeletons entry
        private Material     _depthOnlyMaterial;

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

            // 2. Skeleton-based occluders — auto-resolve wrist bone each frame.
            if (_wearerSkeletons != null && _skeletonOccluders != null)
            {
                int count = Mathf.Min(_wearerSkeletons.Length, _skeletonOccluders.Length);
                for (int i = 0; i < count; i++)
                {
                    var wristPos = IsSkeletonReady(_wearerSkeletons[i])
                        ? FindWristPosition(_wearerSkeletons[i]) : null;
                    _skeletonOccluders[i].SetActive(wristPos.HasValue);
                    if (wristPos.HasValue)
                        _skeletonOccluders[i].transform.position = wristPos.Value;
                }
                for (int i = count; i < _skeletonOccluders.Length; i++)
                    _skeletonOccluders[i].SetActive(false);
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

            _transformOccluders = new GameObject[transformCount];
            _skeletonOccluders  = new GameObject[skeletonCount];

            for (int i = 0; i < transformCount; i++)
                _transformOccluders[i] = CreateOccluderSphere($"TransformOccluder_{i}");

            for (int i = 0; i < skeletonCount; i++)
                _skeletonOccluders[i]  = CreateOccluderSphere($"SkeletonOccluder_{i}");
        }

        private GameObject CreateOccluderSphere(string goName)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = goName;
            go.transform.SetParent(transform, false);
            go.transform.localScale = Vector3.one * (_occluderRadius * 2f);

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

        private static Vector3? FindWristPosition(OVRSkeleton skel)
        {
            foreach (var bone in skel.Bones)
            {
                if (bone == null || bone.Transform == null) continue;
                if (bone.Id == OVRSkeleton.BoneId.Hand_WristRoot)
                    return bone.Transform.position;
            }
            return null;
        }
    }
}
