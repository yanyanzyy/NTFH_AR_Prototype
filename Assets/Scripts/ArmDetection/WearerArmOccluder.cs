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
    /// Assign the wearer's wrist bone transforms to _wearerArmTransforms.
    /// These are the same bones used by WearerHandFilter (e.g. OVRSkeleton wrist joints).
    /// For better arm coverage also add elbow or shoulder bones from OVRBody if available.
    /// </summary>
    public class WearerArmOccluder : MonoBehaviour
    {
        [Tooltip("World-space transforms of the wearer's arm joints.\n" +
                 "Add wrists (required), elbows and/or shoulders for broader coverage.\n" +
                 "Use the same OVRSkeleton wrist bones as WearerHandFilter.")]
        [SerializeField] private Transform[] _wearerArmTransforms;

        [Tooltip("Radius of each sphere occluder in metres. 0.12 m gives ~24 cm diameter, " +
                 "enough to cover the wrist / lower hand.")]
        [SerializeField] private float _occluderRadius = 0.12f;

        private GameObject[] _occluders;
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
            if (_wearerArmTransforms == null || _occluders == null) return;

            int count = Mathf.Min(_wearerArmTransforms.Length, _occluders.Length);
            for (int i = 0; i < count; i++)
            {
                bool valid = _wearerArmTransforms[i] != null;
                _occluders[i].SetActive(valid);
                if (valid)
                    _occluders[i].transform.position = _wearerArmTransforms[i].position;
            }
            for (int i = count; i < _occluders.Length; i++)
                _occluders[i].SetActive(false);
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
            int count = _wearerArmTransforms?.Length ?? 0;
            _occluders = new GameObject[count];
            float diameter = _occluderRadius * 2f;

            for (int i = 0; i < count; i++)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = $"WristDepthOccluder_{i}";
                go.transform.SetParent(transform, false);
                go.transform.localScale = Vector3.one * diameter;

                // Remove collider — we only want rendering.
                var col = go.GetComponent<Collider>();
                if (col != null) Destroy(col);

                var mr = go.GetComponent<MeshRenderer>();
                mr.sharedMaterial = _depthOnlyMaterial;
                mr.shadowCastingMode  = UnityEngine.Rendering.ShadowCastingMode.Off;
                mr.receiveShadows     = false;
                mr.lightProbeUsage    = UnityEngine.Rendering.LightProbeUsage.Off;
                mr.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;

                go.SetActive(false);
                _occluders[i] = go;
            }
        }

        private void DestroyOccluders()
        {
            if (_occluders == null) return;
            foreach (var o in _occluders)
                if (o != null) Destroy(o);
            _occluders = null;
        }

        private static Material CreateDepthOnlyMaterial()
        {
            var shader = Shader.Find("ArmDetection/DepthOccluder");
            if (shader != null) return new Material(shader);

            // Fallback: URP Unlit won't occlude properly, but at least won't crash.
            Debug.LogError("[WearerArmOccluder] Shader 'ArmDetection/DepthOccluder' not found. " +
                           "Ensure DepthOccluder.shader is inside the Assets folder.");
            return new Material(Shader.Find("Universal Render Pipeline/Unlit")
                                ?? Shader.Find("Unlit/Color"));
        }
    }
}
