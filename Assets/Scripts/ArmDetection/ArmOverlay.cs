using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Renders a single red world-space quad aligned along the closest detected arm.
    ///
    /// DEPTH / OCCLUSION
    /// -----------------
    /// Uses ArmDetection/ArmOverlayUnlit (RenderQueue = Geometry+1, ZTest LEqual).
    /// Paired with WearerArmOccluder (RenderQueue = Geometry-10, depth-write-only),
    /// the wearer's arm appears in FRONT of the overlay while the target arm is
    /// always covered by the overlay. See WearerArmOccluder.cs for full explanation.
    /// </summary>
    public class ArmOverlay : MonoBehaviour
    {
        [SerializeField] private Color  _color           = Color.red;
        [Tooltip("Quad thickness as a fraction of arm length.")]
        [SerializeField] private float  _thicknessRatio  = 0.22f;
        [SerializeField] private float  _minThickness    = 0.05f;
        [SerializeField] private float  _maxThickness    = 0.18f;

        private Transform _quad;
        private Material  _material;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake()
        {
            _material = CreateOverlayMaterial();

            var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
            go.name = "ArmOverlayQuad";
            go.transform.SetParent(transform, false);

            var col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);

            var mr = go.GetComponent<MeshRenderer>();
            mr.sharedMaterial       = _material;
            mr.shadowCastingMode    = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = UnityEngine.Rendering.LightProbeUsage.Off;
            mr.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;

            go.SetActive(false);
            _quad = go.transform;
        }

        private void OnDestroy()
        {
            if (_material != null) Destroy(_material);
        }

        // ── Public API ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// Call once per frame from ArmDetectionManager.
        /// Pass null to hide the overlay when no valid target arm is detected.
        /// </summary>
        public void Render((Vector3 shoulder, Vector3 wrist)? arm, Transform cameraTransform)
        {
            if (arm == null)
            {
                _quad.gameObject.SetActive(false);
                return;
            }

            var (shoulder, wrist) = arm.Value;
            Vector3 armDir = wrist - shoulder;
            float length = armDir.magnitude;

            if (length < 0.05f)
            {
                _quad.gameObject.SetActive(false);
                return;
            }

            // Orientation: local Y along the arm, local Z facing the camera.
            Vector3 up      = armDir / length;
            Vector3 mid     = (shoulder + wrist) * 0.5f;
            Vector3 camPos  = cameraTransform != null ? cameraTransform.position
                                                      : Camera.main.transform.position;
            Vector3 toCam   = camPos - mid;
            Vector3 forward = toCam.sqrMagnitude > 1e-6f ? toCam.normalized : Vector3.forward;
            Vector3 right   = Vector3.Cross(up, forward);
            if (right.sqrMagnitude < 1e-6f) right = Vector3.right;
            forward = Vector3.Cross(right.normalized, up).normalized;

            float thickness = Mathf.Clamp(length * _thicknessRatio, _minThickness, _maxThickness);

            _quad.SetPositionAndRotation(mid, Quaternion.LookRotation(forward, up));
            _quad.localScale = new Vector3(thickness, length, 1f);
            _quad.gameObject.SetActive(true);
        }

        // ── Private helpers ────────────────────────────────────────────────────────────

        private Material CreateOverlayMaterial()
        {
            // Prefer the custom depth-aware shader; fall back gracefully.
            var shader = Shader.Find("ArmDetection/ArmOverlayUnlit")
                      ?? Shader.Find("Universal Render Pipeline/Unlit")
                      ?? Shader.Find("Unlit/Color");

            var mat = new Material(shader);

            // Set colour via all common property names.
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", _color);
            if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     _color);

            return mat;
        }
    }
}
