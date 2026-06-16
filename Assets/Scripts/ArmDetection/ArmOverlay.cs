using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Renders a textured cylinder aligned along the closest detected arm.
    ///
    /// A cylinder is used instead of a flat quad so the overlay is visible from
    /// any viewing angle — the user can walk around the arm and always see the texture.
    ///
    /// Assign your arm photo to _overlayTexture in the Inspector; leave it empty to
    /// show a solid colour instead (useful for debugging).
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
        [Tooltip("Photo or texture to wrap around the detected arm. " +
                 "Leave empty to show a solid colour instead.")]
        [SerializeField] private Texture2D _overlayTexture;

        [Tooltip("Solid colour used when no texture is assigned, or as a tint when one is. " +
                 "Set to white for an accurate texture with no tint.")]
        [SerializeField] private Color _color = Color.red;

        [Tooltip("0 = fully transparent, 1 = fully opaque. Values like 0.6–0.8 let you see the real arm through the overlay.")]
        [Range(0f, 1f)]
        [SerializeField] private float _opacity = 1f;

        [Header("Fixed-size mode (mannequin arm)")]
        [Tooltip("When enabled, the cylinder keeps a CONSTANT real-world size matched to the physical " +
                 "mannequin arm instead of stretching to the detected shoulder→wrist span each frame. " +
                 "The detection only drives position and orientation; apparent size changes naturally " +
                 "with distance because the cylinder is a fixed-size world-space object.")]
        [SerializeField] private bool _useFixedSize = true;

        /// <summary>True when the overlay renders at a constant real-world size.</summary>
        public bool UseFixedSize => _useFixedSize;
        /// <summary>
        /// The physical arm length the overlay renders at. ArmDetectionManager uses this
        /// as the calibration constant when deriving depth from the detected pixel span,
        /// which makes the projected shoulder→wrist distance equal this length — so the
        /// overlay ends line up with the detected arm exactly.
        /// </summary>
        public float FixedLengthMeters => _fixedLengthMeters;
        [Tooltip("Real shoulder→wrist length of the mannequin arm in metres. " +
                 "Measure the physical trainer arm (Limbs & Things venipuncture arm ≈ 0.55 m) and enter it here.")]
        [SerializeField] private float _fixedLengthMeters = 0.55f;
        [Tooltip("Real diameter of the mannequin arm in metres (forearm ≈ 0.085 m).")]
        [SerializeField] private float _fixedDiameterMeters = 0.085f;

        [Header("Stretch-to-fit mode (used when fixed size is off)")]
        [Tooltip("Cylinder diameter as a fraction of detected arm length. " +
                 "Increase to make the overlay wider, decrease to make it narrower.")]
        [SerializeField] private float _thicknessRatio = 0.22f;
        [SerializeField] private float _minThickness   = 0.05f;
        [SerializeField] private float _maxThickness   = 0.18f;

        [Header("Texture mapping")]
        [Tooltip("How many times the image tiles on the cylinder surface.\n" +
                 "X = around the circumference (1 = wraps once around, 2 = twice, 0.5 = half).\n" +
                 "Y = along the arm length (1 = image fills shoulder-to-wrist exactly, " +
                 "2 = image repeats twice along the arm).")]
        [SerializeField] private Vector2 _textureTiling = Vector2.one;

        [Tooltip("Shifts the texture on the cylinder.\n" +
                 "Y = move image up (+) or down (-) along the arm.\n" +
                 "X = rotate the image around the cylinder.")]
        [SerializeField] private Vector2 _textureOffset = Vector2.zero;

        [Header("Debug")]
        [Tooltip("Force the overlay visible even when projected arm length is tiny. " +
                 "Useful in Editor where fallback projection may produce short arms. " +
                 "Disable before shipping.")]
        [SerializeField] private bool _forceVisible = false;

        private Transform _mesh;
        private Material  _material;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake()
        {
            // 3D model
            if (_armModelPrefab != null)
            {
                var modelGo = Instantiate(_armModelPrefab, transform);
                modelGo.name = "ArmModelOverlay";
                modelGo.SetActive(false);
                _model = modelGo.transform;
            }

            // Cylinder is visible from all angles — the user can walk around the arm.
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = "ArmOverlayCylinder";
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
            _mesh = go.transform;
        }

        private void OnDestroy()
        {
            if (_quadMaterial != null) Destroy(_quadMaterial);
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
                _mesh.gameObject.SetActive(false);
                return;
            }

            var (shoulder, wrist) = arm.Value;
            Vector3 armDir = wrist - shoulder;
            float   length = armDir.magnitude;

            if (length < 0.05f)
            {
                if (!_forceVisible)
                {
                    Debug.LogWarning($"[ArmOverlay] Arm too short ({length:F3} m) — cylinder hidden. " +
                                     $"Shoulder={shoulder} Wrist={wrist}. " +
                                     "Enable _forceVisible on ArmOverlay to override.");
                    _mesh.gameObject.SetActive(false);
                    return;
                }

                // Force-visible: use the midpoint + a 0.6 m stand-in arm.
                var standInMid = (shoulder + wrist) * 0.5f;
                var standInDir = armDir.sqrMagnitude > 1e-6f ? armDir.normalized : Vector3.up;
                shoulder = standInMid - standInDir * 0.3f;
                wrist    = standInMid + standInDir * 0.3f;
                armDir   = wrist - shoulder;
                length   = armDir.magnitude;
                Debug.Log($"[ArmOverlay] _forceVisible active — original arm too short; using 0.6 m stand-in.");
            }

            // ── Position and orient the cylinder ──────────────────────────────────────
            Vector3 up  = armDir / length;
            Vector3 mid = (shoulder + wrist) * 0.5f;

            // Align the cylinder's local Y axis with the arm direction.
            // A cylinder needs no "face camera" calculation — it is round and looks
            // identical from every angle perpendicular to the arm axis.
            Quaternion rotation = Quaternion.FromToRotation(Vector3.up, up);

            // Unity Cylinder default: height = 2 units (Y: -1 to +1), radius = 0.5 units.
            //   scale.y = length / 2   → makes cylinder exactly arm-length tall
            //   scale.x = scale.z = thickness  → default radius 0.5 * scale = thickness/2
            float renderLength, thickness;
            if (_useFixedSize)
            {
                // Fixed-size mode: the cylinder always matches the physical mannequin
                // arm. Detection noise in bbox / keypoint span cannot stretch it —
                // only position and orientation come from the detection.
                renderLength = _fixedLengthMeters;
                thickness    = _fixedDiameterMeters;
            }
            else
            {
                renderLength = length;
                thickness    = Mathf.Clamp(length * _thicknessRatio, _minThickness, _maxThickness);
            }

            _mesh.SetPositionAndRotation(mid, rotation);
            _mesh.localScale = new Vector3(thickness, renderLength * 0.5f, thickness);

            // Apply tiling, offset and opacity every frame so Inspector tweaks take effect live.
            _material.SetTextureScale ("_BaseMap", _textureTiling);
            _material.SetTextureOffset("_BaseMap", _textureOffset);
            _material.SetColor("_BaseColor", new Color(_color.r, _color.g, _color.b, _opacity));

            if (!_mesh.gameObject.activeSelf)
                Debug.Log($"[ArmOverlay] Cylinder activated — mid={mid} len={renderLength:F2}m thickness={thickness:F3}m" +
                          (_useFixedSize ? " (fixed size)" : ""));

            _mesh.gameObject.SetActive(true);
        }

        private void SetVisible(bool visible)
        {
            if (_model != null) _model.gameObject.SetActive(visible);
            _quad.gameObject.SetActive(visible);
        }

        private Material CreateQuadMaterial()
        {
            var shader = Shader.Find("ArmDetection/ArmOverlayUnlit")
                      ?? Shader.Find("Universal Render Pipeline/Unlit")
                      ?? Shader.Find("Unlit/Color");
            var mat = new Material(shader);

            if (_overlayTexture != null)
            {
                // Texture mode: apply the photo, use _color as a tint (set to white for accurate colours).
                if (mat.HasProperty("_BaseMap"))  mat.SetTexture("_BaseMap",  _overlayTexture);
                if (mat.HasProperty("_MainTex"))  mat.SetTexture("_MainTex",  _overlayTexture);
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", _color);
                if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     _color);
            }
            else
            {
                // No texture — solid colour fallback (default red, good for debugging).
                if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", _color);
                if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     _color);
            }

            return mat;
        }
    }
}
