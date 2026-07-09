using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Renders a world-space overlay aligned along the closest detected arm.
    ///
    /// When a 3D model prefab is assigned it is used as the overlay; otherwise
    /// the component falls back to a simple coloured quad (useful for debugging).
    ///
    /// 3D MODEL SETUP
    /// --------------
    /// 1. Assign your arm prefab (e.g. "3dScanArm") to _armModelPrefab.
    /// 2. Set _modelArmAxis to the local axis of the prefab that points from
    ///    the shoulder end toward the wrist end (default: Vector3.forward / +Z).
    /// 3. Set _naturalArmLengthMeters to the real-world length the model
    ///    represents at scale (1,1,1) — used when _scaleToFitDetectedLength is on.
    /// 4. If the prefab pivot is not at the arm's midpoint, adjust _pivotOffset:
    ///      0.0 = pivot is at the shoulder end
    ///      0.5 = pivot is at the midpoint  (default)
    ///      1.0 = pivot is at the wrist end
    /// 5. If the overlay sits to one side of the real arm, adjust _lateralOffset
    ///    (metres) to slide it left/right perpendicular to the arm.
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
        // ── 3D model ───────────────────────────────────────────────────────────────────
        [Header("3D Model (optional)")]
        [Tooltip("Prefab to use as the arm overlay. If unassigned, falls back to the debug quad.")]
        [SerializeField] private GameObject _armModelPrefab;

        [Tooltip("Which local axis of the prefab points from the shoulder end toward the wrist end.")]
        [SerializeField] private Vector3 _modelArmAxis = Vector3.forward;

        [Tooltip("Real-world arm length (metres) that the prefab represents at its natural scale (1,1,1). " +
                 "Only used when Scale To Fit Detected Length is enabled.")]
        [SerializeField] private float _naturalArmLengthMeters = 0.65f;

        [Tooltip("Uniformly scale the model so its length matches the detected shoulder-to-wrist distance. " +
                 "Disable this if your depth estimate is too noisy and you prefer a fixed-size model.")]
        [SerializeField] private bool _scaleToFitDetectedLength = true;

        [Tooltip("Where the prefab's pivot sits along the arm:\n" +
                 "  0.0 = shoulder end\n  0.5 = midpoint (default)\n  1.0 = wrist end")]
        [SerializeField, Range(0f, 1f)] private float _pivotOffset = 0.5f;

        [Tooltip("Slide the model left/right, perpendicular to the arm in the horizontal plane " +
                 "(metres). +ve/-ve pick a side; choose the sign that lines up with the real arm.")]
        [SerializeField, Range(-0.3f, 0.3f)] private float _lateralOffset = 0f;

        // ── Debug quad (fallback) ──────────────────────────────────────────────────────
        [Header("Debug Quad (fallback when no prefab assigned)")]
        [SerializeField] private Color _color          = Color.red;
        [Tooltip("Quad thickness as a fraction of arm length.")]
        [SerializeField] private float _thicknessRatio = 0.22f;
        [SerializeField] private float _minThickness   = 0.05f;
        [SerializeField] private float _maxThickness   = 0.18f;

        [Header("Debug")]
        [Tooltip("Force the overlay visible even when the projected arm length is tiny (< 0.05 m). " +
                 "Useful in Editor where fallback projection may produce short arms. " +
                 "Disable before shipping.")]
        [SerializeField] private bool _forceVisible = false;
        [Tooltip("Show the debug quad alongside the 3D model for alignment checking.")]
        [SerializeField] private bool _showDebugQuadAlongsideModel = false;

        // ── Private state ──────────────────────────────────────────────────────────────
        private Transform _model;
        private Transform _quad;
        private Material  _quadMaterial;
        private float     _nextShortArmWarning;

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

            // Debug quad (always created; hidden unless needed)
            _quadMaterial = CreateQuadMaterial();
            var quadGo = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quadGo.name = "ArmOverlayQuad";
            quadGo.transform.SetParent(transform, false);
            Destroy(quadGo.GetComponent<Collider>());
            var mr = quadGo.GetComponent<MeshRenderer>();
            mr.sharedMaterial       = _quadMaterial;
            mr.shadowCastingMode    = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows       = false;
            mr.lightProbeUsage      = UnityEngine.Rendering.LightProbeUsage.Off;
            mr.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
            quadGo.SetActive(false);
            _quad = quadGo.transform;
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
                SetVisible(false);
                return;
            }

            var (shoulder, wrist) = arm.Value;
            Vector3 armVec = wrist - shoulder;
            float length = armVec.magnitude;

            if (length < 0.05f)
            {
                if (!_forceVisible)
                {
                    // Throttled: this can trip every frame while a degenerate detection is
                    // held, and per-frame logging (string alloc + logcat I/O) tanks Quest FPS.
                    if (Time.unscaledTime >= _nextShortArmWarning)
                    {
                        _nextShortArmWarning = Time.unscaledTime + 5f;
                        Debug.LogWarning($"[ArmOverlay] Arm too short ({length:F3} m) — overlay hidden. " +
                                         $"Shoulder={shoulder} Wrist={wrist}. " +
                                         "Enable _forceVisible on ArmOverlay to override.");
                    }
                    SetVisible(false);
                    return;
                }

                // Force-visible: stand-in 0.6 m arm at midpoint
                Vector3 mid0 = (shoulder + wrist) * 0.5f;
                Vector3 dir0 = armVec.sqrMagnitude > 1e-6f ? armVec.normalized : Vector3.up;
                shoulder = mid0 - dir0 * 0.3f;
                wrist    = mid0 + dir0 * 0.3f;
                armVec   = wrist - shoulder;
                length   = armVec.magnitude;
            }

            Vector3 armDir = armVec / length;
            Vector3 mid    = Vector3.Lerp(shoulder, wrist, 0.5f);

            if (_model != null)
                PlaceModel(shoulder, wrist, armDir, length);

            bool useQuad = _model == null || _showDebugQuadAlongsideModel;
            if (useQuad)
                PlaceQuad(shoulder, wrist, armDir, length, mid, cameraTransform);
            else
                _quad.gameObject.SetActive(false);
        }

        // ── Private helpers ────────────────────────────────────────────────────────────

        private void PlaceModel(Vector3 shoulder, Vector3 wrist, Vector3 armDir, float length)
        {
            // Compute the world rotation that takes _modelArmAxis onto armDir.
            Vector3 axis = _modelArmAxis.sqrMagnitude > 1e-6f
                ? _modelArmAxis.normalized
                : Vector3.forward;

            Quaternion rot = Quaternion.FromToRotation(axis, armDir);

            // Pivot offset: move the anchor point along the arm direction.
            // _pivotOffset=0 → place at shoulder; 0.5 → midpoint; 1 → wrist.
            Vector3 pos = Vector3.Lerp(shoulder, wrist, _pivotOffset);

            // Lateral offset: slide perpendicular to the arm, in the horizontal plane.
            // Sign picks the side; the user tunes the value to match the real arm.
            if (Mathf.Abs(_lateralOffset) > 1e-6f)
            {
                Vector3 lateralDir = Vector3.Cross(Vector3.up, armDir);
                if (lateralDir.sqrMagnitude < 1e-6f)          // arm ~vertical: pick any perpendicular
                    lateralDir = Vector3.Cross(Vector3.forward, armDir);
                pos += lateralDir.normalized * _lateralOffset;
            }

            // Uniform scale so the model's length matches the detected arm.
            Vector3 scale = Vector3.one;
            if (_scaleToFitDetectedLength && _naturalArmLengthMeters > 0.001f)
            {
                float s = length / _naturalArmLengthMeters;
                scale = new Vector3(s, s, s);
            }

            if (!_model.gameObject.activeSelf)
                Debug.Log($"[ArmOverlay] 3D model activated — mid={(shoulder + wrist) * 0.5f} len={length:F2}m");

            _model.SetPositionAndRotation(pos, rot);
            _model.localScale = scale;
            _model.gameObject.SetActive(true);
        }

        private void PlaceQuad(Vector3 shoulder, Vector3 wrist, Vector3 armDir, float length,
                               Vector3 mid, Transform cameraTransform)
        {
            Vector3 camPos  = cameraTransform != null ? cameraTransform.position
                                                      : Camera.main != null ? Camera.main.transform.position
                                                      : Vector3.zero;
            Vector3 toCam   = camPos - mid;
            Vector3 forward = toCam.sqrMagnitude > 1e-6f ? toCam.normalized : Vector3.forward;
            Vector3 right   = Vector3.Cross(armDir, forward);
            if (right.sqrMagnitude < 1e-6f) right = Vector3.right;
            forward = Vector3.Cross(right.normalized, armDir).normalized;

            float thickness = Mathf.Clamp(length * _thicknessRatio, _minThickness, _maxThickness);

            _quad.SetPositionAndRotation(mid, Quaternion.LookRotation(forward, armDir));
            _quad.localScale = new Vector3(thickness, length, 1f);

            if (!_quad.gameObject.activeSelf)
                Debug.Log($"[ArmOverlay] Quad activated — mid={mid} len={length:F2}m");

            _quad.gameObject.SetActive(true);
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
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", _color);
            if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     _color);
            return mat;
        }
    }
}
