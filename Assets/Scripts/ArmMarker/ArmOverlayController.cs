using UnityEngine;
using UnityEngine.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Controls the 3D arm overlay model based on QR marker detection.
    ///
    /// BEHAVIOUR
    /// ---------
    /// • Model is completely hidden when the QR code is not in view.
    /// • When the QR code is detected the model automatically snaps to the arm
    ///   and becomes visible (Vein View ON by default).
    /// • Left hand pinch  → toggle Vein View ON / OFF while QR is tracked.
    /// • Right hand pinch → force re-lock to the latest marker reading
    ///   (useful if the model drifts slightly).
    /// • If the QR code leaves the camera view the model hides automatically.
    ///   It reappears as soon as the QR is seen again.
    /// </summary>
    public class ArmOverlayController : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("The MarkerDetector component in the scene.")]
        [SerializeField] private MarkerDetector _markerDetector;

        [Tooltip("The 3D scanned arm model prefab (GLB imported via GLTFast).")]
        [SerializeField] private GameObject _armModelPrefab;

        [Header("Hand Tracking")]
        [Tooltip("OVRHand from '[BuildingBlock] Hand Tracking left'. " +
                 "Left pinch = toggle vein view on / off.")]
        [SerializeField] private OVRHand _leftHand;

        [Tooltip("OVRHand from '[BuildingBlock] Hand Tracking right'. " +
                 "Right pinch = force re-lock to current marker position.")]
        [SerializeField] private OVRHand _rightHand;

        [Tooltip("Seconds between pinch triggers to prevent accidental double-fires.")]
        [SerializeField] private float _pinchCooldown = 0.6f;

        [Header("Appearance")]
        [Tooltip("Opacity of the 3D model when Vein View is ON.")]
        [SerializeField, Range(0f, 1f)] private float _veinViewOpacity = 0.85f;

        [Tooltip("Scale multiplier for the 3D model. Adjust until it matches the real arm size.")]
        [SerializeField] private float _modelScale = 1f;

        [Header("Alignment — calibrate these in the Inspector")]
        [Tooltip("Shifts the model relative to the QR code in the MARKER'S local space.\n" +
                 "Y = along the arm (positive = toward wrist).\n" +
                 "Start with Y = half the arm length (e.g. 0.32 for a 65 cm arm) to\n" +
                 "move the model so the shoulder end aligns with the QR code.")]
        [SerializeField] private Vector3 _positionOffset = new Vector3(0f, 0.32f, 0f);

        [Tooltip("Rotates the model relative to the QR code orientation.\n" +
                 "Adjust X, Y, Z until the model lines up with the real arm.\n" +
                 "Start with (90, 0, 0) if the arm model was scanned lying flat.")]
        [SerializeField] private Vector3 _rotationOffset = new Vector3(90f, 0f, 0f);

        [Header("Tracking")]
        [Tooltip("When true, the model position updates every frame while the QR code is visible.\n" +
                 "When false, the position locks on first detection and only updates on right pinch.\n" +
                 "Recommended: true for a handheld arm, false for a fixed arm on a table.")]
        [SerializeField] private bool _liveTracking = true;

        [Header("UI — optional")]
        [Tooltip("World-space Text showing current status.")]
        [SerializeField] private Text _statusLabel;

        // ── Private state ──────────────────────────────────────────────────────────────

        private GameObject _modelInstance;
        private Renderer[] _renderers;
        private bool       _veinViewEnabled = true;   // vein view is ON by default when detected

        private bool  _rightWasPinching = false;
        private bool  _leftWasPinching  = false;
        private float _rightCooldown    = 0f;
        private float _leftCooldown     = 0f;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Start()
        {
            if (_armModelPrefab != null)
            {
                _modelInstance = Instantiate(_armModelPrefab);
                _renderers     = _modelInstance.GetComponentsInChildren<Renderer>(true);
                _modelInstance.SetActive(false);   // hidden until QR detected
            }
            else
            {
                Debug.LogWarning("[ArmOverlayController] No arm model prefab assigned — " +
                                 "drag your scanned arm GLB prefab here.");
            }

            if (_markerDetector == null)
                Debug.LogError("[ArmOverlayController] MarkerDetector not assigned.");

            UpdateLabel("Searching for QR marker…", Color.yellow);
        }

        private void Update()
        {
            if (_modelInstance == null) return;

            // Cooldown timers.
            if (_rightCooldown > 0f) _rightCooldown -= Time.deltaTime;
            if (_leftCooldown  > 0f) _leftCooldown  -= Time.deltaTime;

            bool rightPinch = DetectPinchDown(_rightHand, ref _rightWasPinching, ref _rightCooldown);
            bool leftPinch  = DetectPinchDown(_leftHand,  ref _leftWasPinching,  ref _leftCooldown);

            bool markerSeen = _markerDetector != null && _markerDetector.IsDetected;

            // ── Update model position ──────────────────────────────────────────────────
            if (markerSeen)
            {
                if (_liveTracking || rightPinch)
                    SnapToMarker();    // live: every frame | locked: only on right pinch
            }

            // Right pinch while marker not visible: force a re-search.
            if (rightPinch && !markerSeen)
                UpdateLabel("Point at the QR marker to re-lock…", Color.yellow);

            // ── Toggle vein view with left pinch ───────────────────────────────────────
            if (leftPinch)
            {
                _veinViewEnabled = !_veinViewEnabled;
                Debug.Log($"[ArmOverlayController] Vein view: {(_veinViewEnabled ? "ON" : "OFF")}");
            }

            // ── Show / hide model ──────────────────────────────────────────────────────
            // Visible only when: QR marker is detected AND vein view is toggled on.
            bool shouldShow = markerSeen && _veinViewEnabled;

            if (shouldShow != _modelInstance.activeSelf)
                _modelInstance.SetActive(shouldShow);

            if (shouldShow)
                SetOpacity(_veinViewOpacity);

            // ── Status label ───────────────────────────────────────────────────────────
            if (markerSeen && _veinViewEnabled)
                UpdateLabel("● Vein View  |  Left pinch: hide", Color.green);
            else if (markerSeen && !_veinViewEnabled)
                UpdateLabel("○ Normal View  |  Left pinch: show veins", Color.white);
            else
                UpdateLabel("Searching for QR marker…", Color.yellow);
        }

        // ── Helpers ────────────────────────────────────────────────────────────────────

        private void SnapToMarker()
        {
            Quaternion markerRot   = _markerDetector.MarkerWorldRot;
            Quaternion offsetRot   = Quaternion.Euler(_rotationOffset);
            Quaternion finalRot    = markerRot * offsetRot;

            // Apply position offset in the marker's local space so
            // it shifts along the arm regardless of how the arm is oriented.
            Vector3 finalPos = _markerDetector.MarkerWorldPos
                             + markerRot * _positionOffset;

            _modelInstance.transform.SetPositionAndRotation(finalPos, finalRot);
            _modelInstance.transform.localScale = Vector3.one * _modelScale;
        }

        private bool DetectPinchDown(OVRHand hand, ref bool wasPinching, ref float cooldown)
        {
            if (hand == null || !hand.IsTracked) { wasPinching = false; return false; }
            bool isPinching      = hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
            bool pinchStartedNow = isPinching && !wasPinching && cooldown <= 0f;
            if  (pinchStartedNow) cooldown = _pinchCooldown;
            wasPinching          = isPinching;
            return pinchStartedNow;
        }

        private void SetOpacity(float alpha)
        {
            if (_renderers == null) return;
            foreach (var r in _renderers)
                foreach (var mat in r.materials)
                {
                    Color c = Color.white;
                    if      (mat.HasProperty("_BaseColor")) c = mat.GetColor("_BaseColor");
                    else if (mat.HasProperty("_Color"))     c = mat.GetColor("_Color");
                    c.a = alpha;
                    if      (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", c);
                    else if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     c);
                }
        }

        private void UpdateLabel(string text, Color colour)
        {
            if (_statusLabel == null) return;
            _statusLabel.text  = text;
            _statusLabel.color = colour;
        }
    }
}
