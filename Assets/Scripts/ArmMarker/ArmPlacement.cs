using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Press-to-place arm overlay with vein toggle, controlled entirely by hand tracking.
///
/// GESTURES
/// --------
/// Right hand index pinch — place the model / reposition
/// Left hand index pinch  — toggle vein overlay on / off (only works after placing)
///
/// HOW TO USE
/// ----------
/// 1. Assign the OVRHand components from the Hand Tracking building blocks
///    to _rightHand and _leftHand in the Inspector.
/// 2. Launch the app — a ghost of the 3D model follows your gaze.
/// 3. Look at the mannequin arm and right-hand pinch to lock the model there.
/// 4. Left-hand pinch at any time to toggle veins on/off.
/// 5. Right-hand pinch again to reposition.
/// </summary>
public class ArmPlacement : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The 3D arm model prefab (your scanned GLB).")]
    [SerializeField] private GameObject _armModelPrefab;

    [Tooltip("Camera used for the placement ray — drag CenterEyeAnchor here.")]
    [SerializeField] private Transform _cameraTransform;

    [Header("Hand Tracking")]
    [Tooltip("Drag the OVRHand component from '[BuildingBlock] Hand Tracking right' here.")]
    [SerializeField] private OVRHand _rightHand;

    [Tooltip("Drag the OVRHand component from '[BuildingBlock] Hand Tracking left' here.")]
    [SerializeField] private OVRHand _leftHand;

    [Tooltip("How long (seconds) to wait after a pinch before another pinch can register. " +
             "Prevents accidental double-triggers.")]
    [SerializeField] private float _pinchCooldown = 0.6f;

    [Header("UI — optional")]
    [Tooltip("Shows current mode: Positioning / Vein View ON / Normal View.")]
    [SerializeField] private Text _modeLabel;

    [Tooltip("Shows gesture hints to the user.")]
    [SerializeField] private Text _instructionLabel;

    [Header("Placement")]
    [Tooltip("Distance in front of camera when no surface is hit.")]
    [SerializeField] private float _defaultPlacementDistance = 0.7f;

    [Header("Opacity")]
    [Tooltip("Opacity of the ghost preview while positioning.")]
    [SerializeField, Range(0f, 1f)] private float _previewOpacity = 0.35f;

    [Tooltip("Opacity of the vein overlay once placed.")]
    [SerializeField, Range(0f, 1f)] private float _veinViewOpacity = 0.85f;

    // ── Private state ──────────────────────────────────────────────────────────────────

    private enum PlacementState { Preview, Placed }
    private PlacementState _placementState = PlacementState.Preview;
    private bool           _veinViewActive = false;

    private GameObject _modelInstance;
    private Renderer[] _renderers;

    // Pinch tracking — we only trigger on the LEADING EDGE of a pinch (not while held).
    private bool  _rightWasPinching  = false;
    private bool  _leftWasPinching   = false;
    private float _rightCooldownTimer = 0f;
    private float _leftCooldownTimer  = 0f;

    // ── Unity lifecycle ────────────────────────────────────────────────────────────────

    private void Start()
    {
        if (_cameraTransform == null)
            _cameraTransform = Camera.main?.transform ?? transform;

        if (_armModelPrefab != null)
        {
            _modelInstance = Instantiate(_armModelPrefab);
            _renderers     = _modelInstance.GetComponentsInChildren<Renderer>(true);
            _modelInstance.SetActive(true);
            SetModelOpacity(_previewOpacity);
        }
        else
        {
            Debug.LogWarning("[ArmPlacement] No _armModelPrefab assigned.");
        }

        if (_rightHand == null || _leftHand == null)
            Debug.LogWarning("[ArmPlacement] Hand references not assigned — drag OVRHand components here.");

        RefreshUI();
    }

    private void Update()
    {
        if (_modelInstance == null) return;

        // Cool-down timers — count down each frame.
        if (_rightCooldownTimer > 0f) _rightCooldownTimer -= Time.deltaTime;
        if (_leftCooldownTimer  > 0f) _leftCooldownTimer  -= Time.deltaTime;

        bool rightPinchDown = DetectPinchDown(_rightHand, ref _rightWasPinching, ref _rightCooldownTimer);
        bool leftPinchDown  = DetectPinchDown(_leftHand,  ref _leftWasPinching,  ref _leftCooldownTimer);

        // ── Preview mode: follow gaze, right pinch to place ───────────────────────────
        if (_placementState == PlacementState.Preview)
        {
            UpdatePreviewPosition();

            if (rightPinchDown)
            {
                _placementState = PlacementState.Placed;
                _veinViewActive = true;   // auto-enable vein view on first placement
                ApplyCurrentVisuals();
                RefreshUI();
                Debug.Log("[ArmPlacement] Model placed.");
            }
        }
        // ── Placed mode: right pinch to reposition, left pinch to toggle ─────────────
        else
        {
            if (rightPinchDown)
            {
                _placementState = PlacementState.Preview;
                _veinViewActive = false;
                SetModelOpacity(_previewOpacity);
                _modelInstance.SetActive(true);
                RefreshUI();
                Debug.Log("[ArmPlacement] Repositioning.");
            }

            if (leftPinchDown)
            {
                _veinViewActive = !_veinViewActive;
                ApplyCurrentVisuals();
                RefreshUI();
                Debug.Log($"[ArmPlacement] Vein view: {(_veinViewActive ? "ON" : "OFF")}");
            }
        }
    }

    // ── Pinch detection ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns true on the frame the index-finger pinch starts (leading edge only).
    /// Ignores subsequent frames while the pinch is held, and respects the cooldown.
    /// </summary>
    private bool DetectPinchDown(OVRHand hand, ref bool wasPinching, ref float cooldownTimer)
    {
        if (hand == null || !hand.IsTracked)
        {
            wasPinching = false;
            return false;
        }

        bool isPinching = hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        bool pinchStartedThisFrame = isPinching && !wasPinching && cooldownTimer <= 0f;

        if (pinchStartedThisFrame)
            cooldownTimer = _pinchCooldown;   // start cooldown so next trigger needs to wait

        wasPinching = isPinching;
        return pinchStartedThisFrame;
    }

    // ── Placement helpers ──────────────────────────────────────────────────────────────

    private void UpdatePreviewPosition()
    {
        Ray ray = new Ray(_cameraTransform.position, _cameraTransform.forward);

        Vector3    targetPos;
        Quaternion targetRot;

        if (Physics.Raycast(ray, out RaycastHit hit, 3f))
        {
            targetPos = hit.point;
            targetRot = Quaternion.FromToRotation(Vector3.up, hit.normal)
                      * Quaternion.Euler(0, _cameraTransform.eulerAngles.y, 0);
        }
        else
        {
            targetPos = _cameraTransform.position
                      + _cameraTransform.forward * _defaultPlacementDistance;
            targetRot = Quaternion.Euler(0, _cameraTransform.eulerAngles.y, 0);
        }

        _modelInstance.transform.position = Vector3.Lerp(
            _modelInstance.transform.position, targetPos, Time.deltaTime * 12f);
        _modelInstance.transform.rotation = Quaternion.Slerp(
            _modelInstance.transform.rotation, targetRot, Time.deltaTime * 12f);
    }

    // ── Visuals ────────────────────────────────────────────────────────────────────────

    private void ApplyCurrentVisuals()
    {
        if (_veinViewActive)
        {
            _modelInstance.SetActive(true);
            SetModelOpacity(_veinViewOpacity);
        }
        else
        {
            // Normal view — hide the 3D model so the real arm shows through unobstructed.
            _modelInstance.SetActive(false);
        }
    }

    private void SetModelOpacity(float alpha)
    {
        if (_renderers == null) return;
        foreach (var r in _renderers)
        {
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
    }

    // ── UI ─────────────────────────────────────────────────────────────────────────────

    private void RefreshUI()
    {
        if (_instructionLabel != null)
        {
            _instructionLabel.text = _placementState == PlacementState.Preview
                ? "Look at the arm — right hand pinch to place overlay"
                : "Left pinch: toggle veins  |  Right pinch: reposition";
        }

        if (_modeLabel != null)
        {
            if (_placementState == PlacementState.Preview)
            {
                _modeLabel.text  = "Positioning…";
                _modeLabel.color = Color.yellow;
            }
            else if (_veinViewActive)
            {
                _modeLabel.text  = "● Vein View";
                _modeLabel.color = Color.green;
            }
            else
            {
                _modeLabel.text  = "○ Normal View";
                _modeLabel.color = Color.white;
            }
        }
    }
}
