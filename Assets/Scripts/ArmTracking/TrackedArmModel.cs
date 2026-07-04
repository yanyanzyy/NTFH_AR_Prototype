using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Drives the 3D arm model so it overlays the physical mannequin arm.
    ///
    /// Pose sources, in priority order:
    ///   1. <see cref="MarkerArmTracker"/> — full 6-DoF from the marker band
    ///      (position + rotation incl. roll; follows the arm as it is moved/turned).
    ///   2. YOLO fallback — when markers are occluded/lost for longer than the grace
    ///      period, the arm axis from <see cref="ArmDetectionManager"/> (shoulder→wrist)
    ///      plus a calibrated roll keeps the overlay roughly in place.
    ///
    /// CALIBRATION (once): enter Play mode with the band tracked, move/rotate _model in
    /// the scene until the virtual arm visually covers the physical arm, then use the
    /// component context menu "Capture Model Offset From Current Transform". The offset
    /// is stored in the serialized fields (copy the component values out of Play mode
    /// with "Copy Component" / "Paste Component Values").
    /// </summary>
    public class TrackedArmModel : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private MarkerArmTracker _tracker;
        [SerializeField] private ArmDetectionManager _armManager;
        [Tooltip("Root transform of the 3D arm model to overlay.")]
        [SerializeField] private Transform _model;

        [Header("Model offset (arm-local frame -> model root)")]
        [SerializeField] private Vector3 _modelPositionOffset;
        [SerializeField] private Vector3 _modelEulerOffset;

        [Header("Fallback (YOLO arm axis)")]
        [SerializeField] private bool _useDetectorFallback = true;
        [Tooltip("Seconds after losing the markers before the fallback takes over " +
                 "(prevents flicker on brief occlusions).")]
        [SerializeField] private float _fallbackAfterSeconds = 0.75f;
        [Tooltip("Roll (deg) around the arm axis used by the fallback, since the axis " +
                 "alone can't observe roll. Calibrate for the arm's usual resting pose.")]
        [SerializeField] private float _fallbackRollDegrees;
        [Tooltip("Where the marker-band origin sits along shoulder->wrist (0 = shoulder end).")]
        [SerializeField, Range(0f, 1f)] private float _fallbackBandPosition = 0.2f;

        [Header("Behaviour")]
        [SerializeField] private bool _hideModelWhenLost = true;

        private float _lastMarkerPoseTime = -999f;

        public string Status { get; private set; } = "-";

        private void Update()
        {
            if (_model == null) { Status = "No model assigned"; return; }

            if (_tracker != null && _tracker.TryGetArmPose(out Pose armPose))
            {
                _lastMarkerPoseTime = Time.time;
                Apply(armPose);
                Status = _tracker.IsFrozen ? "markers (frozen)" : "markers";
                return;
            }

            bool graceOver = Time.time - _lastMarkerPoseTime > _fallbackAfterSeconds;
            if (_useDetectorFallback && graceOver && _armManager != null && _armManager.IsLocked &&
                _armManager.TryGetArmEndpoints(out Vector3 shoulder, out Vector3 wrist))
            {
                Vector3 axis = wrist - shoulder;
                if (axis.sqrMagnitude > 1e-4f)
                {
                    Vector3 fwd = axis.normalized; // arm-local +Z points toward the wrist
                    Vector3 anyPerp = Vector3.Cross(fwd, Vector3.up);
                    if (anyPerp.sqrMagnitude < 1e-4f) anyPerp = Vector3.Cross(fwd, Vector3.right);
                    Vector3 up = Quaternion.AngleAxis(_fallbackRollDegrees, fwd) * anyPerp.normalized;
                    var pose = new Pose(
                        Vector3.Lerp(shoulder, wrist, _fallbackBandPosition),
                        Quaternion.LookRotation(fwd, up));
                    Apply(pose);
                    Status = "detector fallback";
                    return;
                }
            }

            if (!graceOver && _lastMarkerPoseTime > 0f)
            {
                Status = "markers (grace)"; // keep last applied transform briefly
                return;
            }

            Status = "lost";
            if (_hideModelWhenLost && _model.gameObject.activeSelf)
                _model.gameObject.SetActive(false);
        }

        private void Apply(Pose armPose)
        {
            if (_hideModelWhenLost && !_model.gameObject.activeSelf)
                _model.gameObject.SetActive(true);
            Quaternion rot = armPose.rotation * Quaternion.Euler(_modelEulerOffset);
            Vector3 pos = armPose.position + armPose.rotation * _modelPositionOffset;
            _model.SetPositionAndRotation(pos, rot);
        }

        [ContextMenu("Capture Model Offset From Current Transform")]
        private void CaptureModelOffset()
        {
            if (_model == null || _tracker == null || !_tracker.TryGetArmPose(out Pose armPose))
            {
                Debug.LogWarning("[TrackedArmModel] Need a tracked pose and an assigned model to calibrate.");
                return;
            }
            Quaternion invRot = Quaternion.Inverse(armPose.rotation);
            _modelPositionOffset = invRot * (_model.position - armPose.position);
            _modelEulerOffset = (invRot * _model.rotation).eulerAngles;
            Debug.Log($"[TrackedArmModel] Captured offset pos={_modelPositionOffset} euler={_modelEulerOffset}. " +
                      "Copy these values out of Play mode (Copy/Paste Component Values).");
        }
    }
}
