using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Connects InjectionSiteDetector with VeinMap to produce directional feedback.
    ///
    /// Decomposes the world-space delta (injection site → nearest vein) into two
    /// clinically meaningful components:
    ///
    ///   Along-arm  — how far to move toward the elbow or toward the wrist
    ///   Cross-arm  — how far to move laterally (left / right when looking down at the arm)
    ///
    /// The results are handed to VeinFeedbackUI each frame for display.
    /// </summary>
    public class VeinFeedbackController : MonoBehaviour
    {
        [SerializeField] private InjectionSiteDetector _injectionDetector;
        [SerializeField] private VeinMap               _veinMap;
        [SerializeField] private VeinFeedbackUI        _feedbackUI;
        [SerializeField] private ArmDetectionManager   _armManager;

        // ── Unity lifecycle ───────────────────────────────────────────────────────────

        private void OnEnable()
        {
            if (_injectionDetector == null) return;
            _injectionDetector.OnInjectionStarted += HandleInjection;
            _injectionDetector.OnInjectionUpdated += HandleInjection;
            _injectionDetector.OnInjectionEnded   += HandleInjectionEnded;
        }

        private void OnDisable()
        {
            if (_injectionDetector == null) return;
            _injectionDetector.OnInjectionStarted -= HandleInjection;
            _injectionDetector.OnInjectionUpdated -= HandleInjection;
            _injectionDetector.OnInjectionEnded   -= HandleInjectionEnded;
        }

        // ── Event handlers ────────────────────────────────────────────────────────────

        private void HandleInjection(Vector3 injectionPoint)
        {
            if (!_veinMap.QueryNearestVein(injectionPoint, out var query)) return;
            if (!_armManager.TryGetArmEndpoints(out var shoulder, out var wrist)) return;

            // ── Decompose delta into along-arm and cross-arm components ───────────────
            Vector3 armAxis    = (wrist - shoulder).normalized;

            // Positive = must move toward wrist; negative = must move toward shoulder/elbow
            float   alongArm   = Vector3.Dot(query.Delta, armAxis);

            // Perpendicular component (lateral offset)
            Vector3 crossDelta = query.Delta - armAxis * alongArm;
            float   crossArm   = crossDelta.magnitude;

            // Determine left/right by projecting onto the arm's inherent right vector
            Vector3 worldRef   = Mathf.Abs(Vector3.Dot(armAxis, Vector3.up)) < 0.99f
                               ? Vector3.up : Vector3.forward;
            Vector3 armRight   = Vector3.Cross(armAxis, worldRef).normalized;
            bool    moveRight  = Vector3.Dot(crossDelta.normalized, armRight) >= 0f;

            _feedbackUI?.Show(new VeinFeedbackUI.FeedbackData
            {
                IsOnVein       = query.IsOnVein,
                VeinName       = query.Vein.name,
                VeinWorldPos   = query.VeinWorldPos,
                InjectionPoint = injectionPoint,
                AlongArmMeters = alongArm,
                CrossArmMeters = crossArm,
                CrossArmRight  = moveRight,
                TotalDistance  = query.DistanceMeters,
            });
        }

        private void HandleInjectionEnded()
        {
            _feedbackUI?.Hide();
        }
    }
}
