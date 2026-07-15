using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Coordinates the poke-feedback loop and drives the shared VeinFeedbackUI.
    ///
    /// While the needle is actually TOUCHING the arm (InjectionSiteDetector.IsContacting):
    ///   • on a vein  → "CORRECT SPOT" plus the insertion-angle verdict (SyringeAngleEstimator)
    ///   • off a vein → "WRONG SPOT POKED" plus which way to move
    /// When nothing is being poked the panel is hidden.
    ///
    /// Each contact episode is one "poke". Consecutive wrong pokes are counted; after
    /// <see cref="_wrongPokesToReveal"/> in a row the arm overlay (the hidden "answer key")
    /// is flashed on for <see cref="_overlayRevealSeconds"/> so the trainee can see where the
    /// veins really are. A correct poke resets the streak.
    /// </summary>
    public class VeinFeedbackController : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private InjectionSiteDetector _injectionDetector;
        [SerializeField] private VeinMap               _veinMap;
        [SerializeField] private VeinFeedbackUI        _feedbackUI;
        [SerializeField] private ArmDetectionManager   _armManager;
        [Tooltip("Optional — Group 2's SyringeAngleEstimator (under SyringePosePrototype / " +
                 "SyringeLabelContainer). Supplies the insertion-angle verdict shown after a correct spot.")]
        [SerializeField] private SyringeAngleEstimator _angleEstimator;
        [Tooltip("Optional — the answer-key overlay flashed on after repeated wrong pokes.")]
        [SerializeField] private ArmOverlay            _armOverlay;

        [Header("Wrong-poke guidance")]
        [Tooltip("Consecutive wrong-spot pokes before the overlay answer key is revealed.")]
        [SerializeField] private int   _wrongPokesToReveal   = 3;
        [Tooltip("How long the overlay answer key stays visible after the streak is hit (seconds).")]
        [SerializeField] private float _overlayRevealSeconds = 5f;
        [Tooltip("A poke ends only after contact has been lost for this long — bridges vision " +
                 "jitter so one jab is counted as one poke, not several.")]
        [SerializeField] private float _contactLossGraceSeconds = 0.3f;

        // ── State ─────────────────────────────────────────────────────────────────────

        private bool  _episodeActive;
        private bool  _lastOnVein;
        private float _lossTimer;
        private int   _wrongStreak;

        // ── Unity lifecycle ───────────────────────────────────────────────────────────

        private void OnDisable()
        {
            _feedbackUI?.Hide();
            _episodeActive = false;
            _lossTimer     = 0f;
        }

        private void Update()
        {
            if (_injectionDetector == null || _armManager == null)
            {
                EndEpisodeSilently();
                return;
            }

            if (_injectionDetector.IsContacting)
            {
                _lossTimer = 0f;
                if (!_episodeActive)        // rising edge: a new poke begins
                {
                    _episodeActive = true;
                    _lastOnVein    = false;
                }
                UpdateLiveFeedback();
                return;
            }

            // Not contacting: hold the episode open through a short grace, then finalise it.
            if (_episodeActive)
            {
                _lossTimer += Time.deltaTime;
                if (_lossTimer < _contactLossGraceSeconds) return;   // still within grace — keep last UI

                FinaliseEpisode();
                _episodeActive = false;
                _lossTimer     = 0f;
            }
            _feedbackUI?.Hide();
        }

        // ── Live feedback while touching the arm ──────────────────────────────────────

        private void UpdateLiveFeedback()
        {
            if (_veinMap == null
                || !_veinMap.QueryNearestVein(_injectionDetector.SurfacePoint, out var query)
                || !_armManager.TryGetArmEndpoints(out var shoulder, out var wrist))
            {
                _feedbackUI?.Hide();
                return;
            }

            _lastOnVein = query.IsOnVein;

            // Decompose the injection→vein delta into along-arm and cross-arm guidance.
            Vector3 armAxis    = (wrist - shoulder).normalized;
            float   alongArm   = Vector3.Dot(query.Delta, armAxis);
            Vector3 crossDelta = query.Delta - armAxis * alongArm;

            Vector3 worldRef  = Mathf.Abs(Vector3.Dot(armAxis, Vector3.up)) < 0.99f
                              ? Vector3.up : Vector3.forward;
            Vector3 armRight  = Vector3.Cross(armAxis, worldRef).normalized;
            bool    moveRight = Vector3.Dot(crossDelta.normalized, armRight) >= 0f;

            var data = new VeinFeedbackUI.FeedbackData
            {
                IsOnVein       = query.IsOnVein,
                VeinName       = query.Vein.name,
                VeinWorldPos   = query.VeinWorldPos,
                InjectionPoint = _injectionDetector.SurfacePoint,
                AlongArmMeters = alongArm,
                CrossArmMeters = crossDelta.magnitude,
                CrossArmRight  = moveRight,
                TotalDistance  = query.DistanceMeters,
            };

            // On a correct spot, follow up with the insertion-angle verdict.
            if (query.IsOnVein && _angleEstimator != null)
            {
                data.HasAngle        = _angleEstimator.HasAngle;
                data.AngleDegrees    = _angleEstimator.CurrentInsertionAngle;
                data.AngleAcceptable = _angleEstimator.IsAngleAcceptable;
                data.AngleMin        = _angleEstimator.MinAcceptableAngle;
                data.AngleMax        = _angleEstimator.MaxAcceptableAngle;
            }

            _feedbackUI?.Show(data);
        }

        // ── Episode bookkeeping ───────────────────────────────────────────────────────

        private void FinaliseEpisode()
        {
            if (_lastOnVein)
            {
                _wrongStreak = 0;          // a correct poke breaks the streak
                return;
            }

            _wrongStreak++;
            if (_wrongStreak >= Mathf.Max(1, _wrongPokesToReveal))
            {
                _armOverlay?.RevealFor(_overlayRevealSeconds);   // show the answer key
                _wrongStreak = 0;
            }
        }

        private void EndEpisodeSilently()
        {
            _episodeActive = false;
            _lossTimer     = 0f;
            _feedbackUI?.Hide();
        }
    }
}
