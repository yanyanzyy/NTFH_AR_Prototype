using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Coordinates the poke-feedback loop and drives the shared VeinFeedbackUI.
    ///
    /// While the needle (finger/pen or vision syringe) is at the arm:
    ///   • on a vein  → "CORRECT SPOT" plus the insertion-angle verdict
    ///   • off a vein → "WRONG SPOT POKED" plus which way to move (yellow arrow to the vein)
    /// When the needle is away from the arm the panel is hidden.
    ///
    /// TWO MEASUREMENT MODES
    /// ---------------------
    /// • Direct vein proximity (default, <see cref="_useDirectVeinProximity"/>): the needle TIP is
    ///   compared straight to the visible vein PATHS (VeinMap). "Touch the vein you can see →
    ///   CORRECT." This is robust for the finger test because it does NOT depend on the invisible
    ///   arm collision cylinder, which is placed from the vision lock and is often offset from both
    ///   the overlay mesh and the veins (the cause of the sequence getting stuck on "Approach").
    /// • Surface contact (legacy): gates on InjectionSiteDetector.IsContacting, i.e. the tip
    ///   reaching the arm-cylinder surface. Kept for a real registered syringe where the cylinder
    ///   is meaningful.
    ///
    /// Each contact/poke episode is one attempt. Consecutive wrong pokes are counted; after
    /// <see cref="_wrongPokesToReveal"/> in a row the arm overlay answer key is flashed on.
    /// A correct poke resets the streak.
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
        [Tooltip("Optional, PREFERRED — the in-namespace NeedleAngleEstimator, which measures the " +
                 "angle from the manager's needle axis (works for BOTH the vision syringe and the " +
                 "simulated finger/pen needle). When it has an angle it wins over the " +
                 "SyringeAngleEstimator above, whose keypoint spheres only move with the vision model.")]
        [SerializeField] private NeedleAngleEstimator _needleAngleEstimator;
        [Tooltip("Optional — the answer-key overlay flashed on after repeated wrong pokes.")]
        [SerializeField] private ArmOverlay            _armOverlay;

        [Header("Measurement mode")]
        [Tooltip("Drive feedback from the needle TIP's distance to the visible vein paths directly, " +
                 "instead of requiring hard contact with the (often misaligned) arm collision cylinder. " +
                 "Recommended for the finger test: 'touch the vein you can see → CORRECT'.")]
        [SerializeField] private bool  _useDirectVeinProximity = true;
        [Tooltip("Direct mode: show the panel while the tip is within this distance (m) of the nearest " +
                 "vein path. Large enough to give live guidance as the finger nears the arm.")]
        [SerializeField] private float _showWithinMeters = 0.10f;
        [Tooltip("Direct mode: tip within this distance (m) of a vein path counts as an actual poke " +
                 "attempt (drives the wrong-streak / answer-key reveal). A vein's own hitRadius decides " +
                 "CORRECT vs WRONG within that.")]
        [SerializeField] private float _pokeWithinMeters = 0.045f;
        [Tooltip("Direct mode: measure the miss distance perpendicular to the headset's view ray " +
                 "(grade what the trainee SEES). The locked overlay can sit several cm off in depth " +
                 "while looking aligned — raw 3D distance then reports 'wrong spot' on a visually " +
                 "perfect touch. Keep identical to InjectionSequenceEvaluator's setting.")]
        [SerializeField] private bool  _gradeFromView = true;
        [Tooltip("Maximum tip↔vein offset (m) ALONG the view ray for a poke to count. Covers lock " +
                 "registration + hand-tracking depth error.")]
        [SerializeField] private float _viewDepthToleranceMeters = 0.12f;

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
            if (_armManager == null || _veinMap == null || _feedbackUI == null)
            {
                EndEpisodeSilently();
                return;
            }

            if (_useDirectVeinProximity)
            {
                UpdateDirect();
                return;
            }

            if (_injectionDetector == null) { EndEpisodeSilently(); return; }
            UpdateContactBased();
        }

        // ── Direct vein-proximity mode (finger/needle test) ───────────────────────────

        /// <summary>
        /// Measures the needle TIP straight against the visible vein paths — no dependency on the
        /// arm collision cylinder — so touching a vein you can see always registers.
        /// </summary>
        private void UpdateDirect()
        {
            if (!_armManager.IsLocked ||
                !_armManager.TryGetNeedleTip(out var tip) ||
                !QueryVein(tip, out var query))
            {
                CoastOrHide();
                return;
            }

            // View grading: DistanceMeters is the LATERAL miss; the unseen depth offset gates
            // separately (looser for merely showing guidance than for counting a poke).
            bool depthOk = query.ViewDepthMeters <= _viewDepthToleranceMeters;
            bool poking = depthOk && query.DistanceMeters <= _pokeWithinMeters;
            bool near   = query.DistanceMeters <= _showWithinMeters &&
                          query.ViewDepthMeters <= _viewDepthToleranceMeters * 2f;

            if (poking)
            {
                _lossTimer = 0f;
                if (!_episodeActive)        // rising edge: a new poke begins
                {
                    _episodeActive = true;
                    _lastOnVein    = false;
                }
                ShowFeedback(query, tip);
                return;
            }

            // Not poking any more: hold an active episode through the grace, then finalise it.
            if (_episodeActive)
            {
                _lossTimer += Time.deltaTime;
                if (_lossTimer < _contactLossGraceSeconds)
                {
                    ShowFeedback(query, tip);   // keep the panel live during grace
                    return;
                }
                FinaliseEpisode();
                _episodeActive = false;
                _lossTimer     = 0f;
            }

            // Hovering near the arm but not poking: still show live guidance onto the vein.
            if (near)
            {
                ShowFeedback(query, tip);
                return;
            }

            _feedbackUI.Hide();
        }

        /// <summary>Runs the vein query in the configured metric: view-perpendicular (grading what
        /// the trainee sees) when enabled and a camera exists, else raw 3D.</summary>
        private bool QueryVein(Vector3 tip, out VeinMap.QueryResult result)
        {
            var cam = Camera.main;
            if (_gradeFromView && cam != null)
                return _veinMap.QueryNearestVeinFromView(tip, cam.transform.position,
                                                         _viewDepthToleranceMeters, out result);
            return _veinMap.QueryNearestVein(tip, out result);
        }

        /// <summary>Finalises an active poke after the grace window, else hides the panel. Used when
        /// the arm/needle is momentarily unavailable in direct mode.</summary>
        private void CoastOrHide()
        {
            if (_episodeActive)
            {
                _lossTimer += Time.deltaTime;
                if (_lossTimer < _contactLossGraceSeconds) return;
                FinaliseEpisode();
                _episodeActive = false;
                _lossTimer     = 0f;
            }
            _feedbackUI.Hide();
        }

        // ── Legacy surface-contact mode (real registered syringe) ─────────────────────

        private void UpdateContactBased()
        {
            // A "poke" episode is defined by real CONTACT (drives the correct/wrong streak and
            // the answer-key reveal). The panel also shows live guidance the moment the finger is
            // merely NEAR the arm, so the trainee always gets a visible response.
            if (_injectionDetector.IsContacting)
            {
                _lossTimer = 0f;
                if (!_episodeActive)        // rising edge: a new poke begins
                {
                    _episodeActive = true;
                    _lastOnVein    = false;
                }
                UpdateLiveFeedbackFromSurface();
                return;
            }

            if (_episodeActive)
            {
                _lossTimer += Time.deltaTime;
                if (_lossTimer < _contactLossGraceSeconds)
                {
                    UpdateLiveFeedbackFromSurface();
                    return;
                }
                FinaliseEpisode();
                _episodeActive = false;
                _lossTimer     = 0f;
            }

            if (_injectionDetector.IsNearArm)
            {
                UpdateLiveFeedbackFromSurface();
                return;
            }

            _feedbackUI.Hide();
        }

        private void UpdateLiveFeedbackFromSurface()
        {
            if (!_veinMap.QueryNearestVein(_injectionDetector.SurfacePoint, out var query))
            {
                _feedbackUI.Hide();
                return;
            }
            ShowFeedback(query, _injectionDetector.SurfacePoint);
        }

        // ── Shared feedback builder ───────────────────────────────────────────────────

        /// <summary>Builds the feedback panel data for a vein query at <paramref name="injectionPoint"/>
        /// and shows it. Shared by both measurement modes.</summary>
        private void ShowFeedback(VeinMap.QueryResult query, Vector3 injectionPoint)
        {
            if (!_armManager.TryGetArmEndpoints(out var shoulder, out var wrist))
            {
                _feedbackUI.Hide();
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
                InjectionPoint = injectionPoint,
                AlongArmMeters = alongArm,
                CrossArmMeters = crossDelta.magnitude,
                CrossArmRight  = moveRight,
                TotalDistance  = query.DistanceMeters,
            };

            // On a correct spot, follow up with the insertion-angle verdict. Prefer the needle-axis
            // estimator (covers vision AND simulated needles); fall back to Group 2's keypoint-sphere
            // estimator when it's the only one measuring.
            if (query.IsOnVein)
            {
                if (_needleAngleEstimator != null && _needleAngleEstimator.HasAngle)
                {
                    data.HasAngle        = true;
                    data.AngleDegrees    = _needleAngleEstimator.CurrentInsertionAngle;
                    data.AngleAcceptable = _needleAngleEstimator.IsAngleAcceptable;
                    data.AngleMin        = _needleAngleEstimator.MinAcceptableAngle;
                    data.AngleMax        = _needleAngleEstimator.MaxAcceptableAngle;
                }
                else if (_angleEstimator != null)
                {
                    data.HasAngle        = _angleEstimator.HasAngle;
                    data.AngleDegrees    = _angleEstimator.CurrentInsertionAngle;
                    data.AngleAcceptable = _angleEstimator.IsAngleAcceptable;
                    data.AngleMin        = _angleEstimator.MinAcceptableAngle;
                    data.AngleMax        = _angleEstimator.MaxAcceptableAngle;
                }
            }

            _feedbackUI.Show(data);
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
