using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Test rig: uses a tracked FINGERTIP (or a pen held in a pinch) as the syringe needle,
    /// so the whole vein/injection pipeline can be exercised without the unreliable syringe
    /// vision model. BOTH hands are tracked; each frame the fingertip closer to the locked
    /// arm (i.e. the hand actually poking) is chosen, so the rig works no matter which hand
    /// the trainee uses. The tip (plus an optional pen-tip extension) is
    /// pushed into <see cref="ArmDetectionManager.SetSimulatedNeedle"/>; from there it flows
    /// through the manager's single TryGetNeedle/TryGetNeedleTip API, so InjectionSiteDetector,
    /// VeinFeedbackController/UI, InjectionSequenceEvaluator, NeedleAngleEstimator and the
    /// occluder all light up exactly as they would for a vision-detected syringe.
    ///
    /// The syringe detection path is left fully intact — disable this component (or the
    /// manager's override toggle) and the vision needle takes over again.
    ///
    /// BONE IDS: the scene's hands are OpenXR skeletons (OVRSkeleton.SkeletonType.XRHandLeft/
    /// Right), whose bone ids do NOT match the legacy Hand_* ids (e.g. legacy Hand_IndexTip
    /// == 20 == XRHand_RingTip). Resolution therefore branches on GetSkeletonType().
    ///
    /// Runs at execution order 50 so the pose is fed BEFORE ArmDetectionManager (order 100)
    /// consumes it in the same frame.
    /// </summary>
    [DefaultExecutionOrder(50)]
    public class SimulatedNeedleProvider : MonoBehaviour
    {
        public enum Finger { Index, Thumb, Middle, Ring, Pinky }

        [Header("References")]
        [Tooltip("Manager receiving the simulated needle pose. Auto-found when left empty.")]
        [SerializeField] private ArmDetectionManager _manager;
        [Tooltip("PRIMARY hand skeleton (auto-found when empty, preferring the right hand). The " +
                 "OTHER hand is ALSO tracked automatically, and whichever tracked fingertip is " +
                 "closer to the locked arm feeds the needle — so poking with either hand just " +
                 "works, instead of silently grading the idle hand's position.")]
        [SerializeField] private OVRSkeleton _handSkeleton;

        [Header("Simulated needle")]
        [Tooltip("Which fingertip acts as the needle tip. Index reads most naturally; Pinky " +
                 "keeps the pointing finger free for UI.")]
        [SerializeField] private Finger _finger = Finger.Index;
        [Tooltip("Extends the needle tip this far (m) beyond the fingertip along the finger " +
                 "direction. 0 = the bare fingertip IS the tip. ~0.10–0.14 reaches the point of " +
                 "a real pen/needle held in a pinch grip.")]
        [SerializeField, Range(0f, 0.25f)] private float _penTipOffsetMeters = 0f;
        [Tooltip("Distance (m) from the (extended) tip back along the finger direction to the " +
                 "simulated hub. Tip−hub defines the needle axis used for the insertion angle.")]
        [SerializeField, Range(0.03f, 0.25f)] private float _hubBackDistanceMeters = 0.10f;

        [Header("Debug")]
        [Tooltip("Draws a thin hub→tip line so you can see exactly where the simulated needle is.")]
        [SerializeField] private bool _drawMarkerLine = true;
        [SerializeField] private Color _markerColor = new Color(0f, 1f, 0.6f, 0.9f);
        [SerializeField] private float _markerWidthMeters = 0.003f;

        /// <summary>Human-readable state for HUDs ("Feeding (Index)", "Waiting for hand", …).</summary>
        public string Status { get; private set; } = "Idle";

        /// <summary>Per-hand bone cache. Two of these exist so BOTH hands can act as the
        /// needle; the poking hand is picked each frame by proximity to the locked arm.</summary>
        private sealed class HandRig
        {
            public OVRSkeleton Skeleton;
            public Transform   Tip;
            public Transform   Distal;
            public OVRSkeleton ResolvedSkeleton;
            public Finger      ResolvedFinger;
        }

        private readonly HandRig _primaryRig = new HandRig();
        private readonly HandRig _otherRig   = new HandRig();
        private bool _scannedForHands;
        private LineRenderer _markerLine;

        private void OnEnable()
        {
            _primaryRig.Tip = _primaryRig.Distal = null;
            _otherRig.Tip   = _otherRig.Distal   = null;
            _scannedForHands = false;
        }

        private void OnDisable()
        {
            if (_manager != null) _manager.ClearSimulatedNeedle();
            if (_markerLine != null) _markerLine.enabled = false;
            Status = "Disabled";
        }

        private void Update()
        {
            if (!SelfHealReferences())
            {
                if (_markerLine != null) _markerLine.enabled = false;
                return;
            }

            HandRig rig = PickActiveRig();
            if (rig == null)
            {
                Status = "Waiting for hand tracking";
                _manager.ClearSimulatedNeedle();
                if (_markerLine != null) _markerLine.enabled = false;
                return;
            }

            Vector3 tipPos = rig.Tip.position;
            Vector3 dir = tipPos - rig.Distal.position;
            if (dir.sqrMagnitude < 1e-8f)
            {
                Status = "Degenerate finger pose";
                return;
            }
            dir.Normalize();

            Vector3 needleTip = tipPos + dir * _penTipOffsetMeters;
            Vector3 needleHub = needleTip - dir * _hubBackDistanceMeters;

            _manager.SetSimulatedNeedle(needleTip, needleHub);
            Status = $"Feeding ({SideLabel(rig)} {_finger}{(_penTipOffsetMeters > 0.001f ? " + pen" : "")})";

            UpdateMarkerLine(needleHub, needleTip);
        }

        // ── Reference resolution ───────────────────────────────────────────────────────

        private bool SelfHealReferences()
        {
            if (_manager == null)
            {
                _manager = FindFirstObjectByType<ArmDetectionManager>();
                if (_manager == null) { Status = "No ArmDetectionManager"; return false; }
            }

            if (!_scannedForHands)
            {
                // Primary = the explicitly wired skeleton (or the right hand); the OTHER hand is
                // picked up as well so a trainee poking with either hand feeds the needle.
                _primaryRig.Skeleton = _handSkeleton != null ? _handSkeleton
                                                             : FindHandSkeleton(preferRight: true);
                _otherRig.Skeleton = FindOtherHandSkeleton(_primaryRig.Skeleton);

                if (_primaryRig.Skeleton == null && _otherRig.Skeleton == null)
                {
                    Status = "No OVRSkeleton hand found";
                    return false;   // _scannedForHands stays false → rescan next frame
                }

                _scannedForHands = true;
                Debug.Log($"[SimulatedNeedle] Hands: primary=" +
                          $"{(_primaryRig.Skeleton != null ? _primaryRig.Skeleton.GetSkeletonType().ToString() : "none")}, " +
                          $"other={(_otherRig.Skeleton != null ? _otherRig.Skeleton.GetSkeletonType().ToString() : "none")}.");
            }
            return true;
        }

        /// <summary>
        /// Picks which tracked hand feeds the needle this frame: the one whose fingertip is
        /// closer to the locked arm (i.e. the hand actually poking). Falls back to whichever
        /// single hand is tracked, then to the primary when no arm lock exists to compare against.
        /// </summary>
        private HandRig PickActiveRig()
        {
            bool primaryReady = ResolveRig(_primaryRig) && _primaryRig.Skeleton.IsDataValid;
            bool otherReady   = ResolveRig(_otherRig)   && _otherRig.Skeleton.IsDataValid;

            if (primaryReady && otherReady)
            {
                if (_manager.TryGetArmEndpoints(out var s, out var w))
                {
                    Vector3 mid = (s + w) * 0.5f;
                    return (mid - _primaryRig.Tip.position).sqrMagnitude
                        <= (mid - _otherRig.Tip.position).sqrMagnitude ? _primaryRig : _otherRig;
                }
                return _primaryRig;
            }
            if (primaryReady) return _primaryRig;
            if (otherReady) return _otherRig;
            return null;
        }

        private static string SideLabel(HandRig rig)
        {
            var t = rig.Skeleton.GetSkeletonType();
            bool right = t == OVRSkeleton.SkeletonType.HandRight ||
                         t == OVRSkeleton.SkeletonType.XRHandRight;
            return right ? "R" : "L";
        }

        private static OVRSkeleton FindOtherHandSkeleton(OVRSkeleton primary)
        {
            foreach (var skel in FindObjectsByType<OVRSkeleton>(FindObjectsSortMode.None))
            {
                if (skel == primary) continue;
                var t = skel.GetSkeletonType();
                bool isHand = t == OVRSkeleton.SkeletonType.HandRight ||
                              t == OVRSkeleton.SkeletonType.XRHandRight ||
                              t == OVRSkeleton.SkeletonType.HandLeft ||
                              t == OVRSkeleton.SkeletonType.XRHandLeft;
                if (isHand) return skel;
            }
            return null;
        }

        private static OVRSkeleton FindHandSkeleton(bool preferRight)
        {
            OVRSkeleton fallback = null;
            foreach (var skel in FindObjectsByType<OVRSkeleton>(FindObjectsSortMode.None))
            {
                var type = skel.GetSkeletonType();
                bool isRight = type == OVRSkeleton.SkeletonType.HandRight ||
                               type == OVRSkeleton.SkeletonType.XRHandRight;
                bool isLeft = type == OVRSkeleton.SkeletonType.HandLeft ||
                              type == OVRSkeleton.SkeletonType.XRHandLeft;
                if (!isRight && !isLeft) continue;
                if (isRight == preferRight) return skel;
                fallback = skel;
            }
            return fallback;
        }

        /// <summary>
        /// Resolves the tip + distal-phalanx bones of the configured finger for ONE hand rig,
        /// branching on the skeleton type (legacy Hand_* ids vs OpenXR XRHand_* ids share
        /// integer values but mean DIFFERENT bones). Re-resolves when the skeleton or finger
        /// selection changes.
        /// </summary>
        private bool ResolveRig(HandRig rig)
        {
            if (rig.Skeleton == null) return false;
            if (rig.Tip != null && rig.Distal != null &&
                rig.ResolvedSkeleton == rig.Skeleton && rig.ResolvedFinger == _finger)
                return true;

            rig.Tip = null;
            rig.Distal = null;
            if (!rig.Skeleton.IsInitialized ||
                rig.Skeleton.Bones == null || rig.Skeleton.Bones.Count == 0)
                return false;

            GetFingerBoneIds(rig.Skeleton.GetSkeletonType(), _finger,
                             out var tipId, out var distalId);

            foreach (var bone in rig.Skeleton.Bones)
            {
                if (bone == null || bone.Transform == null) continue;
                if (bone.Id == tipId) rig.Tip = bone.Transform;
                else if (bone.Id == distalId) rig.Distal = bone.Transform;
            }

            if (rig.Tip == null || rig.Distal == null) return false;

            rig.ResolvedSkeleton = rig.Skeleton;
            rig.ResolvedFinger = _finger;
            Debug.Log($"[SimulatedNeedle] Resolved {_finger} bones on " +
                      $"{rig.Skeleton.GetSkeletonType()} (tip={tipId}, distal={distalId}).");
            return true;
        }

        private static void GetFingerBoneIds(OVRSkeleton.SkeletonType type, Finger finger,
                                             out OVRSkeleton.BoneId tip, out OVRSkeleton.BoneId distal)
        {
            bool xr = type == OVRSkeleton.SkeletonType.XRHandLeft ||
                      type == OVRSkeleton.SkeletonType.XRHandRight;
            if (xr)
            {
                switch (finger)
                {
                    case Finger.Thumb:
                        tip = OVRSkeleton.BoneId.XRHand_ThumbTip;
                        distal = OVRSkeleton.BoneId.XRHand_ThumbDistal;
                        return;
                    case Finger.Middle:
                        tip = OVRSkeleton.BoneId.XRHand_MiddleTip;
                        distal = OVRSkeleton.BoneId.XRHand_MiddleDistal;
                        return;
                    case Finger.Ring:
                        tip = OVRSkeleton.BoneId.XRHand_RingTip;
                        distal = OVRSkeleton.BoneId.XRHand_RingDistal;
                        return;
                    case Finger.Pinky:
                        tip = OVRSkeleton.BoneId.XRHand_LittleTip;
                        distal = OVRSkeleton.BoneId.XRHand_LittleDistal;
                        return;
                    default:
                        tip = OVRSkeleton.BoneId.XRHand_IndexTip;
                        distal = OVRSkeleton.BoneId.XRHand_IndexDistal;
                        return;
                }
            }

            switch (finger)
            {
                case Finger.Thumb:
                    tip = OVRSkeleton.BoneId.Hand_ThumbTip;
                    distal = OVRSkeleton.BoneId.Hand_Thumb3;
                    return;
                case Finger.Middle:
                    tip = OVRSkeleton.BoneId.Hand_MiddleTip;
                    distal = OVRSkeleton.BoneId.Hand_Middle3;
                    return;
                case Finger.Ring:
                    tip = OVRSkeleton.BoneId.Hand_RingTip;
                    distal = OVRSkeleton.BoneId.Hand_Ring3;
                    return;
                case Finger.Pinky:
                    tip = OVRSkeleton.BoneId.Hand_PinkyTip;
                    distal = OVRSkeleton.BoneId.Hand_Pinky3;
                    return;
                default:
                    tip = OVRSkeleton.BoneId.Hand_IndexTip;
                    distal = OVRSkeleton.BoneId.Hand_Index3;
                    return;
            }
        }

        // ── Marker line ────────────────────────────────────────────────────────────────

        private void UpdateMarkerLine(Vector3 hub, Vector3 tip)
        {
            if (!_drawMarkerLine)
            {
                if (_markerLine != null) _markerLine.enabled = false;
                return;
            }

            if (_markerLine == null)
            {
                var go = new GameObject("SimulatedNeedleMarker");
                go.transform.SetParent(transform, false);
                _markerLine = go.AddComponent<LineRenderer>();
                _markerLine.useWorldSpace = true;
                _markerLine.positionCount = 2;
                _markerLine.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                _markerLine.receiveShadows = false;
                var shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color");
                _markerLine.material = new Material(shader) { color = _markerColor };
            }

            _markerLine.enabled = true;
            _markerLine.startWidth = _markerWidthMeters;
            _markerLine.endWidth = _markerWidthMeters * 0.4f;   // taper toward the tip
            _markerLine.SetPosition(0, hub);
            _markerLine.SetPosition(1, tip);
        }
    }
}
