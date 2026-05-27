using UnityEngine;

namespace ARArmDetection
{
    public class WearerHandFilter : MonoBehaviour
    {
        [Tooltip("World-space transforms of the wearer's own wrists.\n" +
                 "Leave empty when using _wearerSkeletons instead.")]
        [SerializeField] private Transform[] _wearerWristTransforms;

        [Tooltip("Drag the OVRSkeleton components from the Hand Tracking building blocks here " +
                 "(one per hand). The wrist bone is located automatically at runtime — " +
                 "no need to navigate the bone list in the Inspector.")]
        [SerializeField] private OVRSkeleton[] _wearerSkeletons;

        [Tooltip("Image-space radius (pixels) around each wearer wrist within which detections " +
                 "are treated as the wearer's own arm.")]
        [SerializeField] private float _imageRadiusPixels = 120f;

        public bool IsWearerArm(ArmDetection arm, PassthroughCameraSource cameraSource)
        {
            if (cameraSource == null) return false;

            // 1. Explicit transform list (manual / legacy setup).
            if (_wearerWristTransforms != null)
            {
                foreach (var t in _wearerWristTransforms)
                {
                    if (t == null) continue;
                    if (!cameraSource.WorldToImagePoint(t.position, out var proj)) continue;
                    if (Vector2.Distance(proj, arm.WristImage) <= _imageRadiusPixels) return true;
                }
            }

            // 2. Auto-resolve wrist bone from OVRSkeleton at runtime.
            if (_wearerSkeletons != null)
            {
                foreach (var skel in _wearerSkeletons)
                {
                    if (!IsSkeletonReady(skel)) continue;
                    var wristPos = FindWristPosition(skel);
                    if (wristPos == null) continue;
                    if (!cameraSource.WorldToImagePoint(wristPos.Value, out var proj)) continue;
                    if (Vector2.Distance(proj, arm.WristImage) <= _imageRadiusPixels) return true;
                }
            }

            return false;
        }

        // ── Static helpers ─────────────────────────────────────────────────────────────

        private static bool IsSkeletonReady(OVRSkeleton skel)
            => skel != null && skel.IsInitialized && skel.Bones != null && skel.Bones.Count > 0;

        /// <summary>
        /// Finds the Hand_WristRoot bone in the OVRSkeleton and returns its world position,
        /// or null if the bone is not present / not yet initialised.
        /// </summary>
        private static Vector3? FindWristPosition(OVRSkeleton skel)
        {
            foreach (var bone in skel.Bones)
            {
                if (bone == null || bone.Transform == null) continue;
                if (bone.Id == OVRSkeleton.BoneId.Hand_WristRoot)
                    return bone.Transform.position;
            }
            return null;
        }
    }
}
