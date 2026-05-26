using UnityEngine;

namespace ARArmDetection
{
    public class WearerHandFilter : MonoBehaviour
    {
        [Tooltip("World-space transforms of the wearer's own wrists. Assign OVRSkeleton wrist bones, or leave empty to disable filtering.")]
        [SerializeField] private Transform[] _wearerWristTransforms;

        [Tooltip("Image-space radius (pixels) around each wearer wrist within which detections are treated as the wearer's own.")]
        [SerializeField] private float _imageRadiusPixels = 120f;

        public bool IsWearerArm(ArmDetection arm, PassthroughCameraSource cameraSource)
        {
            if (cameraSource == null || _wearerWristTransforms == null) return false;

            for (int i = 0; i < _wearerWristTransforms.Length; i++)
            {
                var t = _wearerWristTransforms[i];
                if (t == null) continue;
                if (!cameraSource.WorldToImagePoint(t.position, out var proj)) continue;
                if (Vector2.Distance(proj, arm.WristImage) <= _imageRadiusPixels)
                    return true;
            }
            return false;
        }
    }
}
