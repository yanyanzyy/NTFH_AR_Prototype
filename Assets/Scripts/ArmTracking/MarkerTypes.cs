using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// One detected fiducial marker in the camera image.
    /// Corners are in image pixel coordinates (origin TOP-LEFT, +y down — the same
    /// convention PassthroughCameraSource.ImagePointToRay expects), in ArUco order:
    /// top-left, top-right, bottom-right, bottom-left of the marker's canonical
    /// (upright) orientation.
    /// </summary>
    public struct MarkerObservation
    {
        public int Id;
        public Vector2 C0; // top-left
        public Vector2 C1; // top-right
        public Vector2 C2; // bottom-right
        public Vector2 C3; // bottom-left

        public Vector2 Corner(int i) => i switch
        {
            0 => C0,
            1 => C1,
            2 => C2,
            _ => C3,
        };

        public Vector2 Center => (C0 + C1 + C2 + C3) * 0.25f;
    }

    /// <summary>
    /// Supplies 2D marker corner detections to <see cref="MarkerArmTracker"/>.
    /// Implemented by <see cref="ArUcoCornerProvider"/> (OpenCV for Unity backend);
    /// any custom detector can be plugged in by implementing this on a MonoBehaviour
    /// on the same GameObject as the tracker.
    /// </summary>
    public interface IMarkerCornerProvider
    {
        bool IsReady { get; }
        string Status { get; }

        /// <summary>Detects markers in <paramref name="source"/>. Appends to <paramref name="results"/>. Returns true if detection ran (even with zero markers found).</summary>
        bool TryDetect(Texture source, System.Collections.Generic.List<MarkerObservation> results);
    }
}
