using UnityEngine;

namespace ARArmDetection
{
    public enum CocoKeypoint
    {
        Nose = 0,
        LeftEye = 1, RightEye = 2,
        LeftEar = 3, RightEar = 4,
        LeftShoulder = 5, RightShoulder = 6,
        LeftElbow = 7, RightElbow = 8,
        LeftWrist = 9, RightWrist = 10,
        LeftHip = 11, RightHip = 12,
        LeftKnee = 13, RightKnee = 14,
        LeftAnkle = 15, RightAnkle = 16,
    }

    public struct Keypoint
    {
        public Vector2 ImagePos;
        public float Confidence;
    }

    public struct PersonDetection
    {
        public Rect ImageBounds;
        public float Confidence;
        public Keypoint[] Keypoints;
    }

    public enum Side { Left, Right }

    public struct ArmDetection
    {
        public Side Side;
        public Vector2 ShoulderImage;
        public Vector2 ElbowImage;
        public Vector2 WristImage;
        public float Confidence;
    }

    public struct NeedleDetection
    {
        public Rect ImageBounds;
        public float Confidence;
        public Vector2 TipImage;      // kpt0 = NeedleTip (contact point)
        public Vector2 HubImage;      // last kpt = hub/Plunger
        public float TipConfidence;   // per-keypoint visibility score (tip runs ~0.98)
        public float HubConfidence;   // plunger runs ~0.03-0.05 even on clean detections
    }
}
