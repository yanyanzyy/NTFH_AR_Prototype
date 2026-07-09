using UnityEngine;

public class SyringeVisualizer : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private CustomSyringeDetector detector;

    [Header("Visual Indicators")]
    [Tooltip("Assign your 4 colored spheres in physical sequence")]
    [SerializeField] private Transform[] keyPointLabels = new Transform[4];
    
    [Header("Physical Constraints")]
    [Tooltip("Approximate distance in front of headset face (meters)")]
    [SerializeField] private float estimatedDistance = 0.45f;

    [Header("Per-Keypoint Confidence")]
    [Tooltip("A sphere is only shown/moved when its OWN keypoint confidence is at least this - " +
             "replaces the old 'force everything visible' debug override. Point 4 (Plunger) " +
             "specifically runs ~0.03-0.05 even on otherwise-correct detections (0-2 run ~0.98-0.99, " +
             "VPIC finding) - at that confidence its position is closer to noise than data, so hiding " +
             "it is safer than drawing it somewhere misleading. Lower this if you'd rather always see " +
             "all 4 (accepting the 4th may be jittery/wrong) than have it disappear.")]
    [Range(0f, 1f)]
    [SerializeField] private float _keypointConfidenceThreshold = 0.05f;

    // Update is called once per frame
    void Update()
    {
        if (detector == null) return;

        var cameraSource = detector.cameraSourceComponent;
        if (cameraSource == null) return;

        float camWidth = cameraSource.Width > 0 ? cameraSource.Width : 640f;
        float camHeight = cameraSource.Height > 0 ? cameraSource.Height : 640f;

        for (int j = 0; j < 4; j++)
        {
            if (keyPointLabels[j] == null) continue;

            bool shouldShow = detector.IsSyringeDetected && detector.KeypointConfidences[j] >= _keypointConfidenceThreshold;
            keyPointLabels[j].gameObject.SetActive(shouldShow);
            if (!shouldShow) continue;

            // Recover the normalized coordinate pairs
            Vector2 normalizedKpt = detector.NormalizedKeypoints[j];

            // Denormalize into active hardware camera image layout bounds
            float mappedX = normalizedKpt.x * camWidth;
            float mappedY = normalizedKpt.y * camHeight;

            // Project out into true 3D spatial coordinate vectors
            Vector3 worldPos = cameraSource.ImagePointToWorld(new Vector2(mappedX, mappedY), estimatedDistance);

            // Update the position of the corresponding colored sphere
            keyPointLabels[j].position = worldPos;
        }
    }
}
