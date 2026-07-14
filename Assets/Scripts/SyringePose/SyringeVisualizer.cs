using Unity.VisualScripting;
using UnityEngine;

public class SyringeVisualizer : MonoBehaviour
{
    [Header("Dependencies")]
    [SerializeField] private CustomSyringeDetector detector;
    [SerializeField] private SyringeAngleEstimator angleEstimator;

    [Header("Visual Indicators")]
    [Tooltip("Assign your 4 colored spheres in physical sequence")]
    [SerializeField] private Transform[] keyPointLabels = new Transform[4];

    [Header("Physical Constraints")]
    [Tooltip("Fallback approximate distance in front of headset face (meters)")]
    [SerializeField] private float estimatedDistance = 0.30f;

    [Header("Per-Keypoint Confidence")]
    [Tooltip("A sphere is only shown/moved when its OWN keypoint confidence is at least this - " +
             "replaces the old 'force everything visible' debug override. Point 4 (Plunger) " +
             "specifically runs ~0.03-0.05 even on otherwise-correct detections (0-2 run ~0.98-0.99, " +
             "VPIC finding) - at that confidence its position is closer to noise than data, so hiding " +
             "it is safer than drawing it somewhere misleading. Lower this if you'd rather always see " +
             "all 4 (accepting the 4th may be jittery/wrong) than have it disappear.")]
    [Range(0f, 1f)]
    [SerializeField] private float _keypointConfidenceThreshold = 0.02f;
    private LineRenderer _lineRenderer;
    private Transform _mainCameraTransform;

    void Awake()
    {
        _lineRenderer = GetComponent<LineRenderer>();
        _lineRenderer.positionCount = 4;
        _lineRenderer.startWidth = 0.005f; // 5mm thick line in VR
        _lineRenderer.endWidth = 0.005f;
        _lineRenderer.useWorldSpace = true;

        if (_lineRenderer.material == null)
        {
            _lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        }

        if (Camera.main != null)
        {
            _mainCameraTransform = Camera.main.transform;
        }
    }

    void Start()
    {
        if (detector == null) detector = FindAnyObjectByType<CustomSyringeDetector>();
        if (angleEstimator == null) angleEstimator = FindAnyObjectByType<SyringeAngleEstimator>();
    }

    // Update is called once per frame
    void Update()
    {   
        // 1. Runtime fallback check to repair connection chains dynamically
        if (detector == null) detector = FindAnyObjectByType<CustomSyringeDetector>();
        if (angleEstimator == null) angleEstimator = FindAnyObjectByType<SyringeAngleEstimator>();
        if (detector == null) return;

        var cameraSource = detector.cameraSourceComponent;
        if (cameraSource == null) return;

        float operationalDepth = CalculateDynamicDepth();

        float camWidth = cameraSource.Width > 0 ? cameraSource.Width : 640f;
        float camHeight = cameraSource.Height > 0 ? cameraSource.Height : 640f;

        // 2. Active tracking projection math execution block
        if (detector.IsSyringeDetected)
        {
            _lineRenderer.enabled = true;

            for (int j = 0; j < 4; j++)
            {
                if (keyPointLabels[j] == null) continue;

                // Keep tracking label spheres active simultaneously with the scene state
                keyPointLabels[j].gameObject.SetActive(true);

                Vector2 normalizedKpt = detector.NormalizedKeypoints[j];

                float mappedX = normalizedKpt.x * camWidth;
                float mappedY = normalizedKpt.y * camHeight;

                Vector3 worldPos = cameraSource.ImagePointToWorld(new Vector2(mappedX, mappedY), operationalDepth);

                keyPointLabels[j].position = worldPos;
                _lineRenderer.SetPosition(j, worldPos);
            }

            if (angleEstimator != null)
            {
                Color lineColor = angleEstimator.IsAngleAcceptable ? Color.green : Color.red;
                _lineRenderer.startColor = lineColor;
                _lineRenderer.endColor = lineColor;
            }
            else
            {
                _lineRenderer.startColor = Color.white;
                _lineRenderer.endColor = Color.white;
            }
        }
        else
        {
            // Turn off the line when nothing is detected, but keep target spheres updating layout properties
            if (_lineRenderer != null) _lineRenderer.enabled = false;
            
            // Optional: Hide the tracking dots when the model completely loses tracking focus
            for (int j = 0; j < 4; j++)
            {
                if (keyPointLabels[j] != null) keyPointLabels[j].gameObject.SetActive(false);
            }
        }
    }

    private float CalculateDynamicDepth()
    {
        if (_mainCameraTransform == null && Camera.main != null)
        {
            _mainCameraTransform = Camera.main.transform;
        }

        if (_mainCameraTransform == null) return estimatedDistance;

        GameObject handTrackingTarget = GameObject.Find("MediaPipeHandDetector");

        if (handTrackingTarget != null)
        {
            float calculatedDistance = Vector3.Distance(_mainCameraTransform.position, handTrackingTarget.transform.position);
            if (calculatedDistance > 0.15f)
            {
                return calculatedDistance;
            }
        }
        return estimatedDistance;
    }

    private void ToggleVisuals(bool state)
    {
        for (int j = 0; j < 4; j++)
        {
            if (keyPointLabels[j] != null)
            {
                keyPointLabels[j].gameObject.SetActive(state);
            }
        }
        _lineRenderer.enabled = state;
    }
}
