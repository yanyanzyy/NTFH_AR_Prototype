using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
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
    [Tooltip("A sphere is only shown/moved when its OWN keypoint confidence is at least this.")]
    [Range(0f, 1f)]
    [SerializeField] private float _keypointConfidenceThreshold = 0.02f;
    
    private LineRenderer _lineRenderer;
    private Transform _mainCameraTransform;

    void Awake()
    {
        // Safe via RequireComponent attribute guarantee
        _lineRenderer = GetComponent<LineRenderer>();
        _lineRenderer.positionCount = 4;
        _lineRenderer.startWidth = 0.005f; 
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
        if (detector == null) detector = UnityEngine.Object.FindAnyObjectByType<CustomSyringeDetector>();
        if (angleEstimator == null) angleEstimator = UnityEngine.Object.FindAnyObjectByType<SyringeAngleEstimator>();
    }

    void Update()
    {   
        if (detector == null) detector = FindAnyObjectByType<CustomSyringeDetector>();
        if (angleEstimator == null) angleEstimator = FindAnyObjectByType<SyringeAngleEstimator>();
        if (detector == null) return;

        var cameraSource = detector.cameraSourceComponent;
        if (cameraSource == null) return;

        float operationalDepth = CalculateDynamicDepth();

        float modelInputSize = 640f; 

        bool trackingValidLine = detector.IsSyringeDetected;

        for (int j = 0; j < 4; j++)
        {
            if (keyPointLabels[j] == null) continue;
            float[] confidencesArray = (float[])detector.KeypointConfidences;

            // PER-KEYPOINT CONFIDENCE CHECK: Only evaluate and draw if point hits confidence threshold
            bool pointIsReliable = detector.IsSyringeDetected && confidencesArray[j] >= _keypointConfidenceThreshold;
            keyPointLabels[j].gameObject.SetActive(pointIsReliable);

            if (pointIsReliable)
            {
                Vector2 normalizedKpt = detector.NormalizedKeypoints[j];
                float mappedX = normalizedKpt.x * modelInputSize;
                float mappedY = normalizedKpt.y * modelInputSize;

                Vector3 worldPos = cameraSource.ImagePointToWorld(new Vector2(mappedX, mappedY), operationalDepth);
                keyPointLabels[j].position = worldPos;
                _lineRenderer.SetPosition(j, worldPos);
            }
            else
            {
                // If a required layout node drops, we invalidate the action line representation completely
                trackingValidLine = false; 
            }
        }

        if (trackingValidLine)
        {
            _lineRenderer.enabled = true;
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
            if (_lineRenderer != null) _lineRenderer.enabled = false;
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
            if (calculatedDistance > 0.15f) return calculatedDistance;
        }
        return estimatedDistance;
    }
}