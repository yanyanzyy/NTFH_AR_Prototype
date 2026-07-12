using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Draws the tracked needle in AR: a small sphere on the tip, one on the hub/plunger,
    /// and a line between them, coloured green/red by <see cref="NeedleAngleEstimator"/>
    /// acceptability (white when no estimator is wired).
    ///
    /// Port of the Jin-Rui branch's SyringeVisualizer. Differences: world positions come
    /// from ArmDetectionManager's smoothed, depth-projected needle (not a fixed
    /// 0.45 m-from-face guess), the markers are auto-created (no scene-wired spheres),
    /// and only tip + hub are drawn (the manager doesn't track the mid-barrel points).
    /// Their per-keypoint confidence gating is kept: the hub marker hides when its own
    /// keypoint confidence is below the threshold — the Plunger keypoint runs ~0.03-0.05
    /// even on clean detections (Group 2 finding), so at very low values its position is
    /// noise, not data. The line stays visible either way.
    /// </summary>
    public class NeedleVisualizer : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private ArmDetectionManager _manager;
        [SerializeField] private NeedleDetector _detector;
        [SerializeField] private NeedleAngleEstimator _angleEstimator;

        [Header("Appearance")]
        [SerializeField] private float _markerRadiusMeters = 0.008f;
        [SerializeField] private float _lineWidthMeters = 0.005f;

        [Header("Per-Keypoint Confidence")]
        [Tooltip("The hub marker is only shown when the hub keypoint's OWN confidence is at least " +
                 "this. The Plunger keypoint runs ~0.03-0.05 on otherwise-correct detections " +
                 "(tip runs ~0.98), so the default lets typical values through and only hides " +
                 "outright garbage. Raise it if the hub marker jumps around.")]
        [SerializeField, Range(0f, 1f)] private float _keypointConfidenceThreshold = 0.02f;

        private Transform _tipMarker;
        private Transform _hubMarker;
        private LineRenderer _lineRenderer;

        private void Awake()
        {
            _tipMarker = CreateMarker("NeedleTipMarker", Color.cyan);
            _hubMarker = CreateMarker("NeedleHubMarker", Color.yellow);

            _lineRenderer = gameObject.GetComponent<LineRenderer>();
            if (_lineRenderer == null) _lineRenderer = gameObject.AddComponent<LineRenderer>();
            _lineRenderer.positionCount = 2;
            _lineRenderer.startWidth = _lineWidthMeters;
            _lineRenderer.endWidth = _lineWidthMeters;
            _lineRenderer.useWorldSpace = true;
            if (_lineRenderer.sharedMaterial == null)
                _lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            _lineRenderer.enabled = false;
        }

        private Transform CreateMarker(string name, Color color)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.SetParent(transform, false);
            go.transform.localScale = Vector3.one * (_markerRadiusMeters * 2f);
            var collider = go.GetComponent<Collider>();
            if (collider != null) Destroy(collider);
            var renderer = go.GetComponent<Renderer>();
            renderer.material = new Material(Shader.Find("Sprites/Default")) { color = color };
            go.SetActive(false);
            return go.transform;
        }

        private void Update()
        {
            if (_manager == null || !_manager.TryGetNeedle(out var tipPos, out var hubPos))
            {
                _tipMarker.gameObject.SetActive(false);
                _hubMarker.gameObject.SetActive(false);
                _lineRenderer.enabled = false;
                return;
            }

            _tipMarker.gameObject.SetActive(true);
            _tipMarker.position = tipPos;

            bool showHub = true;
            if (_detector != null && _detector.TryGetLatest(out var detection))
                showHub = detection.HubConfidence >= _keypointConfidenceThreshold;
            _hubMarker.gameObject.SetActive(showHub);
            if (showHub) _hubMarker.position = hubPos;

            _lineRenderer.enabled = true;
            _lineRenderer.SetPosition(0, tipPos);
            _lineRenderer.SetPosition(1, hubPos);

            Color lineColor = Color.white;
            if (_angleEstimator != null && _angleEstimator.HasAngle)
                lineColor = _angleEstimator.IsAngleAcceptable ? Color.green : Color.red;
            _lineRenderer.startColor = lineColor;
            _lineRenderer.endColor = lineColor;
        }
    }
}
