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

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        return;
    }

    // Update is called once per frame
    void Update()
    {
        if (detector == null) return;

        // TEMPORARY DEBUG OVERRIDE: 
        // This forces the spheres to stay turned ON permanently so you can find them!
        ToggleLabels(true); 

        var cameraSource = detector.cameraSourceComponent;
        if (cameraSource == null) return;

        float camWidth = cameraSource.Width > 0 ? cameraSource.Width : 640f;
        float camHeight = cameraSource.Height > 0 ? cameraSource.Height : 640f;

        for (int j = 0; j < 4; j++)
        {
            if (keyPointLabels[j] != null)
            {
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

    private void ToggleLabels(bool visible)
    {
        foreach (var label in keyPointLabels)
        {
            if (label != null) label.gameObject.SetActive(visible);
        }
    }
}
