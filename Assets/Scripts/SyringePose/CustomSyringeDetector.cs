using UnityEngine;
using Unity.InferenceEngine;


public class CustomSyringeDetector : MonoBehaviour
{
    [Header("Model Configuration")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private BackendType _backend = BackendType.GPUCompute;
    [SerializeField] private int _inputSize = 640; 
    [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.15f;

    [Header("Texture Input Source")]
    public ARArmDetection.PassthroughCameraSource cameraSourceComponent; 

    private Worker _worker;
    private Tensor<float> _inputTensor; 

    // Public properties that the visualizer script can read
    public bool IsSyringeDetected { get; private set; }
    public float HighestConfidence { get; private set; }
    public Vector2[] NormalizedKeypoints { get; private set; } = new Vector2[4];

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // Load the neural network model
        if (modelAsset != null)
        {
            var model = ModelLoader.Load(modelAsset);
            _worker = new Worker(model, _backend);
            _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));
        }
        else
        {
            Debug.LogError("CustomSyringeDetector: Missing Model Asset! Please attach your .onnx model.");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (_worker == null || _inputTensor == null || cameraSourceComponent == null) return;

        // Get the current frame from the camera source
        Texture liveTex = cameraSourceComponent.CurrentTexture;
        if (liveTex == null) return;

        // Convert the image texture to Tensor<float> format
        TextureConverter.ToTensor(liveTex, _inputTensor);
        // Execute the model inference
        _worker.Schedule(_inputTensor);

        // Retrieve the output tensor
        Tensor<float> outputTensor =  (Tensor<float>)_worker.PeekOutput();
        if (outputTensor != null)
        {
            ParseSyringeOutput(outputTensor);
        }
    }

    private void ParseSyringeOutput(Tensor<float> output)
    {
        // Assume model outputs a flat array of coordinates: 
        float[] predictionData = output.DownloadToArray();

        int totalAnchors = output.shape[2]; // 8400
        
        int bestAnchorIndex = 0;
        HighestConfidence = -1f;

        for (int i = 0; i < totalAnchors; i++)
        {
            float confidence = predictionData[4 * totalAnchors + i]; // Feature index 4 is object confidence
            if (confidence > HighestConfidence)
            {
                HighestConfidence = confidence;
                bestAnchorIndex = i;
            }
        }

        if (HighestConfidence > _confidenceThreshold) // Threshold for detection
        {
            IsSyringeDetected = true;
            int kptOffset = 5; // Keypoints start at feature index 5

            for (int j = 0; j < 4; j++)
            {
                // Extract using features-first stride math
                float pixelX = predictionData[(kptOffset + j * 3) * totalAnchors + bestAnchorIndex];
                float pixelY = predictionData[(kptOffset + 1 + j * 3) * totalAnchors + bestAnchorIndex];

                // Save clean normalized 0.0 to 1.0 coordinates
                NormalizedKeypoints[j] = new Vector2(pixelX / _inputSize, pixelY / _inputSize);
            }
        }
        else
        {
            IsSyringeDetected = false;
        }
    }

    private void OnDisable()
    {
        // Clean up allocation pools identically to Group 1
        _worker?.Dispose();
        _worker = null;
        _inputTensor?.Dispose();
        _inputTensor = null;
    }
}
