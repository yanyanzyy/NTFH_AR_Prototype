using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.Android;

public class CustomSyringeDetector : MonoBehaviour
{
    [Header("Model Configuration")]
    [SerializeField] private ModelAsset modelAsset;
    [SerializeField] private BackendType _backend = BackendType.GPUCompute;
    [SerializeField] private int _inputSize = 640;
    [SerializeField, Range(0f, 1f)] private float _confidenceThreshold = 0.15f;

    [Header("Performance")]
    [Tooltip("Throttles how often inference runs (still a blocking GPU readback each time it fires - " +
             "see ParseSyringeOutput - but at least not on literally every Update()). An async " +
             "ReadbackRequest()/IsReadbackRequestDone() version was tried here and reverted: " +
             "IsReadbackRequestDone() never returned true in testing, which froze every detector " +
             "output at its default forever (0 confidence, keypoints stuck at (0,0)) - looked exactly " +
             "like 'detection stopped working'. 0.05 = ~20 inferences/sec max.")]
    [SerializeField] private float _inferenceInterval = 0.05f;

    [Header("Texture Input Source")]
    public ARArmDetection.PassthroughCameraSource cameraSourceComponent;

    private Worker _worker;
    private Tensor<float> _inputTensor;

    // GPU-side letterbox target: the raw camera texture (e.g. 1280x960, 4:3) is aspect-preserved
    // cropped/scaled into this square RT before conversion to a tensor, instead of being
    // squash-stretched directly into 640x640 (which distorts the syringe's proportions relative
    // to what the model was trained on).
    private RenderTexture _processedRT;

    private float _lastInferenceTime;

    // Public properties that the visualizer script can read
    public bool IsSyringeDetected { get; private set; }
    public float HighestConfidence { get; private set; }
    public Vector2[] NormalizedKeypoints { get; private set; } = new Vector2[4];
    /// <summary>Per-keypoint confidence (the 3rd value of each x,y,conf triple), same order as
    /// NormalizedKeypoints. Was previously discarded entirely - exposed so callers (e.g.
    /// SyringeDebugHUD) can tell "unreliable point" apart from "correct point in an odd spot".</summary>
    public float[] KeypointConfidences { get; private set; } = new float[4];

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {   
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
        }

        // Auto-discover the Camera Passthrough source if it wasn't dropped in the inspector
        if (cameraSourceComponent == null)
        {
            cameraSourceComponent = Object.FindAnyObjectByType<ARArmDetection.PassthroughCameraSource>();
        }

        try 
        {
            if (modelAsset != null)
            {
                var model = ModelLoader.Load(modelAsset);
                _worker = new Worker(model, _backend);
                _inputTensor = new Tensor<float>(new TensorShape(1, 3, _inputSize, _inputSize));
                _processedRT = new RenderTexture(_inputSize, _inputSize, 0, RenderTextureFormat.ARGB32);
                _processedRT.Create();
                Debug.Log("CustomSyringeDetector: Neural Engine initialized successfully.");
            }
            else
            {
                Debug.LogError("CustomSyringeDetector: Missing Model Asset! Please attach your .onnx model.");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"CustomSyringeDetector: Neural Engine crashed during initialization! Error details: {e.Message}");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (_worker == null || _inputTensor == null || cameraSourceComponent == null) return;
        if (Time.time - _lastInferenceTime < _inferenceInterval) return;
        _lastInferenceTime = Time.time;

        Texture liveTex = cameraSourceComponent.CurrentTexture;
        if (liveTex == null) return;

        // Aspect-preserving center-crop into the square RT (no stretching), instead of handing
        // the raw (likely non-square) camera texture straight to TextureConverter.ToTensor.
        float textureAspect = (float)liveTex.width / liveTex.height;
        float targetAspect = 1f; // _inputSize x _inputSize is always square

        Vector2 scale = Vector2.one;
        Vector2 offset = Vector2.zero;
        if (textureAspect > targetAspect)
        {
            scale.x = targetAspect / textureAspect;
            offset.x = (1f - scale.x) / 2f;
        }
        else if (textureAspect < targetAspect)
        {
            scale.y = textureAspect / targetAspect;
            offset.y = (1f - scale.y) / 2f;
        }

        Graphics.Blit(liveTex, _processedRT, scale, offset);
        TextureConverter.ToTensor(_processedRT, _inputTensor, new TextureTransform());

        // Execute the model inference
        _worker.Schedule(_inputTensor);

        // Retrieve the output tensor - DownloadToArray() below blocks until the GPU is done and
        // always returns real data, which is why this synchronous form is used instead of the
        // ReadbackRequest()/IsReadbackRequestDone() async pattern (see _inferenceInterval tooltip).
        Tensor<float> outputTensor = _worker.PeekOutput() as Tensor<float>;
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
                float kptConf = predictionData[(kptOffset + 2 + j * 3) * totalAnchors + bestAnchorIndex];

                // Save clean normalized 0.0 to 1.0 coordinates
                NormalizedKeypoints[j] = new Vector2(pixelX / _inputSize, pixelY / _inputSize);
                KeypointConfidences[j] = kptConf;
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

        if (_processedRT != null)
        {
            _processedRT.Release();
            Destroy(_processedRT);
            _processedRT = null;
        }
    }
}
