using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Runs YOLO11n-pose inference and returns PersonDetection objects.
    ///
    /// ARM-ONLY FALLBACK
    /// -----------------
    /// YOLO is trained on full-body COCO images and requires a visible torso to fire
    /// its person-confidence head with useful scores. An isolated arm (e.g. mannequin)
    /// will often score near 0 on person confidence even if the arm keypoints are fine.
    ///
    /// When _armOnlyFallback is enabled, a second pass scans all 8400 raw anchors
    /// directly for high shoulder + wrist keypoint visibility, ignoring person confidence
    /// entirely. This lets the system detect an arm with no torso or context.
    /// </summary>
    public class YoloPoseDetector : MonoBehaviour
    {
        public enum Backend { GPUCompute, GPUPixel, CPU }

        [Header("Model")]
        [SerializeField] private Unity.InferenceEngine.ModelAsset _modelAsset;
        [SerializeField] private Backend _backend = Backend.GPUCompute;
        [SerializeField] private int _inputSize = 640;

        [Header("Full-body detection")]
        [Range(0f, 1f)][SerializeField] private float _confidenceThreshold = 0.25f;
        [Range(0f, 1f)][SerializeField] private float _nmsIoUThreshold = 0.45f;
        [Range(0f, 1f)][SerializeField] private float _keypointConfidenceThreshold = 0.15f;
        [SerializeField] private int _maxDetections = 16;

        [Header("Arm-only fallback (for mannequins / isolated arms)")]
        [Tooltip("When the full-body pass finds 0 persons, scan raw YOLO anchors for just arm keypoints.\n" +
                 "Enables detection of an isolated arm with no torso visible.")]
        [SerializeField] private bool _armOnlyFallback = true;
        [Tooltip("Minimum visibility for both shoulder AND wrist to count as an arm-only candidate.\n" +
                 "Too low fires on noise anchors (ghost boxes with nothing in view). Compare against " +
                 "MaxArmKP in the debug HUD: set this comfortably above the empty-scene value and " +
                 "below the value seen when the real arm is in frame.")]
        [Range(0f, 1f)][SerializeField] private float _armOnlyKeypointThreshold = 0.25f;
        [Tooltip("Minimum shoulder→wrist distance in model pixels for an arm-only candidate. " +
                 "Rejects degenerate noise anchors whose keypoints collapse to a point.")]
        [SerializeField] private float _armOnlyMinArmPixels = 50f;

        // ── Runtime ────────────────────────────────────────────────────────────────────

        private Unity.InferenceEngine.Model  _runtimeModel;
        private Unity.InferenceEngine.Worker _worker;
        private RenderTexture                _resizeTarget;

        private readonly List<PersonDetection> _scratchDetections = new();
        private readonly List<int>             _scratchIndices    = new();

        public bool  IsReady                     => _worker != null;
        public float KeypointConfidenceThreshold => _keypointConfidenceThreshold;
        public float ConfidenceThreshold         => _confidenceThreshold;
        /// <summary>True when the last Run() result came from the arm-only fallback path.</summary>
        public bool  LastRunWasArmOnlyFallback   { get; private set; }
        /// <summary>
        /// The highest min(shoulder_v, wrist_v) seen across ALL anchors in the last
        /// arm-only scan. Compare this against _armOnlyKeypointThreshold to understand
        /// whether the threshold needs lowering. 0 if the arm-only pass did not run.
        /// </summary>
        public float LastArmOnlyMaxScore         { get; private set; }

        /// <summary>
        /// When true, skips the full-body person-confidence pass entirely and goes
        /// straight to the arm-keypoint scan. Set this at runtime via the mode button
        /// for mannequin / isolated-arm scenarios.
        /// </summary>
        public bool ArmOnlyMode { get; set; } = false;

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void OnEnable()
        {
            if (_modelAsset == null)
            {
                Debug.LogError("[YoloPoseDetector] No ModelAsset assigned. Drag yolo11n-pose.onnx here.");
                return;
            }
            _runtimeModel = Unity.InferenceEngine.ModelLoader.Load(_modelAsset);
            _worker = new Unity.InferenceEngine.Worker(_runtimeModel, ResolveBackendType());
        }

        private void OnDisable()
        {
            _worker?.Dispose();
            _worker = null;
            if (_resizeTarget != null)
            {
                _resizeTarget.Release();
                Destroy(_resizeTarget);
                _resizeTarget = null;
            }
        }

        private Unity.InferenceEngine.BackendType ResolveBackendType() => _backend switch
        {
            Backend.GPUCompute => Unity.InferenceEngine.BackendType.GPUCompute,
            Backend.GPUPixel   => Unity.InferenceEngine.BackendType.GPUPixel,
            _                  => Unity.InferenceEngine.BackendType.CPU,
        };

        // ── Resize target ──────────────────────────────────────────────────────────────

        private void EnsureResizeTarget()
        {
            if (_resizeTarget != null && _resizeTarget.width == _inputSize) return;
            if (_resizeTarget != null) { _resizeTarget.Release(); Destroy(_resizeTarget); }
            _resizeTarget = new RenderTexture(_inputSize, _inputSize, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = false,
                useMipMap         = false,
                autoGenerateMips  = false,
                wrapMode          = TextureWrapMode.Clamp,
                filterMode        = FilterMode.Bilinear,
            };
            _resizeTarget.Create();
        }

        // ── Public API ─────────────────────────────────────────────────────────────────

        public List<PersonDetection> Run(Texture cameraTexture)
        {
            _scratchDetections.Clear();
            LastRunWasArmOnlyFallback = false;
            if (!IsReady || cameraTexture == null) return _scratchDetections;

            try
            {
                // Inference Engine 2.x: pre-allocate the input tensor at the model's
                // input dimensions, then let TextureConverter do source→input resize
                // internally. Following the official Meta sample
                // (Unity-PassthroughCameraApiSamples / SentisInferenceRunManager.cs):
                //   var tt = new TextureTransform().SetDimensions(srcW, srcH, 3);
                // The `3` makes the converter sample only RGB channels (drops alpha),
                // and passing the source dimensions enables aspect-aware resampling.
                var tt = new Unity.InferenceEngine.TextureTransform()
                    .SetDimensions(cameraTexture.width, cameraTexture.height, 3);
                using var input = new Unity.InferenceEngine.Tensor<float>(
                    new Unity.InferenceEngine.TensorShape(1, 3, _inputSize, _inputSize));
                Unity.InferenceEngine.TextureConverter.ToTensor(cameraTexture, input, tt);
                _worker.Schedule(input);

                var rawOutput = _worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
                if (rawOutput == null)
                {
                    Debug.LogWarning("[YoloPoseDetector] PeekOutput() returned null.");
                    return _scratchDetections;
                }

                using var output = rawOutput.ReadbackAndClone();
                var data  = output.DownloadToArray();
                var shape = output.shape;

                float scaleX = (float)cameraTexture.width  / _inputSize;
                float scaleY = (float)cameraTexture.height / _inputSize;

                // Pass 1: full-body detection — skipped when ArmOnlyMode is active.
                if (!ArmOnlyMode)
                    ParseAndNms(data, shape, scaleX, scaleY, _scratchDetections);

                // Pass 2: arm-keypoint fallback.
                //   • Always runs in ArmOnlyMode.
                //   • Runs in Normal mode only when pass 1 found nothing.
                if (_scratchDetections.Count == 0 && (ArmOnlyMode || _armOnlyFallback))
                {
                    ArmOnlyPass(data, shape, scaleX, scaleY, _scratchDetections);
                    if (_scratchDetections.Count > 0)
                        LastRunWasArmOnlyFallback = true;
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[YoloPoseDetector] Run() threw: {ex.GetType().Name}: {ex.Message}");
            }

            return _scratchDetections;
        }

        // ── Full-body pass ─────────────────────────────────────────────────────────────

        private void ParseAndNms(float[] data, Unity.InferenceEngine.TensorShape shape,
                                 float scaleX, float scaleY, List<PersonDetection> output)
        {
            if (shape.rank < 3) return;
            int channels = shape[1];
            int anchors  = shape[2];
            if (channels < 5 + 17 * 3) return;

            var candIdx = _scratchIndices;
            candIdx.Clear();

            float[] confs = new float[anchors];
            for (int i = 0; i < anchors; i++)
            {
                float c = data[4 * anchors + i];
                if (c >= _confidenceThreshold) { confs[i] = c; candIdx.Add(i); }
            }
            candIdx.Sort((a, b) => confs[b].CompareTo(confs[a]));

            float[] bx = new float[candIdx.Count], by = new float[candIdx.Count];
            float[] bw = new float[candIdx.Count], bh = new float[candIdx.Count];
            for (int k = 0; k < candIdx.Count; k++)
            {
                int i = candIdx[k];
                bx[k] = data[0 * anchors + i]; by[k] = data[1 * anchors + i];
                bw[k] = data[2 * anchors + i]; bh[k] = data[3 * anchors + i];
            }

            bool[] suppressed = new bool[candIdx.Count];
            for (int a = 0; a < candIdx.Count && output.Count < _maxDetections; a++)
            {
                if (suppressed[a]) continue;
                int i = candIdx[a];

                var person = new PersonDetection
                {
                    Confidence  = confs[i],
                    ImageBounds = new Rect(
                        (bx[a] - bw[a] * 0.5f) * scaleX, (by[a] - bh[a] * 0.5f) * scaleY,
                        bw[a] * scaleX, bh[a] * scaleY),
                    Keypoints = BuildKeypoints(data, anchors, i, scaleX, scaleY),
                };
                output.Add(person);

                for (int b = a + 1; b < candIdx.Count; b++)
                {
                    if (!suppressed[b] &&
                        BoxIoU(bx[a], by[a], bw[a], bh[a], bx[b], by[b], bw[b], bh[b]) > _nmsIoUThreshold)
                        suppressed[b] = true;
                }
            }
        }

        // ── Arm-only fallback pass ─────────────────────────────────────────────────────

        // COCO keypoint indices used for arm detection.
        // Each row is (shoulderKpIndex, wristKpIndex) for one arm side.
        private static readonly (int sh, int wr)[] ArmPairs =
        {
            ((int)CocoKeypoint.LeftShoulder,  (int)CocoKeypoint.LeftWrist),
            ((int)CocoKeypoint.RightShoulder, (int)CocoKeypoint.RightWrist),
        };

        /// <summary>
        /// Bypass person-confidence gating entirely and scan raw anchors for any with
        /// high shoulder + wrist keypoint visibility. Designed for isolated-arm scenarios
        /// (e.g. mannequin arm, partial body) where the YOLO person head scores near zero.
        /// </summary>
        private void ArmOnlyPass(float[] data, Unity.InferenceEngine.TensorShape shape,
                                 float scaleX, float scaleY, List<PersonDetection> output)
        {
            if (shape.rank < 3) return;
            int anchors = shape[2];

            // Score each anchor by its best arm pair: min(shoulder_v, wrist_v).
            // Also track the global max so the HUD can show it for threshold tuning.
            float[] armScores = new float[anchors];
            float globalMax   = 0f;
            for (int i = 0; i < anchors; i++)
            {
                float best = 0f;
                foreach (var (sh, wr) in ArmPairs)
                {
                    float sv = data[(5 + sh * 3 + 2) * anchors + i]; // shoulder visibility
                    float wv = data[(5 + wr * 3 + 2) * anchors + i]; // wrist visibility
                    float s  = Mathf.Min(sv, wv);
                    if (s > best) best = s;
                }
                armScores[i] = best;
                if (best > globalMax) globalMax = best;
            }
            LastArmOnlyMaxScore = globalMax;

            // Collect candidates above the threshold, sorted by score descending.
            // Geometric plausibility (arm length, in-frame keypoints) is checked too,
            // because raw noise anchors can carry high visibility scores with
            // degenerate keypoint geometry.
            var cands = new List<int>(64);
            for (int i = 0; i < anchors; i++)
                if (armScores[i] >= _armOnlyKeypointThreshold && IsPlausibleArm(data, anchors, i))
                    cands.Add(i);
            if (cands.Count == 0) return;

            cands.Sort((a, b) => armScores[b].CompareTo(armScores[a]));

            // Simple spatial NMS: suppress anchors whose best-arm shoulder falls within
            // 80 model-pixels of an already-accepted anchor's shoulder.
            const float NmsRadiusPx = 80f;
            bool[] suppressed = new bool[cands.Count];

            for (int a = 0; a < cands.Count && output.Count < _maxDetections; a++)
            {
                if (suppressed[a]) continue;

                int i = cands[a];
                var kps = BuildKeypoints(data, anchors, i, scaleX, scaleY);

                // Find the shoulder position of the best arm side for NMS.
                Vector2 refShoulder = GetBestArmShoulder(data, anchors, i);

                // Build a bounding box from the visible arm keypoints.
                Rect bounds = BuildArmBounds(kps, scaleX);
                if (bounds.width < 1f && bounds.height < 1f) continue; // degenerate

                output.Add(new PersonDetection
                {
                    Confidence  = armScores[i],
                    ImageBounds = bounds,
                    Keypoints   = kps,
                });

                // Suppress nearby duplicates.
                for (int b = a + 1; b < cands.Count; b++)
                {
                    if (suppressed[b]) continue;
                    Vector2 otherShoulder = GetBestArmShoulder(data, anchors, cands[b]);
                    if (Vector2.Distance(refShoulder, otherShoulder) < NmsRadiusPx)
                        suppressed[b] = true;
                }
            }
        }

        /// <summary>
        /// Rejects implausible arm-only anchors: the best arm pair must have a sensible
        /// shoulder→wrist pixel length and both keypoints inside the model input frame.
        /// Positions here are model-space (pre-scale, 0.._inputSize).
        /// </summary>
        private bool IsPlausibleArm(float[] data, int anchors, int anchorIdx)
        {
            float bestScore = -1f;
            Vector2 sh = default, wr = default;
            foreach (var (s, w) in ArmPairs)
            {
                float sv = data[(5 + s * 3 + 2) * anchors + anchorIdx];
                float wv = data[(5 + w * 3 + 2) * anchors + anchorIdx];
                float sc = Mathf.Min(sv, wv);
                if (sc > bestScore)
                {
                    bestScore = sc;
                    sh = new Vector2(data[(5 + s * 3 + 0) * anchors + anchorIdx],
                                     data[(5 + s * 3 + 1) * anchors + anchorIdx]);
                    wr = new Vector2(data[(5 + w * 3 + 0) * anchors + anchorIdx],
                                     data[(5 + w * 3 + 1) * anchors + anchorIdx]);
                }
            }

            if (Vector2.Distance(sh, wr) < _armOnlyMinArmPixels) return false;

            const float Margin = 8f;
            if (sh.x < -Margin || sh.x > _inputSize + Margin ||
                sh.y < -Margin || sh.y > _inputSize + Margin) return false;
            if (wr.x < -Margin || wr.x > _inputSize + Margin ||
                wr.y < -Margin || wr.y > _inputSize + Margin) return false;
            return true;
        }

        // ── Shared helpers ─────────────────────────────────────────────────────────────

        private static Keypoint[] BuildKeypoints(float[] data, int anchors, int anchorIdx,
                                                  float scaleX, float scaleY)
        {
            var kps = new Keypoint[17];
            for (int k = 0; k < 17; k++)
            {
                int ch = 5 + k * 3;
                kps[k] = new Keypoint
                {
                    ImagePos   = new Vector2(data[(ch + 0) * anchors + anchorIdx] * scaleX,
                                             data[(ch + 1) * anchors + anchorIdx] * scaleY),
                    Confidence = data[(ch + 2) * anchors + anchorIdx],
                };
            }
            return kps;
        }

        /// <summary>Returns the model-space (pre-scale) shoulder position of the higher-scoring arm side.</summary>
        private static Vector2 GetBestArmShoulder(float[] data, int anchors, int anchorIdx)
        {
            float bestScore = -1f;
            Vector2 best    = Vector2.zero;
            foreach (var (sh, wr) in ArmPairs)
            {
                float sv = data[(5 + sh * 3 + 2) * anchors + anchorIdx];
                float wv = data[(5 + wr * 3 + 2) * anchors + anchorIdx];
                if (Mathf.Min(sv, wv) > bestScore)
                {
                    bestScore = Mathf.Min(sv, wv);
                    best = new Vector2(data[(5 + sh * 3 + 0) * anchors + anchorIdx],
                                       data[(5 + sh * 3 + 1) * anchors + anchorIdx]);
                }
            }
            return best;
        }

        /// <summary>
        /// Computes a tight bounding box from the visible arm keypoints
        /// (shoulders, elbows, wrists) already scaled to camera-image pixels.
        /// </summary>
        private static Rect BuildArmBounds(Keypoint[] kps, float scaleX)
        {
            // Arm keypoint indices in COCO: 5-10 (shoulders, elbows, wrists)
            const float VisThresh = 0.05f;
            const float Padding   = 15f;

            float minX =  float.MaxValue, minY =  float.MaxValue;
            float maxX = -float.MaxValue, maxY = -float.MaxValue;
            bool any = false;

            for (int k = 5; k <= 10; k++)
            {
                if (kps[k].Confidence < VisThresh) continue;
                minX = Mathf.Min(minX, kps[k].ImagePos.x);
                minY = Mathf.Min(minY, kps[k].ImagePos.y);
                maxX = Mathf.Max(maxX, kps[k].ImagePos.x);
                maxY = Mathf.Max(maxY, kps[k].ImagePos.y);
                any  = true;
            }

            if (!any) return default;

            float pad = Padding * scaleX;
            return new Rect(minX - pad, minY - pad,
                            maxX - minX + pad * 2f,
                            maxY - minY + pad * 2f);
        }

        private static float BoxIoU(float ax, float ay, float aw, float ah,
                                     float bx, float by, float bw, float bh)
        {
            float ax1 = ax - aw * .5f, ay1 = ay - ah * .5f, ax2 = ax + aw * .5f, ay2 = ay + ah * .5f;
            float bx1 = bx - bw * .5f, by1 = by - bh * .5f, bx2 = bx + bw * .5f, by2 = by + bh * .5f;
            float iw = Mathf.Max(0, Mathf.Min(ax2, bx2) - Mathf.Max(ax1, bx1));
            float ih = Mathf.Max(0, Mathf.Min(ay2, by2) - Mathf.Max(ay1, by1));
            float inter = iw * ih;
            float union = aw * ah + bw * bh - inter;
            return union > 0f ? inter / union : 0f;
        }
    }
}
