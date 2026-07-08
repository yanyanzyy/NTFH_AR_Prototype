using System;
using System.Collections;
using System.Reflection;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Provides the passthrough camera texture on Quest 3 PLUS calibrated image-to-world
    /// projection using the device's real camera intrinsics and timestamp-aware pose.
    ///
    /// Resolution order (highest priority first):
    ///   1. Meta.XR.PassthroughCameraAccess (MRUK 201+, the official PCA API — gives us
    ///      real intrinsics + timestamp-locked camera pose).
    ///   2. Older Meta WebCamTextureManager / PassthroughCameraManager (found via
    ///      reflection — kept for backwards compatibility, uses an FOV-based fallback).
    ///   3. Direct Unity WebCamTexture.devices fallback (when no manager is present).
    ///   4. Editor WebCamTexture fallback (for desktop testing).
    ///
    /// Image-to-world projection
    /// -------------------------
    /// When PassthroughCameraAccess is the source, ImagePointToWorld() and
    /// ImagePointToRay() use the device intrinsics + cached camera pose, which is
    /// vastly more accurate than the previous hardcoded-FOV / Camera.main-transform
    /// path. The pose is cached once per render frame via Refresh() (called implicitly
    /// from CurrentTexture) so all projections within a frame use a single consistent
    /// pose, matching the inferred image.
    /// </summary>
    public class PassthroughCameraSource : MonoBehaviour
    {
        [Tooltip("Drag the '[BuildingBlock] Passthrough Camera Access' GameObject here. " +
                 "If left empty the component is found automatically at runtime.")]
        [SerializeField] private MonoBehaviour _webCamTextureManager;

        [Tooltip("Fallback used in the Unity Editor when running in Play mode on desktop.")]
        [SerializeField] private WebCamTexture _editorFallbackWebCamTexture;

        [Tooltip("Reference transform used as the camera origin in the FOV-based fallback path. " +
                 "Ignored when PassthroughCameraAccess is the active source. Defaults to Camera.main.")]
        [SerializeField] private Transform _cameraReferenceTransform;

        [Tooltip("Horizontal FOV (degrees) used by the FOV-based fallback projection. " +
                 "Ignored when PassthroughCameraAccess provides real intrinsics.")]
        [SerializeField] private float _horizontalFovDegrees = 82f;

        // ── Reflection cache for the various manager APIs ──────────────────────────────

        private enum SourceKind { None, PassthroughCameraAccess, LegacyManager, DirectWebCam, EditorWebCam }
        private SourceKind _sourceKind = SourceKind.None;

        // Reflection handles for the Meta.XR.PassthroughCameraAccess API (used so we
        // don't add a hard compile-time dependency on a specific SDK version).
        private MethodInfo   _pcaGetTexture;        // Texture GetTexture()
        private PropertyInfo _pcaIsPlaying;         // bool IsPlaying
        private PropertyInfo _pcaCurrentResolution; // Vector2Int CurrentResolution
        private PropertyInfo _pcaIntrinsics;        // PCA.CameraIntrinsics
        private MethodInfo   _pcaGetCameraPose;     // Pose GetCameraPose()
        private FieldInfo    _intrFocalLength;      // CameraIntrinsics.FocalLength
        private FieldInfo    _intrPrincipalPoint;   // CameraIntrinsics.PrincipalPoint
        private FieldInfo    _intrSensorResolution; // CameraIntrinsics.SensorResolution

        // Compiled-delegate fast paths for the members hit every frame. MethodInfo.Invoke
        // allocates (boxing + args array) on every call; a bound delegate does not. Falls
        // back to reflection when a signature doesn't match this SDK version.
        private Func<Texture>    _pcaGetTextureFast;
        private Func<bool>       _pcaIsPlayingFast;
        private Func<Vector2Int> _pcaCurrentResolutionFast;
        private Func<Pose>       _pcaGetCameraPoseFast;
        // Intrinsics are constant after Play() (sensor-space values), so they are read via
        // reflection only until the first successful read, then served from cache.
        private bool _intrinsicsCached;

        // Reflection handles for the legacy WebCamTextureManager-style API.
        private PropertyInfo _legacyTexProperty;
        private MethodInfo   _legacyTexMethodCam;
        private MethodInfo   _legacyTexMethodNoArg;
        private bool         _reflectionDone;

        private WebCamTexture _directCam;
        private WebCamTexture _editorCam;

        // ── Cached per-frame state (refreshed by CurrentTexture / Refresh) ────────────

        private int    _lastRefreshFrame = -1;
        private Pose   _cachedPose;
        private bool   _cachedPoseValid;
        private Vector2 _focalLengthPx;
        private Vector2 _principalPointPx;
        private Vector2Int _sensorResolution;   // intrinsics native pixel grid
        private bool   _hasIntrinsics;
        private int    _imageWidth;
        private int    _imageHeight;
        private Texture _currentTexture;

        // Shown in the debug HUD.
        public string CameraManagerStatus { get; private set; } = "Initialising…";

        // ── Public surface ─────────────────────────────────────────────────────────────

        /// <summary>Returns the camera transform used by the fallback projection path. Ignored when PassthroughCameraAccess is active.</summary>
        public Transform CameraTransform =>
            _cameraReferenceTransform
            ?? (Camera.main != null ? Camera.main.transform : transform);

        /// <summary>World-space pose of the RGB camera at the moment the current frame was captured. Falls back to Camera.main when intrinsics aren't available.</summary>
        public Pose CameraPose
        {
            get
            {
                EnsureRefreshed();
                if (_cachedPoseValid) return _cachedPose;
                var t = CameraTransform;
                return t != null ? new Pose(t.position, t.rotation) : default;
            }
        }

        /// <summary>True when the active source provides real intrinsics + a calibrated pose (PassthroughCameraAccess).</summary>
        public bool HasCalibratedProjection
        {
            get { EnsureRefreshed(); return _hasIntrinsics && _cachedPoseValid; }
        }

        public bool HasFrame
        {
            get { EnsureRefreshed(); return _currentTexture != null && _imageWidth > 16; }
        }

        public int Width  { get { EnsureRefreshed(); return _imageWidth;  } }
        public int Height { get { EnsureRefreshed(); return _imageHeight; } }

        public Texture CurrentTexture
        {
            get { EnsureRefreshed(); return _currentTexture; }
        }

        public float HorizontalFovRadians => _horizontalFovDegrees * Mathf.Deg2Rad;

        public float VerticalFovRadians
        {
            get
            {
                if (Width == 0 || Height == 0)
                    return _horizontalFovDegrees * Mathf.Deg2Rad * 0.75f;
                float halfH = Mathf.Tan(HorizontalFovRadians * 0.5f);
                return 2f * Mathf.Atan(halfH / ((float)Width / Height));
            }
        }

        // ── Unity lifecycle ────────────────────────────────────────────────────────────

        private void Awake()
        {
            TryFindManager();

            if (_editorFallbackWebCamTexture != null)
            {
                _editorCam = _editorFallbackWebCamTexture;
                if (!_editorCam.isPlaying) _editorCam.Play();
            }

            // Direct WebCam fallback. Coroutine waits 1 s so the permission prompt has time to land.
            StartCoroutine(TryDirectWebCamAfterDelay());
        }

        private void OnDestroy()
        {
            if (_directCam != null && _directCam.isPlaying) _directCam.Stop();
        }

        // ── Refresh: resolve the texture + intrinsics + pose for THIS frame ───────────

        /// <summary>
        /// Resolves the current texture, intrinsics, and camera pose. Cached so multiple
        /// calls within a single frame return identical values (essential so detection
        /// and the bounding-box visualiser project against the same pose).
        /// </summary>
        private void EnsureRefreshed()
        {
            if (_lastRefreshFrame == Time.frameCount) return;
            _lastRefreshFrame = Time.frameCount;

            _currentTexture  = null;
            _cachedPoseValid = false;
            _hasIntrinsics   = false;
            _imageWidth = _imageHeight = 0;

            // 1. Editor webcam fallback.
            if (_editorCam != null && _editorCam.isPlaying)
            {
                _currentTexture = _editorCam;
                _imageWidth  = _editorCam.width;
                _imageHeight = _editorCam.height;
                _sourceKind = SourceKind.EditorWebCam;
                return;
            }

            // 2. PassthroughCameraAccess (preferred Quest 3 path).
            if (_webCamTextureManager == null) TryFindManager();
            if (_webCamTextureManager != null && _sourceKind == SourceKind.PassthroughCameraAccess)
            {
                if (TryRefreshFromPassthroughCameraAccess()) return;
                // Fall through if PCA returned no texture (e.g. permission not yet granted).
            }

            // 3. Legacy WebCamTextureManager (older Meta SDKs).
            if (_webCamTextureManager != null && _sourceKind == SourceKind.LegacyManager)
            {
                var t = ResolveViaLegacyManagerReflection();
                if (t != null)
                {
                    _currentTexture = t;
                    _imageWidth = t.width;
                    _imageHeight = t.height;
                    return;
                }
            }

            // 4. Direct WebCam fallback.
            if (_directCam != null && _directCam.isPlaying && _directCam.width > 16)
            {
                _currentTexture = _directCam;
                _imageWidth = _directCam.width;
                _imageHeight = _directCam.height;
                _sourceKind = SourceKind.DirectWebCam;
            }
        }

        private bool TryRefreshFromPassthroughCameraAccess()
        {
            try
            {
                if (_pcaIsPlayingFast != null)
                {
                    if (!_pcaIsPlayingFast()) return false;
                }
                else if (_pcaIsPlaying != null)
                {
                    var playing = _pcaIsPlaying.GetValue(_webCamTextureManager);
                    if (playing is bool b && !b) return false;
                }

                Texture tex;
                if (_pcaGetTextureFast != null)
                {
                    tex = _pcaGetTextureFast();
                }
                else
                {
                    if (_pcaGetTexture == null) return false;
                    tex = _pcaGetTexture.Invoke(_webCamTextureManager, null) as Texture;
                }
                if (tex == null) return false;

                _currentTexture = tex;

                // CurrentResolution is the active stream resolution (may differ from the texture's reported size).
                Vector2Int res = default;
                if (_pcaCurrentResolutionFast != null)
                    res = _pcaCurrentResolutionFast();
                else if (_pcaCurrentResolution != null &&
                         _pcaCurrentResolution.GetValue(_webCamTextureManager) is Vector2Int r)
                    res = r;

                if (res.x > 0 && res.y > 0)
                {
                    _imageWidth  = res.x;
                    _imageHeight = res.y;
                }
                else
                {
                    _imageWidth  = tex.width;
                    _imageHeight = tex.height;
                }

                // Intrinsics: focal length & principal point in sensor pixels. Constant after
                // Play(), so the boxing-heavy reflection reads run only until the first success.
                if (_intrinsicsCached)
                {
                    _hasIntrinsics = true;
                }
                else if (_pcaIntrinsics != null)
                {
                    var intr = _pcaIntrinsics.GetValue(_webCamTextureManager);
                    if (intr != null && _intrFocalLength != null && _intrPrincipalPoint != null)
                    {
                        _focalLengthPx    = (Vector2)_intrFocalLength.GetValue(intr);
                        _principalPointPx = (Vector2)_intrPrincipalPoint.GetValue(intr);
                        if (_intrSensorResolution != null)
                            _sensorResolution = (Vector2Int)_intrSensorResolution.GetValue(intr);
                        _hasIntrinsics = _focalLengthPx.x > 0f && _focalLengthPx.y > 0f
                                      && _sensorResolution.x > 0 && _sensorResolution.y > 0;
                        _intrinsicsCached = _hasIntrinsics;
                    }
                }

                // Timestamp-aware camera pose (the pose at the moment the image was captured).
                if (_pcaGetCameraPoseFast != null)
                {
                    var p = _pcaGetCameraPoseFast();
                    _cachedPose = p;
                    _cachedPoseValid = p.rotation != default;
                }
                else if (_pcaGetCameraPose != null)
                {
                    var poseObj = _pcaGetCameraPose.Invoke(_webCamTextureManager, null);
                    if (poseObj is Pose p)
                    {
                        _cachedPose = p;
                        _cachedPoseValid = p.rotation != default;
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[PassthroughCameraSource] PCA refresh threw: {ex.GetType().Name} {ex.Message}");
                return false;
            }
        }

        // ── Manager discovery ──────────────────────────────────────────────────────────

        private void TryFindManager()
        {
            // If something is already assigned, classify each component on that GO.
            if (_webCamTextureManager != null)
            {
                var (correct, kind) = ClassifyManagerOnGameObject(_webCamTextureManager.gameObject);
                if (correct != null)
                {
                    _webCamTextureManager = correct;
                    _sourceKind = kind;
                    _reflectionDone = false;
                    BuildReflectionCache();
                    CameraManagerStatus = $"Manager: {correct.GetType().Name} ({kind})";
                    return;
                }
            }

            // Scene-wide search.
            foreach (var mb in FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include))
            {
                if (mb == null) continue;
                var kind = ClassifyType(mb.GetType().Name);
                if (kind != SourceKind.None)
                {
                    _webCamTextureManager = mb;
                    _sourceKind = kind;
                    _reflectionDone = false;
                    BuildReflectionCache();
                    CameraManagerStatus = $"Scene-found: {mb.GetType().Name} ({kind}) on '{mb.gameObject.name}'";
                    Debug.Log($"[PassthroughCameraSource] {CameraManagerStatus}");
                    return;
                }
            }

            _sourceKind = SourceKind.None;
            CameraManagerStatus = "Manager: NOT FOUND (using direct WebCam fallback)";
            Debug.LogWarning("[PassthroughCameraSource] No PassthroughCameraAccess / WebCamTextureManager in scene. " +
                             "Will try WebCamTexture.devices directly.");
        }

        private static (MonoBehaviour mb, SourceKind kind) ClassifyManagerOnGameObject(GameObject go)
        {
            foreach (var mb in go.GetComponents<MonoBehaviour>())
            {
                if (mb == null) continue;
                var kind = ClassifyType(mb.GetType().Name);
                if (kind != SourceKind.None) return (mb, kind);
            }
            return (null, SourceKind.None);
        }

        private static SourceKind ClassifyType(string typeName) => typeName switch
        {
            "PassthroughCameraAccess"  => SourceKind.PassthroughCameraAccess,
            "WebCamTextureManager"     => SourceKind.LegacyManager,
            "PassthroughCameraManager" => SourceKind.LegacyManager,
            _                          => SourceKind.None,
        };

        // ── Direct WebCamTexture fallback ──────────────────────────────────────────────

        private IEnumerator TryDirectWebCamAfterDelay()
        {
            yield return new WaitForSeconds(1f);
            if (HasFrame) yield break;

            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                CameraManagerStatus += " | No WebCam devices found";
                Debug.LogWarning("[PassthroughCameraSource] WebCamTexture.devices is empty. " +
                                 "Camera permission may not be granted yet.");
                yield break;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"[PassthroughCameraSource] WebCam devices ({devices.Length}):");
            for (int i = 0; i < devices.Length; i++)
                sb.AppendLine($"  [{i}] '{devices[i].name}'  front={devices[i].isFrontFacing}");
            Debug.Log(sb.ToString());

            string chosen = devices[0].name;
            foreach (var d in devices)
                if (!d.isFrontFacing) { chosen = d.name; break; }

            _directCam = new WebCamTexture(chosen, 1280, 960, 30);
            _directCam.Play();
            CameraManagerStatus = $"Direct WebCam: '{chosen}'";
            Debug.Log($"[PassthroughCameraSource] Started direct WebCam: {chosen}");
        }

        // ── Reflection cache builder ───────────────────────────────────────────────────

        private void BuildReflectionCache()
        {
            _reflectionDone = true;
            if (_webCamTextureManager == null) return;
            var type = _webCamTextureManager.GetType();

            if (_sourceKind == SourceKind.PassthroughCameraAccess)
            {
                _pcaGetTexture        = type.GetMethod("GetTexture", Type.EmptyTypes);
                _pcaIsPlaying         = type.GetProperty("IsPlaying");
                _pcaCurrentResolution = type.GetProperty("CurrentResolution");
                _pcaIntrinsics        = type.GetProperty("Intrinsics");
                _pcaGetCameraPose     = type.GetMethod("GetCameraPose", Type.EmptyTypes);

                // Bind allocation-free delegates for the per-frame members; any that don't
                // match this SDK version's signature stay null and use reflection instead.
                _pcaGetTextureFast        = TryBindDelegate<Func<Texture>>(_pcaGetTexture);
                _pcaIsPlayingFast         = TryBindDelegate<Func<bool>>(_pcaIsPlaying?.GetGetMethod());
                _pcaCurrentResolutionFast = TryBindDelegate<Func<Vector2Int>>(_pcaCurrentResolution?.GetGetMethod());
                _pcaGetCameraPoseFast     = TryBindDelegate<Func<Pose>>(_pcaGetCameraPose);
                _intrinsicsCached         = false;

                if (_pcaIntrinsics != null)
                {
                    var intrType = _pcaIntrinsics.PropertyType;
                    _intrFocalLength      = intrType.GetField("FocalLength");
                    _intrPrincipalPoint   = intrType.GetField("PrincipalPoint");
                    _intrSensorResolution = intrType.GetField("SensorResolution");
                }
                Debug.Log("[PassthroughCameraSource] PCA reflection bound: " +
                          $"GetTexture={_pcaGetTexture != null} IsPlaying={_pcaIsPlaying != null} " +
                          $"Intrinsics={_pcaIntrinsics != null} GetCameraPose={_pcaGetCameraPose != null}");
                return;
            }

            // Legacy manager path.
            _legacyTexMethodCam   = type.GetMethod("GetWebCamTexture", new[] { typeof(Camera) });
            _legacyTexMethodNoArg = type.GetMethod("GetWebCamTexture", Type.EmptyTypes);
            _legacyTexProperty    = type.GetProperty("WebCamTexture")
                                 ?? type.GetProperty("Texture")
                                 ?? type.GetProperty("CameraTexture");

            if (_legacyTexMethodCam != null)
                Debug.Log($"[PassthroughCameraSource] Legacy: GetWebCamTexture(Camera) on {type.Name}");
            else if (_legacyTexMethodNoArg != null)
                Debug.Log($"[PassthroughCameraSource] Legacy: GetWebCamTexture() on {type.Name}");
            else if (_legacyTexProperty != null)
                Debug.Log($"[PassthroughCameraSource] Legacy: property '{_legacyTexProperty.Name}' on {type.Name}");
            else
                Debug.LogError($"[PassthroughCameraSource] No texture accessor on {type.FullName}");
        }

        private TDelegate TryBindDelegate<TDelegate>(MethodInfo method) where TDelegate : class
        {
            if (method == null || _webCamTextureManager == null) return null;
            return Delegate.CreateDelegate(typeof(TDelegate), _webCamTextureManager, method, false) as TDelegate;
        }

        private Texture ResolveViaLegacyManagerReflection()
        {
            if (!_reflectionDone) BuildReflectionCache();

            if (_legacyTexMethodCam != null)
            {
                try { var r = _legacyTexMethodCam.Invoke(_webCamTextureManager, new object[] { Camera.main });
                      if (r is Texture t && t) return t; } catch { }
            }
            if (_legacyTexMethodNoArg != null)
            {
                try { var r = _legacyTexMethodNoArg.Invoke(_webCamTextureManager, null);
                      if (r is Texture t && t) return t; } catch { }
            }
            if (_legacyTexProperty != null)
            {
                try { var r = _legacyTexProperty.GetValue(_webCamTextureManager);
                      if (r is Texture t && t) return t; } catch { }
            }
            return null;
        }

        // ── Projection helpers ─────────────────────────────────────────────────────────

        /// <summary>
        /// Returns a normalized direction in world space pointing through the given image pixel,
        /// using real camera intrinsics when available, otherwise the FOV-based fallback.
        /// </summary>
        public Ray ImagePointToRay(Vector2 pt)
        {
            EnsureRefreshed();
            if (_hasIntrinsics && _cachedPoseValid && _imageWidth > 0 && _imageHeight > 0)
            {
                // Convert image coords (origin top-left) to viewport coords (origin bottom-left)
                // matching Meta PCA's ViewportPointToLocalRay convention.
                float u = pt.x / _imageWidth;
                float v = 1f - (pt.y / _imageHeight);

                // PCA's intrinsics are in SENSOR pixel space. Current stream resolution may be
                // smaller than the sensor (cropped), so map (u,v) into sensor space using the
                // same crop maths as Meta.XR.PassthroughCameraAccess.CalcSensorCropRegion.
                Vector2 sensorRes  = (Vector2)_sensorResolution;
                Vector2 currentRes = new Vector2(_imageWidth, _imageHeight);
                Vector2 scaleFactor = currentRes / sensorRes;
                scaleFactor /= Mathf.Max(scaleFactor.x, scaleFactor.y);
                Rect crop = new Rect(
                    sensorRes.x * (1f - scaleFactor.x) * 0.5f,
                    sensorRes.y * (1f - scaleFactor.y) * 0.5f,
                    sensorRes.x * scaleFactor.x,
                    sensorRes.y * scaleFactor.y);

                Vector3 dirCam = new Vector3(
                    (crop.x + crop.width  * u - _principalPointPx.x) / _focalLengthPx.x,
                    (crop.y + crop.height * v - _principalPointPx.y) / _focalLengthPx.y,
                    1f);

                Vector3 dirWorld = (_cachedPose.rotation * dirCam).normalized;
                return new Ray(_cachedPose.position, dirWorld);
            }

            // Fallback: hardcoded-FOV path using Camera.main.
            float w = Mathf.Max(1, _imageWidth), h = Mathf.Max(1, _imageHeight);
            float ndcX  = (pt.x / w) * 2f - 1f;
            float ndcY  = 1f - (pt.y / h) * 2f;
            float halfH = Mathf.Tan(HorizontalFovRadians * 0.5f);
            float halfV = Mathf.Tan(VerticalFovRadians   * 0.5f);
            Vector3 localDir = new Vector3(ndcX * halfH, ndcY * halfV, 1f).normalized;
            var cam = CameraTransform;
            return new Ray(cam != null ? cam.position : Vector3.zero,
                           cam != null ? cam.rotation * localDir : localDir);
        }

        /// <summary>Legacy convenience: direction-only version, in CameraTransform's local frame.</summary>
        public Vector3 ImagePointToCameraRay(Vector2 pt)
        {
            float w = Width, h = Height;
            if (w <= 0 || h <= 0) return Vector3.forward;
            float ndcX  = (pt.x / w) * 2f - 1f;
            float ndcY  = 1f - (pt.y / h) * 2f;
            float halfH = Mathf.Tan(HorizontalFovRadians * 0.5f);
            float halfV = Mathf.Tan(VerticalFovRadians   * 0.5f);
            return new Vector3(ndcX * halfH, ndcY * halfV, 1f).normalized;
        }

        /// <summary>
        /// Projects an image-space pixel into world space at the given depth (metres).
        /// Uses real intrinsics when available, otherwise the FOV-based fallback.
        /// </summary>
        public Vector3 ImagePointToWorld(Vector2 pt, float distanceMeters)
        {
            var ray = ImagePointToRay(pt);
            return ray.origin + ray.direction * distanceMeters;
        }

        public bool WorldToImagePoint(Vector3 world, out Vector2 imagePoint)
        {
            imagePoint = default;
            EnsureRefreshed();
            float w = _imageWidth, h = _imageHeight;
            if (w <= 0 || h <= 0) return false;

            if (_hasIntrinsics && _cachedPoseValid)
            {
                Vector3 local = Quaternion.Inverse(_cachedPose.rotation) * (world - _cachedPose.position);
                if (local.z <= 0.01f) return false;

                // Sensor-space pixel of the world point.
                float sx = local.x / local.z * _focalLengthPx.x + _principalPointPx.x;
                float sy = local.y / local.z * _focalLengthPx.y + _principalPointPx.y;

                // Sensor → viewport using same crop region as the forward path.
                Vector2 sensorRes  = (Vector2)_sensorResolution;
                Vector2 currentRes = new Vector2(_imageWidth, _imageHeight);
                Vector2 scaleFactor = currentRes / sensorRes;
                scaleFactor /= Mathf.Max(scaleFactor.x, scaleFactor.y);
                Rect crop = new Rect(
                    sensorRes.x * (1f - scaleFactor.x) * 0.5f,
                    sensorRes.y * (1f - scaleFactor.y) * 0.5f,
                    sensorRes.x * scaleFactor.x,
                    sensorRes.y * scaleFactor.y);

                float u = (sx - crop.x) / crop.width;
                float v = (sy - crop.y) / crop.height;
                imagePoint = new Vector2(u * w, (1f - v) * h);
                return true;
            }

            // Fallback.
            var cam = CameraTransform;
            var localF = Quaternion.Inverse(cam.rotation) * (world - cam.position);
            if (localF.z <= 0.01f) return false;
            float halfH = Mathf.Tan(HorizontalFovRadians * 0.5f);
            float halfV = Mathf.Tan(VerticalFovRadians   * 0.5f);
            imagePoint  = new Vector2(
                ((localF.x / localF.z) / halfH * 0.5f + 0.5f) * w,
                (0.5f - (localF.y / localF.z) / halfV * 0.5f) * h);
            return true;
        }
    }
}
