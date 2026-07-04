using System.Collections.Generic;
using UnityEngine;
#if OPENCV_FOR_UNITY
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;
#endif

namespace ARArmDetection
{
    /// <summary>
    /// ArUco (DICT_4X4_50) corner detection backend for <see cref="MarkerArmTracker"/>,
    /// built on the OpenCV for Unity asset.
    ///
    /// SETUP (the tracker compiles and runs without this, but reports no markers):
    ///   1. Import "OpenCV for Unity" (Enox Software) from the Asset Store.
    ///   2. Project Settings > Player > Scripting Define Symbols: add OPENCV_FOR_UNITY.
    ///   3. Put this component on the same GameObject as MarkerArmTracker.
    ///
    /// Corner coordinates are emitted in image pixels with a TOP-LEFT origin, matching
    /// what PassthroughCameraSource.ImagePointToRay expects. If the overlay ends up
    /// vertically mirrored on your device, toggle Flip Y (WebCamTexture orientation
    /// differs across sources).
    /// </summary>
    public class ArUcoCornerProvider : MonoBehaviour, IMarkerCornerProvider
    {
        [Tooltip("Downscale factor for detection (1 = full resolution). 2 halves CPU cost but " +
                 "also halves corner precision, which directly degrades pose accuracy - prefer " +
                 "raising the tracker's Detect Every N Frames instead.")]
        [SerializeField, Range(1, 4)] private int _downscale = 1;

        [Tooltip("Flip detected corner Y (enable if the overlay is vertically mirrored).")]
        [SerializeField] private bool _flipY;

        public string Status { get; private set; } = "Not started";

#if OPENCV_FOR_UNITY
        private ArucoDetector _detector;
        private Mat _rgbaMat;
        private Mat _grayMat;
        private Mat _smallMat;
        private Texture2D _readbackTex;
        private readonly List<Mat> _cornersOut = new();
        private Mat _idsOut;

        public bool IsReady => true;

        private void OnEnable()
        {
            var dict = Objdetect.getPredefinedDictionary(Objdetect.DICT_4X4_50);
            var detParams = new DetectorParameters();
            // Subpixel corners are REQUIRED for pose quality: at 1 px corner noise the
            // two-facet geometry degrades to tens of degrees of rotation error; at
            // ~0.2 px (subpix) it is ~2 deg / ~2 mm per frame (see MarkerPnP notes).
            detParams.set_cornerRefinementMethod(Objdetect.CORNER_REFINE_SUBPIX);
            _detector = new ArucoDetector(dict, detParams, new RefineParameters());
            _idsOut = new Mat();
            Status = "ArUco detector ready (DICT_4X4_50, subpix corners)";
        }

        private void OnDisable()
        {
            _rgbaMat?.Dispose(); _rgbaMat = null;
            _grayMat?.Dispose(); _grayMat = null;
            _smallMat?.Dispose(); _smallMat = null;
            _idsOut?.Dispose(); _idsOut = null;
            _detector?.Dispose(); _detector = null;
            if (_readbackTex != null) { Destroy(_readbackTex); _readbackTex = null; }
        }

        public bool TryDetect(Texture source, List<MarkerObservation> results)
        {
            if (_detector == null || source == null) return false;

            int w = source.width, h = source.height;
            if (w <= 16 || h <= 16) { Status = "Source texture too small"; return false; }

            if (_rgbaMat == null || _rgbaMat.cols() != w || _rgbaMat.rows() != h)
            {
                _rgbaMat?.Dispose();
                _grayMat?.Dispose();
                _rgbaMat = new Mat(h, w, CvType.CV_8UC4);
                _grayMat = new Mat(h, w, CvType.CV_8UC1);
            }

            if (!FillRgbaMat(source)) return false;
            Imgproc.cvtColor(_rgbaMat, _grayMat, Imgproc.COLOR_RGBA2GRAY);

            Mat detectMat = _grayMat;
            float scale = 1f;
            if (_downscale > 1)
            {
                int sw = w / _downscale, sh = h / _downscale;
                if (_smallMat == null || _smallMat.cols() != sw || _smallMat.rows() != sh)
                {
                    _smallMat?.Dispose();
                    _smallMat = new Mat(sh, sw, CvType.CV_8UC1);
                }
                Imgproc.resize(_grayMat, _smallMat, _smallMat.size());
                detectMat = _smallMat;
                scale = _downscale;
            }

            foreach (var m in _cornersOut) m.Dispose();
            _cornersOut.Clear();
            _detector.detectMarkers(detectMat, _cornersOut, _idsOut);

            int count = _cornersOut.Count;
            var cornerVals = new float[8];
            for (int i = 0; i < count; i++)
            {
                _cornersOut[i].get(0, 0, cornerVals);
                int id = (int)_idsOut.get(i, 0)[0];
                var obs = new MarkerObservation { Id = id };
                for (int c = 0; c < 4; c++)
                {
                    float x = cornerVals[c * 2] * scale;
                    float y = cornerVals[c * 2 + 1] * scale;
                    if (_flipY) y = h - 1 - y;
                    switch (c)
                    {
                        case 0: obs.C0 = new Vector2(x, y); break;
                        case 1: obs.C1 = new Vector2(x, y); break;
                        case 2: obs.C2 = new Vector2(x, y); break;
                        case 3: obs.C3 = new Vector2(x, y); break;
                    }
                }
                results.Add(obs);
            }

            Status = $"detected {count} marker(s)";
            return true;
        }

        private bool FillRgbaMat(Texture source)
        {
            switch (source)
            {
                case WebCamTexture wct:
                    Utils.webCamTextureToMat(wct, _rgbaMat);
                    return true;
                case Texture2D t2d:
                    Utils.texture2DToMat(t2d, _rgbaMat);
                    return true;
                default:
                    // RenderTexture / external texture: GPU blit + readback (slow path).
                    var prev = RenderTexture.active;
                    var rt = RenderTexture.GetTemporary(source.width, source.height, 0,
                                                        RenderTextureFormat.ARGB32);
                    Graphics.Blit(source, rt);
                    RenderTexture.active = rt;
                    if (_readbackTex == null || _readbackTex.width != source.width ||
                        _readbackTex.height != source.height)
                        _readbackTex = new Texture2D(source.width, source.height,
                                                     TextureFormat.RGBA32, false);
                    _readbackTex.ReadPixels(new UnityEngine.Rect(0, 0, source.width, source.height), 0, 0);
                    _readbackTex.Apply(false);
                    RenderTexture.active = prev;
                    RenderTexture.ReleaseTemporary(rt);
                    Utils.texture2DToMat(_readbackTex, _rgbaMat);
                    return true;
            }
        }
#else
        public bool IsReady => false;

        private void OnEnable()
        {
            Status = "OpenCV for Unity not installed (import the asset and add the " +
                     "OPENCV_FOR_UNITY scripting define)";
            Debug.LogWarning($"[ArUcoCornerProvider] {Status}");
        }

        public bool TryDetect(Texture source, List<MarkerObservation> results) => false;
#endif
    }
}
