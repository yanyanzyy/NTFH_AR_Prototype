using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;

namespace ARArmDetection
{
    /// <summary>
    /// Movable world-space control panel for the arm overlay.
    ///
    /// Buttons (index-pinch a button and hold briefly — the fill bar shows progress):
    ///   OVERLAY ON    — resume the overlay pipeline (ArmDetectionManager.SetOverlayEnabled(true))
    ///   OVERLAY OFF   — hide the overlay and suspend detection (lock is kept)
    ///   RE-DETECT ARM — shown only while the overlay is ON and locked onto an arm;
    ///                   releases the lock so a fresh arm can be acquired and overlaid.
    ///                   This replaces the old thumb+middle pinch-hold unlock gesture.
    ///
    /// MOVING THE PANEL: index-pinch the title bar (directly with the fingertip, or from a
    /// distance with the hand pointer ray) and drag. The panel follows the camera until the
    /// first grab, then stays where it was placed and keeps facing the viewer.
    ///
    /// Hand input is read straight from OVRHand (pointer pose + index pinch), the same
    /// proven path as FacilitatorHandPanelDrag — the scene has no XRI ray interactors, so
    /// plain Unity Buttons would never receive hand clicks. The Unity Button components on
    /// each row exist only so the panel is clickable with the mouse in the Editor.
    ///
    /// Auto-created in any arm-detection scene by ArmOverlayControlPanelBootstrap.
    /// </summary>
    public class ArmOverlayControlPanel : MonoBehaviour
    {
        [SerializeField] private ArmDetectionManager _manager;

        [Header("Placement (follows the view until first grab)")]
        [SerializeField] private float _distanceMeters = 1.3f;
        [SerializeField] private float _heightInViewMeters = 0.15f;
        [Tooltip("Negative = left of view. The status panels sit on the right (+0.48).")]
        [SerializeField] private float _rightInViewMeters = -0.5f;

        [Header("Hand interaction")]
        [Tooltip("How long a button must be pinch-held before it fires. The fill bar shows progress.")]
        [SerializeField, Range(0.2f, 2f)] private float _holdDurationSeconds = 0.6f;
        [Tooltip("Fingertip distance (m) that counts as touching the panel for direct grab/press.")]
        [SerializeField] private float _directGrabDistance = 0.12f;
        [SerializeField] private float _maximumRayDistance = 4f;
        [SerializeField] private float _handSearchInterval = 3f;

        private class PanelButton
        {
            public RectTransform Rect;
            public Image Background;
            public Image HoldFill;
            public Action OnActivate;
            public Func<bool> Visible;
            public float HoldTimer;
            public bool RequiresRelease;
            public float LastActivatedTime = -10f;
        }

        private readonly List<PanelButton> _buttons = new();
        private bool[] _holdActiveScratch;   // per-frame button hold flags, allocated once
        private readonly List<OVRHand> _hands = new();
        private readonly Dictionary<OVRHand, Transform> _indexTips = new();

        private RectTransform _canvasRect;
        private RectTransform _titleRect;
        private Text _statusLabel;
        private RectTransform _cursor;
        private Image _cursorImage;
        private LineRenderer _pointerLine;
        private Material _pointerLineMaterial;

        private OVRHand _grabHand;
        private bool _directGrab;
        private float _grabRayDistance;
        private Vector3 _grabOffset;
        private bool _placedByUser;

        private float _nextHandSearchTime;
        private float _nextVisualRefresh;
        private string _lastStatusText;

        private const float ActivateCooldownSeconds = 0.4f;
        private const float VisualRefreshInterval = 0.25f;

        private static readonly Color PanelColor      = new Color(0.055f, 0.065f, 0.075f, 0.96f);
        private static readonly Color TitleColor      = new Color(0.12f, 0.16f, 0.22f, 0.98f);
        private static readonly Color OnActiveColor   = new Color(0.13f, 0.55f, 0.24f, 0.95f);
        private static readonly Color OnIdleColor     = new Color(0.16f, 0.26f, 0.19f, 0.92f);
        private static readonly Color OffActiveColor  = new Color(0.62f, 0.18f, 0.16f, 0.95f);
        private static readonly Color OffIdleColor    = new Color(0.27f, 0.20f, 0.20f, 0.92f);
        private static readonly Color RedetectColor   = new Color(0.85f, 0.50f, 0.08f, 0.95f);
        private static readonly Color HoldFillColor   = new Color(1f, 1f, 1f, 0.30f);

        private void Awake()
        {
            if (_manager == null) _manager = FindFirstObjectByType<ArmDetectionManager>();
            BuildUI();
            EnsureEventSystem();
        }

        private void OnDestroy()
        {
            if (_pointerLineMaterial != null) Destroy(_pointerLineMaterial);
        }

        private void Update()
        {
            if (Time.unscaledTime >= _nextHandSearchTime) FindTrackedHands();

            UpdateButtonVisibility();
            UpdateHandInteraction();

            if (Time.unscaledTime >= _nextVisualRefresh)
            {
                _nextVisualRefresh = Time.unscaledTime + VisualRefreshInterval;
                RefreshVisuals();
            }
        }

        private void LateUpdate()
        {
            if (!_placedByUser && _grabHand == null) FollowCamera();
        }

        // ── Button actions ─────────────────────────────────────────────────────────────

        public void OverlayOn()  { if (_manager != null) _manager.SetOverlayEnabled(true); }
        public void OverlayOff() { if (_manager != null) _manager.SetOverlayEnabled(false); }

        /// <summary>Releases the arm lock so the manager re-detects the physical arm and
        /// locks/overlays again. Replaces the thumb+middle pinch-hold gesture.</summary>
        public void RedetectArm() { if (_manager != null) _manager.Unlock(); }

        private bool RedetectAvailable() =>
            _manager != null && _manager.OverlayEnabled && _manager.IsLocked;

        // ── Hand interaction ───────────────────────────────────────────────────────────

        private void UpdateHandInteraction()
        {
            bool cursorShown = false;
            bool lineShown = false;
            bool anyPinching = false;
            bool[] holdActive = _holdActiveScratch;
            Array.Clear(holdActive, 0, holdActive.Length);

            // Active grab first: keep moving the panel while the grabbing pinch is held.
            if (_grabHand != null)
            {
                bool stillPinching = IsTracked(_grabHand) && IsIndexPinching(_grabHand);
                if (!stillPinching)
                {
                    _grabHand = null;
                }
                else if (_directGrab && TryGetIndexTip(_grabHand, out Vector3 tip))
                {
                    transform.position = tip + _grabOffset;
                    FaceViewer();
                    ShowCursor(NearestTitlePoint(tip), true);
                    cursorShown = true;
                }
                else if (!_directGrab && TryGetPointerRay(_grabHand, out Ray grabRay))
                {
                    Vector3 target = grabRay.GetPoint(_grabRayDistance);
                    transform.position = target + _grabOffset;
                    FaceViewer();
                    ShowCursor(target, true);
                    cursorShown = true;
                }
            }

            foreach (OVRHand hand in _hands)
            {
                if (hand == null || !IsTracked(hand)) continue;

                bool pinching = IsIndexPinching(hand);
                anyPinching |= pinching;
                if (hand == _grabHand) continue;

                bool hasRay = TryGetPointerRay(hand, out Ray ray);

                // Buttons: hovering (fingertip or ray) + pinching charges the hold.
                for (int i = 0; i < _buttons.Count; i++)
                {
                    PanelButton b = _buttons[i];
                    if (!b.Rect.gameObject.activeInHierarchy) continue;

                    bool directHover = TryGetRectDirectHover(hand, b.Rect, out Vector3 directPoint);
                    Vector3 rayPoint = default;
                    bool rayHover = hasRay && TryGetRectRayHover(ray, b.Rect, out rayPoint, out _);
                    if (!directHover && !rayHover) continue;

                    Vector3 point = directHover ? directPoint : rayPoint;
                    if (_grabHand == null && pinching) holdActive[i] = true;
                    if (!cursorShown)
                    {
                        ShowCursor(point, pinching);
                        cursorShown = true;
                    }
                    if (rayHover && !lineShown)
                    {
                        ShowPointerLine(ray, point, true);
                        lineShown = true;
                    }
                }

                // Title bar: pinch to grab and move the panel.
                bool titleDirect = TryGetRectDirectHover(hand, _titleRect, out Vector3 titleDirectPoint);
                Vector3 titleRayPoint = default;
                float titleRayDist = 0f;
                bool titleRay = hasRay && TryGetRectRayHover(ray, _titleRect, out titleRayPoint, out titleRayDist);
                if (titleDirect || titleRay)
                {
                    if (!cursorShown)
                    {
                        ShowCursor(titleDirect ? titleDirectPoint : titleRayPoint, pinching);
                        cursorShown = true;
                    }
                    if (titleRay && !lineShown)
                    {
                        ShowPointerLine(ray, titleRayPoint, true);
                        lineShown = true;
                    }

                    if (_grabHand == null && pinching)
                    {
                        if (titleDirect && TryGetIndexTip(hand, out Vector3 grabTip))
                        {
                            _grabHand = hand;
                            _directGrab = true;
                            _grabOffset = transform.position - grabTip;
                            _placedByUser = true;
                        }
                        else if (titleRay)
                        {
                            _grabHand = hand;
                            _directGrab = false;
                            _grabRayDistance = titleRayDist;
                            _grabOffset = transform.position - titleRayPoint;
                            _placedByUser = true;
                        }
                    }
                }
            }

            for (int i = 0; i < _buttons.Count; i++)
                UpdateButtonHold(_buttons[i], holdActive[i], anyPinching);

            if (!cursorShown && _cursor != null) _cursor.gameObject.SetActive(false);
            if (!lineShown && _pointerLine != null) _pointerLine.enabled = false;
        }

        /// <summary>Hold-to-activate: charging fill while pinch-hovered; fires once at full,
        /// then requires the pinch to be released before it can charge again.</summary>
        private void UpdateButtonHold(PanelButton button, bool active, bool anyPinching)
        {
            if (!anyPinching) button.RequiresRelease = false;

            if (!active || button.RequiresRelease)
            {
                button.HoldTimer = 0f;
                if (button.HoldFill != null) button.HoldFill.fillAmount = 0f;
                return;
            }

            button.HoldTimer += Time.unscaledDeltaTime;
            if (button.HoldFill != null)
                button.HoldFill.fillAmount = Mathf.Clamp01(button.HoldTimer / _holdDurationSeconds);

            if (button.HoldTimer < _holdDurationSeconds) return;

            button.HoldTimer = 0f;
            button.RequiresRelease = true;
            if (button.HoldFill != null) button.HoldFill.fillAmount = 0f;
            Activate(button);
        }

        /// <summary>Single activation entry point for both the hand-hold path and the Editor
        /// mouse Button path, with a short cooldown so they can never double-fire.</summary>
        private void Activate(PanelButton button)
        {
            if (Time.unscaledTime - button.LastActivatedTime < ActivateCooldownSeconds) return;
            button.LastActivatedTime = Time.unscaledTime;
            button.OnActivate?.Invoke();
            RefreshVisuals();
        }

        private void UpdateButtonVisibility()
        {
            foreach (PanelButton b in _buttons)
            {
                bool visible = b.Visible == null || b.Visible();
                if (b.Rect.gameObject.activeSelf != visible)
                {
                    b.Rect.gameObject.SetActive(visible);
                    if (!visible)
                    {
                        b.HoldTimer = 0f;
                        if (b.HoldFill != null) b.HoldFill.fillAmount = 0f;
                    }
                }
            }
        }

        // ── Visuals ────────────────────────────────────────────────────────────────────

        private void RefreshVisuals()
        {
            bool overlayOn = _manager != null && _manager.OverlayEnabled;
            bool locked = _manager != null && _manager.IsLocked;

            if (_buttons.Count >= 2)
            {
                _buttons[0].Background.color = overlayOn ? OnActiveColor : OnIdleColor;
                _buttons[1].Background.color = overlayOn ? OffIdleColor : OffActiveColor;
            }

            string status;
            if (_manager == null)
                status = "ArmDetectionManager not found";
            else if (!overlayOn)
                status = "Overlay OFF — detection paused\n(lock kept; ON resumes in place)";
            else if (locked)
                status = $"{_manager.LockStatus}\nRE-DETECT ARM re-acquires";
            else
                status = _manager.LockStatus;

            if (status != _lastStatusText && _statusLabel != null)
            {
                _lastStatusText = status;
                _statusLabel.text = $"<size=15>{status}</size>";
            }
        }

        private void FollowCamera()
        {
            var cam = Camera.main;
            if (cam == null) return;

            transform.position = cam.transform.position
                + cam.transform.forward * _distanceMeters
                + cam.transform.up * _heightInViewMeters
                + cam.transform.right * _rightInViewMeters;
            FaceViewer();
        }

        private void FaceViewer()
        {
            var cam = Camera.main;
            if (cam == null) return;
            Vector3 facing = transform.position - cam.transform.position;
            if (facing.sqrMagnitude < 0.0001f) return;
            transform.rotation = Quaternion.LookRotation(facing.normalized, cam.transform.up);
        }

        private void ShowCursor(Vector3 worldPoint, bool pinching)
        {
            if (_cursor == null) return;
            Vector3 local = _canvasRect.InverseTransformPoint(worldPoint);
            _cursor.localPosition = new Vector3(local.x, local.y, -8f);
            _cursor.gameObject.SetActive(true);
            if (_cursorImage != null)
                _cursorImage.color = pinching
                    ? new Color(0.15f, 1f, 0.45f, 1f)
                    : new Color(0.2f, 0.9f, 0.82f, 0.95f);
        }

        private void ShowPointerLine(Ray ray, Vector3 endPoint, bool hovering)
        {
            if (_pointerLine == null) return;
            Color color = hovering
                ? new Color(0.15f, 1f, 0.55f, 0.95f)
                : new Color(0.2f, 0.75f, 0.9f, 0.55f);
            _pointerLine.startColor = color;
            _pointerLine.endColor = color;
            _pointerLine.SetPosition(0, ray.origin + ray.direction * 0.05f);
            _pointerLine.SetPosition(1, endPoint);
            _pointerLine.enabled = true;
        }

        private Vector3 NearestTitlePoint(Vector3 worldPoint)
        {
            var plane = new Plane(_titleRect.forward, _titleRect.position);
            return worldPoint - _titleRect.forward * plane.GetDistanceToPoint(worldPoint);
        }

        // ── Hand helpers (same approach as FacilitatorHandPanelDrag) ──────────────────

        private void FindTrackedHands()
        {
            _nextHandSearchTime = Time.unscaledTime + _handSearchInterval;
            var found = FindObjectsByType<OVRHand>(FindObjectsInactive.Include, FindObjectsSortMode.None);

            _hands.Clear();
            _indexTips.Clear();
            foreach (OVRHand hand in found)
            {
                if (hand == null) continue;
                _hands.Add(hand);

                OVRSkeleton skeleton = hand.GetComponent<OVRSkeleton>();
                if (skeleton == null) skeleton = hand.GetComponentInParent<OVRSkeleton>();
                Transform tip = FindBone(skeleton, OVRSkeleton.BoneId.Hand_IndexTip);
                if (tip != null) _indexTips[hand] = tip;
            }
        }

        private bool TryGetIndexTip(OVRHand hand, out Vector3 position)
        {
            position = default;
            if (!IsTracked(hand)) return false;

            if (!_indexTips.TryGetValue(hand, out Transform tip) || tip == null)
            {
                OVRSkeleton skeleton = hand.GetComponent<OVRSkeleton>();
                if (skeleton == null) skeleton = hand.GetComponentInParent<OVRSkeleton>();
                tip = FindBone(skeleton, OVRSkeleton.BoneId.Hand_IndexTip);
                if (tip == null) return false;
                _indexTips[hand] = tip;
            }

            position = tip.position;
            return true;
        }

        private bool TryGetRectDirectHover(OVRHand hand, RectTransform rect, out Vector3 point)
        {
            point = default;
            if (rect == null || !rect.gameObject.activeInHierarchy ||
                !TryGetIndexTip(hand, out Vector3 tip)) return false;

            var plane = new Plane(rect.forward, rect.position);
            float signedDistance = plane.GetDistanceToPoint(tip);
            if (Mathf.Abs(signedDistance) > _directGrabDistance) return false;

            point = tip - rect.forward * signedDistance;
            Vector3 local = rect.InverseTransformPoint(point);
            return rect.rect.Contains(new Vector2(local.x, local.y));
        }

        private bool TryGetRectRayHover(Ray ray, RectTransform rect, out Vector3 point, out float distance)
        {
            point = default;
            distance = 0f;
            if (rect == null || !rect.gameObject.activeInHierarchy ||
                ray.direction.sqrMagnitude < 0.001f) return false;

            var plane = new Plane(rect.forward, rect.position);
            if (!plane.Raycast(ray, out distance) || distance < 0f || distance > _maximumRayDistance)
                return false;

            point = ray.GetPoint(distance);
            Vector3 local = rect.InverseTransformPoint(point);
            return rect.rect.Contains(new Vector2(local.x, local.y));
        }

        private static Transform FindBone(OVRSkeleton skeleton, OVRSkeleton.BoneId id)
        {
            if (skeleton == null || !skeleton.IsInitialized || skeleton.Bones == null) return null;
            foreach (OVRBone bone in skeleton.Bones)
                if (bone != null && bone.Id == id) return bone.Transform;
            return null;
        }

        private static bool IsTracked(OVRHand hand) =>
            hand != null && hand.IsTracked && hand.IsDataValid;

        private static bool IsIndexPinching(OVRHand hand) =>
            hand.GetFingerIsPinching(OVRHand.HandFinger.Index);

        private static bool TryGetPointerRay(OVRHand hand, out Ray ray)
        {
            ray = default;
            if (!IsTracked(hand) || !hand.IsPointerPoseValid || hand.PointerPose == null) return false;
            Transform pointer = hand.PointerPose;
            ray = new Ray(pointer.position, pointer.rotation * Vector3.forward);
            return true;
        }

        // ── UI construction ────────────────────────────────────────────────────────────

        private void BuildUI()
        {
            var canvasGO = new GameObject("ArmOverlayControlCanvas");
            canvasGO.transform.SetParent(transform, false);

            var canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();
            canvasGO.AddComponent<TrackedDeviceGraphicRaycaster>();

            _canvasRect = (RectTransform)canvasGO.transform;
            _canvasRect.sizeDelta = new Vector2(300, 340);
            _canvasRect.localScale = Vector3.one * 0.0017f;

            CreateImage("Background", _canvasRect, new Vector2(300, 340), Vector2.zero, PanelColor);

            // Title bar — the grab handle for moving the panel.
            var titleBg = CreateImage("TitleBar", _canvasRect, new Vector2(300, 50), new Vector2(0, 145), TitleColor);
            _titleRect = (RectTransform)titleBg.transform;
            CreateLabel(_titleRect, "<b>ARM OVERLAY</b>\n<size=12>pinch here to move panel</size>", 17);

            _buttons.Add(CreateButton("OverlayOnButton", new Vector2(0, 77),
                "<b>OVERLAY ON</b>", OnIdleColor, OverlayOn, null));
            _buttons.Add(CreateButton("OverlayOffButton", new Vector2(0, 1),
                "<b>OVERLAY OFF</b>", OffIdleColor, OverlayOff, null));
            _buttons.Add(CreateButton("RedetectButton", new Vector2(0, -75),
                "<b>RE-DETECT ARM</b>\n<size=12>unlock & re-acquire</size>", RedetectColor,
                RedetectArm, RedetectAvailable));

            var statusGO = new GameObject("Status", typeof(RectTransform), typeof(Text));
            statusGO.transform.SetParent(_canvasRect, false);
            var statusRT = (RectTransform)statusGO.transform;
            statusRT.sizeDelta = new Vector2(284, 52);
            statusRT.anchoredPosition = new Vector2(0, -140);
            _statusLabel = statusGO.GetComponent<Text>();
            ConfigureText(_statusLabel, 15);

            // Cursor dot (feedback for fingertip / ray hover).
            var cursorImage = CreateImage("HandCursor", _canvasRect, new Vector2(16, 16),
                Vector2.zero, new Color(0.2f, 0.9f, 0.82f, 0.95f));
            _cursor = (RectTransform)cursorImage.transform;
            _cursorImage = cursorImage;
            cursorImage.raycastTarget = false;
            _cursor.gameObject.SetActive(false);

            _holdActiveScratch = new bool[_buttons.Count];
            BuildPointerLine();
            RefreshVisuals();
        }

        private PanelButton CreateButton(string name, Vector2 position, string label,
            Color color, Action onActivate, Func<bool> visible)
        {
            Image bg = CreateImage(name, _canvasRect, new Vector2(280, 66), position, color);
            var rect = (RectTransform)bg.transform;

            // Editor-mouse support only; on device the OVRHand hold path activates it.
            var uiButton = bg.gameObject.AddComponent<Button>();
            uiButton.targetGraphic = bg;
            uiButton.transition = Selectable.Transition.None;

            var fillGO = new GameObject("HoldFill", typeof(RectTransform), typeof(Image));
            fillGO.transform.SetParent(rect, false);
            var fillRT = (RectTransform)fillGO.transform;
            fillRT.anchorMin = Vector2.zero;
            fillRT.anchorMax = Vector2.one;
            fillRT.offsetMin = Vector2.zero;
            fillRT.offsetMax = Vector2.zero;
            var fill = fillGO.GetComponent<Image>();
            fill.color = HoldFillColor;
            fill.raycastTarget = false;
            fill.type = Image.Type.Filled;
            fill.fillMethod = Image.FillMethod.Horizontal;
            fill.fillOrigin = (int)Image.OriginHorizontal.Left;
            fill.fillAmount = 0f;
            // Filled type needs a sprite to render; a plain white one works.
            fill.sprite = Sprite.Create(Texture2D.whiteTexture,
                new Rect(0, 0, 4, 4), new Vector2(0.5f, 0.5f));

            CreateLabel(rect, label, 20);

            var button = new PanelButton
            {
                Rect = rect,
                Background = bg,
                HoldFill = fill,
                OnActivate = onActivate,
                Visible = visible,
            };
            uiButton.onClick.AddListener(() => Activate(button));
            return button;
        }

        private static Image CreateImage(string name, RectTransform parent, Vector2 size,
            Vector2 position, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.sizeDelta = size;
            rt.anchoredPosition = position;
            var image = go.GetComponent<Image>();
            image.color = color;
            return image;
        }

        private static Text CreateLabel(RectTransform parent, string content, int fontSize)
        {
            var go = new GameObject("Label", typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(6, 4);
            rt.offsetMax = new Vector2(-6, -4);
            var text = go.GetComponent<Text>();
            ConfigureText(text, fontSize);
            text.text = content;
            return text;
        }

        private static void ConfigureText(Text text, int fontSize)
        {
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.color = Color.white;
            text.alignment = TextAnchor.MiddleCenter;
            text.supportRichText = true;
            text.raycastTarget = false;
            text.horizontalOverflow = HorizontalWrapMode.Overflow;
            text.verticalOverflow = VerticalWrapMode.Overflow;
        }

        private void BuildPointerLine()
        {
            var go = new GameObject("OverlayPanelPointerLine");
            go.transform.SetParent(transform, false);
            _pointerLine = go.AddComponent<LineRenderer>();
            _pointerLine.positionCount = 2;
            _pointerLine.useWorldSpace = true;
            _pointerLine.startWidth = 0.003f;
            _pointerLine.endWidth = 0.0015f;
            _pointerLine.numCapVertices = 4;
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Sprites/Default");
            _pointerLineMaterial = new Material(shader);
            _pointerLine.sharedMaterial = _pointerLineMaterial;
            _pointerLine.enabled = false;
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null) return;
            var go = new GameObject("XR EventSystem", typeof(EventSystem), typeof(XRUIInputModule));
            DontDestroyOnLoad(go);
        }
    }

    /// <summary>
    /// Auto-creates the overlay control panel in any scene that runs arm detection, so
    /// existing scenes get it without re-running the editor setup (same pattern as
    /// ArmLockButtonBootstrap / FacilitatorModeBootstrap).
    /// </summary>
    public static class ArmOverlayControlPanelBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateForArmDetectionScene()
        {
            if (UnityEngine.Object.FindFirstObjectByType<ArmDetectionManager>() == null) return;
            if (UnityEngine.Object.FindFirstObjectByType<ArmOverlayControlPanel>() != null) return;

            var go = new GameObject("ArmOverlayControlPanel");
            go.AddComponent<ArmOverlayControlPanel>();
        }
    }
}
