using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.XR.Interaction.Toolkit.UI;

namespace ARArmDetection.Facilitator
{
    public class FacilitatorModeController : MonoBehaviour
    {
        [Header("Procedure")]
        [SerializeField] private FacilitatorProcedure _procedure;
        [SerializeField] private bool _autoStart = true;
        [SerializeField] private bool _playNarrationOnStepChange = true;

        [Header("Placement")]
        [Tooltip("Initial distance to the left of the ARM DETECTION panel. The facilitator panel is then detached and fixed in world space.")]
        [SerializeField] private float _statusPanelLeftOffsetMeters = 0.84f;

        [Header("Gaze dwell")]
        [SerializeField] private bool _enableGazeDwell = false;
        [SerializeField, Range(0.5f, 3f)] private float _gazeDwellSeconds = 1.5f;

        private AudioSource _audioSource;
        private GameObject _panelRoot;
        private Text _procedureText;
        private Text _progressText;
        private Text _titleText;
        private Text _instructionText;
        private Text _audioStatusText;
        private Text _nextButtonText;
        private Image _progressFill;
        private Material _panelAlwaysOnTopMaterial;
        private Texture2D _dwellCircleTexture;
        private Sprite _dwellCircleSprite;
        private RectTransform _nextButtonRect;
        private Image _gazeDwellFill;
        private RectTransform _previousButtonRect;
        private Image _previousHoldFill;
        private GameObject _previousButtonObject;
        private int _stepIndex = -1;
        private bool _narrationPaused;
        private bool _waitingForNarrationEnd;
        private bool _completed;
        private bool _gazeRequiresExit;
        private float _gazeDwellTimer;
        private ARArmDetection.DetectionModeButton _statusPanel;
        private bool _panelPlacedInWorld;
        private Plane _panelDragPlane;
        private Vector3 _panelDragOffset;

        public event Action<int, FacilitatorStep> StepChanged;
        public event Action ProcedureCompleted;

        public FacilitatorProcedure Procedure => _procedure;
        public int CurrentStepIndex => _stepIndex;
        public FacilitatorStep CurrentStep => IsValidStep(_stepIndex) ? _procedure.Steps[_stepIndex] : null;

        public void Initialize(FacilitatorProcedure procedure)
        {
            _procedure = procedure;
            if (_autoStart && _stepIndex < 0) StartProcedure();
            else RefreshUI();
        }

        private void Awake()
        {
            _audioSource = gameObject.AddComponent<AudioSource>();
            _audioSource.playOnAwake = false;
            _audioSource.spatialBlend = 0f;
            BuildUI();
            EnsureEventSystem();
        }

        private void OnDestroy()
        {
            if (_panelAlwaysOnTopMaterial != null)
                Destroy(_panelAlwaysOnTopMaterial);
            if (_dwellCircleSprite != null)
                Destroy(_dwellCircleSprite);
            if (_dwellCircleTexture != null)
                Destroy(_dwellCircleTexture);
        }

        private void Start()
        {
            if (_autoStart && _stepIndex < 0) StartProcedure();
        }

        private void Update()
        {
            HandleControllerInput();
            UpdateGazeDwell();

            if (_waitingForNarrationEnd && !_audioSource.isPlaying && !_narrationPaused)
            {
                _waitingForNarrationEnd = false;
                if (CurrentStep != null && CurrentStep.AdvanceMode == FacilitatorAdvanceMode.AfterNarration)
                    NextStep();
                else
                    RefreshAudioStatus();
            }
        }

        private void LateUpdate()
        {
            PlacePanelInWorldOnce();
        }

        public void StartProcedure()
        {
            _completed = false;
            SetStep(0);
        }

        public void PreviousStep()
        {
            if (_procedure == null || _procedure.Steps.Count == 0) return;
            _completed = false;
            SetStep(Mathf.Max(0, _stepIndex - 1));
        }

        public void NextStep()
        {
            if (_procedure == null || _procedure.Steps.Count == 0) return;
            if (_completed)
            {
                StartProcedure();
                return;
            }

            if (_stepIndex < _procedure.Steps.Count - 1)
            {
                SetStep(_stepIndex + 1);
                return;
            }

            CompleteProcedure();
        }

        public void RepeatNarration()
        {
            PlayCurrentNarration();
        }

        public void TogglePause()
        {
            if (_audioSource.clip == null) return;

            if (_narrationPaused)
            {
                _audioSource.UnPause();
                _narrationPaused = false;
            }
            else if (_audioSource.isPlaying)
            {
                _audioSource.Pause();
                _narrationPaused = true;
            }

            RefreshAudioStatus();
        }

        public void SignalCompletion(string signal)
        {
            var step = CurrentStep;
            if (step == null || step.AdvanceMode != FacilitatorAdvanceMode.ExternalSignal) return;
            if (string.Equals(step.CompletionSignal, signal, StringComparison.OrdinalIgnoreCase))
                NextStep();
        }

        private void SetStep(int index)
        {
            if (!IsValidStep(index))
            {
                RefreshUI();
                return;
            }

            _stepIndex = index;
            _completed = false;
            _audioSource.Stop();
            _audioSource.clip = null;
            _narrationPaused = false;
            _waitingForNarrationEnd = false;
            ResetGazeDwell(true);
            RefreshUI();

            if (_playNarrationOnStepChange) PlayCurrentNarration();
            StepChanged?.Invoke(_stepIndex, CurrentStep);
        }

        private void CompleteProcedure()
        {
            _audioSource.Stop();
            _waitingForNarrationEnd = false;
            _narrationPaused = false;
            _completed = true;
            ResetGazeDwell(true);
            RefreshUI();
            ProcedureCompleted?.Invoke();
        }

        private void PlayCurrentNarration()
        {
            var clip = CurrentStep?.Narration;
            _audioSource.Stop();
            _audioSource.clip = clip;
            _narrationPaused = false;
            _waitingForNarrationEnd = false;

            if (clip != null)
            {
                _audioSource.Play();
                _waitingForNarrationEnd = true;
            }

            RefreshAudioStatus();
        }

        private void RefreshUI()
        {
            int count = _procedure != null ? _procedure.Steps.Count : 0;
            if (_procedureText != null)
                _procedureText.text = _procedure != null ? _procedure.ProcedureTitle : "FACILITATOR MODE";

            if (_completed)
            {
                if (_previousButtonObject != null) _previousButtonObject.SetActive(false);
                if (_progressText != null) _progressText.text = $"COMPLETE  {count}/{count}";
                if (_titleText != null) _titleText.text = "Procedure complete";
                if (_instructionText != null) _instructionText.text = "All guided performance steps have been completed.";
                if (_progressFill != null) _progressFill.fillAmount = 1f;
                if (_nextButtonText != null) _nextButtonText.text = "Restart";
                RefreshAudioStatus();
                return;
            }

            var step = CurrentStep;
            if (step == null)
            {
                if (_previousButtonObject != null) _previousButtonObject.SetActive(false);
                if (_progressText != null) _progressText.text = "NO PROCEDURE";
                if (_titleText != null) _titleText.text = "Procedure unavailable";
                if (_instructionText != null) _instructionText.text = "Assign a FacilitatorProcedure asset.";
                if (_progressFill != null) _progressFill.fillAmount = 0f;
                RefreshAudioStatus();
                return;
            }

            if (_previousButtonObject != null)
                _previousButtonObject.SetActive(_stepIndex > 0);
            if (_progressText != null) _progressText.text = $"STEP {_stepIndex + 1} OF {count}  |  {step.Id}";
            if (_titleText != null) _titleText.text = step.Title;
            if (_instructionText != null) _instructionText.text = step.Instruction;
            if (_progressFill != null) _progressFill.fillAmount = count > 0 ? (_stepIndex + 1f) / count : 0f;
            if (_nextButtonText != null)
                _nextButtonText.text = _stepIndex == count - 1 ? "Finish" : "Next step";
            RefreshAudioStatus();
        }

        private void RefreshAudioStatus()
        {
            if (_audioStatusText != null)
            {
                if (CurrentStep?.Narration == null) _audioStatusText.text = "Narration not assigned";
                else if (_narrationPaused) _audioStatusText.text = "Narration paused";
                else if (_audioSource.isPlaying) _audioStatusText.text = "Narration playing";
                else _audioStatusText.text = "Narration ready";
            }

        }

        private bool IsValidStep(int index) =>
            _procedure != null && index >= 0 && index < _procedure.Steps.Count;

        private void HandleControllerInput()
        {
            var keyboard = Keyboard.current;
            if (keyboard == null) return;
            if (keyboard.rightArrowKey.wasPressedThisFrame) NextStep();
        }

        private void UpdateGazeDwell()
        {
            if (!_enableGazeDwell || _nextButtonRect == null) return;

            var cam = Camera.main;
            if (cam == null) return;

            Rect pixelRect = cam.pixelRect;
            Vector2 gazePoint = new Vector2(pixelRect.center.x, pixelRect.center.y);
            bool hovering = RectTransformUtility.RectangleContainsScreenPoint(
                _nextButtonRect, gazePoint, cam);

            if (!hovering)
            {
                _gazeRequiresExit = false;
                ResetGazeDwell(false);
                return;
            }

            if (_gazeRequiresExit) return;

            _gazeDwellTimer += Time.unscaledDeltaTime;
            if (_gazeDwellFill != null)
                _gazeDwellFill.fillAmount = Mathf.Clamp01(_gazeDwellTimer / _gazeDwellSeconds);

            if (_gazeDwellTimer < _gazeDwellSeconds) return;

            _gazeRequiresExit = true;
            ResetGazeDwell(true);
            NextStep();
        }

        private void ResetGazeDwell(bool preserveExitRequirement)
        {
            _gazeDwellTimer = 0f;
            if (_gazeDwellFill != null) _gazeDwellFill.fillAmount = 0f;
            if (!preserveExitRequirement) _gazeRequiresExit = false;
        }

        public void BeginPanelDrag(PointerEventData eventData)
        {
            if (!_panelPlacedInWorld) PlacePanelInWorldOnce();

            Vector3 normal = transform.forward.sqrMagnitude > 0.001f
                ? transform.forward.normalized
                : Vector3.forward;
            _panelDragPlane = new Plane(normal, transform.position);
            _panelDragOffset = Vector3.zero;

            if (TryGetPointerPlanePoint(eventData, out Vector3 hitPoint))
                _panelDragOffset = transform.position - hitPoint;
        }

        public void DragPanel(PointerEventData eventData)
        {
            if (TryGetPointerPlanePoint(eventData, out Vector3 hitPoint))
            {
                transform.position = hitPoint + _panelDragOffset;
                FacePanelTowardViewer();
            }
        }

        public void EndPanelDrag()
        {
            FacePanelTowardViewer();
            _panelPlacedInWorld = true;
        }

        public void FacePanelTowardViewer()
        {
            var cam = Camera.main;
            if (cam == null) return;

            Vector3 facingDirection = transform.position - cam.transform.position;
            if (facingDirection.sqrMagnitude < 0.0001f) return;
            transform.rotation = Quaternion.LookRotation(facingDirection.normalized, cam.transform.up);
        }

        private void PlacePanelInWorldOnce()
        {
            if (_panelRoot == null || _panelPlacedInWorld) return;
            if (_statusPanel == null)
                _statusPanel = FindAnyObjectByType<ARArmDetection.DetectionModeButton>();

            if (_statusPanel != null)
            {
                Transform statusTransform = _statusPanel.transform;
                Vector3 worldPosition = statusTransform.TransformPoint(
                    Vector3.left * _statusPanelLeftOffsetMeters);
                Quaternion worldRotation = statusTransform.rotation;
                transform.SetParent(null, true);
                transform.SetPositionAndRotation(worldPosition, worldRotation);
                _panelPlacedInWorld = true;
                return;
            }

            var cam = Camera.main;
            if (cam == null) return;
            Transform ct = cam.transform;
            transform.SetParent(null, true);
            transform.position = ct.position + ct.forward * 1.4f + ct.up * 0.05f;
            transform.rotation = Quaternion.LookRotation(
                (transform.position - ct.position).normalized, ct.up);
            _panelPlacedInWorld = true;
        }

        private bool TryGetPointerPlanePoint(PointerEventData eventData, out Vector3 point)
        {
            point = default;

            if (eventData is TrackedDeviceEventData tracked &&
                tracked.rayPoints != null && tracked.rayPoints.Count >= 2)
            {
                for (int i = 1; i < tracked.rayPoints.Count; i++)
                {
                    Vector3 origin = tracked.rayPoints[i - 1];
                    Vector3 segment = tracked.rayPoints[i] - origin;
                    float length = segment.magnitude;
                    if (length < 0.0001f) continue;

                    var ray = new Ray(origin, segment / length);
                    if (_panelDragPlane.Raycast(ray, out float enter) && enter <= length + 0.01f)
                    {
                        point = ray.GetPoint(enter);
                        return true;
                    }
                }
            }

            Camera eventCamera = eventData != null ? eventData.pressEventCamera : null;
            if (eventCamera == null) eventCamera = Camera.main;
            if (eventCamera == null || eventData == null) return false;

            Ray screenRay = eventCamera.ScreenPointToRay(eventData.position);
            if (!_panelDragPlane.Raycast(screenRay, out float screenEnter)) return false;
            point = screenRay.GetPoint(screenEnter);
            return true;
        }

        private void BuildUI()
        {
            _panelRoot = new GameObject("FacilitatorCanvas");
            _panelRoot.transform.SetParent(transform, false);

            var canvas = _panelRoot.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            _panelRoot.AddComponent<CanvasScaler>();
            _panelRoot.AddComponent<GraphicRaycaster>();
            _panelRoot.AddComponent<TrackedDeviceGraphicRaycaster>();

            var canvasRt = (RectTransform)_panelRoot.transform;
            canvasRt.sizeDelta = new Vector2(560f, 340f);
            canvasRt.localScale = Vector3.one * 0.0015f;

            var background = CreateImage("Background", canvasRt, Vector2.zero, Vector2.one,
                new Color(0.055f, 0.065f, 0.075f, 0.96f));
            background.raycastTarget = true;
            var dragHandle = background.gameObject.AddComponent<FacilitatorPanelDragHandle>();
            dragHandle.Initialize(this);

            // Covers the panel body but leaves the bottom action row free so a
            // pinch on Next step cannot accidentally start moving the panel.
            var handGrabCollider = _panelRoot.AddComponent<BoxCollider>();
            handGrabCollider.center = new Vector3(0f, 35f, 0f);
            handGrabCollider.size = new Vector3(560f, 270f, 12f);

            var handCursor = CreateImage("HandPointerCursor", canvasRt,
                new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                new Color(0.2f, 0.9f, 0.82f, 0.95f));
            var handCursorRect = (RectTransform)handCursor.transform;
            handCursorRect.sizeDelta = new Vector2(22f, 22f);
            handCursor.gameObject.SetActive(false);

            _procedureText = CreateText("Procedure", canvasRt, new Vector2(0.04f, 0.86f), new Vector2(0.72f, 0.97f),
                22, FontStyle.Bold, TextAnchor.MiddleLeft, new Color(0.95f, 0.97f, 1f));
            _progressText = CreateText("Progress", canvasRt, new Vector2(0.72f, 0.86f), new Vector2(0.96f, 0.97f),
                15, FontStyle.Bold, TextAnchor.MiddleRight, new Color(0.35f, 0.85f, 0.78f));

            CreateImage("ProgressTrack", canvasRt, new Vector2(0.04f, 0.835f), new Vector2(0.96f, 0.85f),
                new Color(0.22f, 0.24f, 0.27f, 1f));
            _progressFill = CreateImage("ProgressFill", canvasRt, new Vector2(0.04f, 0.835f), new Vector2(0.96f, 0.85f),
                new Color(0.12f, 0.72f, 0.62f, 1f));
            _progressFill.type = Image.Type.Filled;
            _progressFill.fillMethod = Image.FillMethod.Horizontal;
            _progressFill.fillOrigin = 0;

            _titleText = CreateText("StepTitle", canvasRt, new Vector2(0.05f, 0.67f), new Vector2(0.95f, 0.82f),
                25, FontStyle.Bold, TextAnchor.MiddleLeft, Color.white);
            _instructionText = CreateText("Instruction", canvasRt, new Vector2(0.05f, 0.26f), new Vector2(0.95f, 0.68f),
                19, FontStyle.Normal, TextAnchor.UpperLeft, new Color(0.92f, 0.94f, 0.96f));
            var previousButton = CreateButton("Previous step", canvasRt,
                new Vector2(0.04f, 0.04f), new Vector2(0.42f, 0.17f),
                null, out _);
            _previousButtonObject = previousButton.gameObject;
            _previousButtonRect = (RectTransform)previousButton.transform;

            var nextButton = CreateButton("Next step", canvasRt,
                new Vector2(0.58f, 0.04f), new Vector2(0.96f, 0.17f),
                null, out _nextButtonText);
            _nextButtonRect = (RectTransform)nextButton.transform;
            _dwellCircleSprite = CreateCircleSprite(out _dwellCircleTexture);
            var dwellTrack = CreateImage("HandHoldProgressTrack", _nextButtonRect,
                new Vector2(0.86f, 0.5f), new Vector2(0.86f, 0.5f),
                new Color(0.32f, 0.36f, 0.39f, 0.9f));
            var dwellTrackRect = (RectTransform)dwellTrack.transform;
            dwellTrackRect.sizeDelta = new Vector2(30f, 30f);
            dwellTrack.sprite = _dwellCircleSprite;
            _gazeDwellFill = CreateImage("GazeDwellProgress", _nextButtonRect,
                new Vector2(0.86f, 0.5f), new Vector2(0.86f, 0.5f),
                new Color(0.12f, 0.85f, 0.70f, 1f));
            var dwellFillRect = (RectTransform)_gazeDwellFill.transform;
            dwellFillRect.sizeDelta = new Vector2(30f, 30f);
            _gazeDwellFill.sprite = _dwellCircleSprite;
            _gazeDwellFill.type = Image.Type.Filled;
            _gazeDwellFill.fillMethod = Image.FillMethod.Radial360;
            _gazeDwellFill.fillOrigin = (int)Image.Origin360.Top;
            _gazeDwellFill.fillClockwise = true;
            _gazeDwellFill.fillAmount = 0f;

            var previousTrack = CreateImage("PreviousHoldProgressTrack", _previousButtonRect,
                new Vector2(0.86f, 0.5f), new Vector2(0.86f, 0.5f),
                new Color(0.32f, 0.36f, 0.39f, 0.9f));
            var previousTrackRect = (RectTransform)previousTrack.transform;
            previousTrackRect.sizeDelta = new Vector2(30f, 30f);
            previousTrack.sprite = _dwellCircleSprite;
            _previousHoldFill = CreateImage("PreviousHoldProgress", _previousButtonRect,
                new Vector2(0.86f, 0.5f), new Vector2(0.86f, 0.5f),
                new Color(0.12f, 0.85f, 0.70f, 1f));
            var previousFillRect = (RectTransform)_previousHoldFill.transform;
            previousFillRect.sizeDelta = new Vector2(30f, 30f);
            _previousHoldFill.sprite = _dwellCircleSprite;
            _previousHoldFill.type = Image.Type.Filled;
            _previousHoldFill.fillMethod = Image.FillMethod.Radial360;
            _previousHoldFill.fillOrigin = (int)Image.Origin360.Top;
            _previousHoldFill.fillClockwise = true;
            _previousHoldFill.fillAmount = 0f;

            var handDrag = gameObject.AddComponent<FacilitatorHandPanelDrag>();
            handDrag.Initialize(transform, handGrabCollider, handCursorRect, handCursor,
                _nextButtonRect, _gazeDwellFill, NextStep,
                _previousButtonRect, _previousHoldFill, PreviousStep);

            ApplyDepthIndependentPanelMaterial();
        }

        private void ApplyDepthIndependentPanelMaterial()
        {
            Shader shader = Resources.Load<Shader>("Facilitator/FacilitatorUIAlwaysOnTop")
                         ?? Shader.Find("Facilitator/UI Depth Aware");
            if (shader == null)
            {
                Debug.LogWarning("[Facilitator] Always-on-top UI shader was not found; hand occluders may hide the panel.");
                return;
            }

            _panelAlwaysOnTopMaterial = new Material(shader)
            {
                name = "Facilitator UI Depth Aware (Runtime)"
            };

            var graphics = _panelRoot.GetComponentsInChildren<Graphic>(true);
            for (int i = 0; i < graphics.Length; i++)
                graphics[i].material = _panelAlwaysOnTopMaterial;
        }

        private static Image CreateImage(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Image));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            SetAnchors(rt, anchorMin, anchorMax, Vector2.zero, Vector2.zero);
            var image = go.GetComponent<Image>();
            image.color = color;
            image.raycastTarget = false;
            return image;
        }

        private static Sprite CreateCircleSprite(out Texture2D texture)
        {
            const int size = 64;
            texture = new Texture2D(size, size, TextureFormat.RGBA32, false)
            {
                name = "Facilitator Hold Circle",
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };

            var pixels = new Color32[size * size];
            float radius = size * 0.48f;
            Vector2 center = new Vector2((size - 1) * 0.5f, (size - 1) * 0.5f);
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float alpha = Mathf.Clamp01(radius + 0.75f - Vector2.Distance(new Vector2(x, y), center));
                    pixels[y * size + x] = new Color32(255, 255, 255, (byte)(alpha * 255f));
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply(false, true);
            return Sprite.Create(texture, new Rect(0f, 0f, size, size), new Vector2(0.5f, 0.5f), size);
        }

        private static Text CreateText(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
            int fontSize, FontStyle style, TextAnchor alignment, Color color)
        {
            var go = new GameObject(name, typeof(RectTransform), typeof(Text));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            SetAnchors(rt, anchorMin, anchorMax, Vector2.zero, Vector2.zero);
            var text = go.GetComponent<Text>();
            text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            text.fontSize = fontSize;
            text.fontStyle = style;
            text.alignment = alignment;
            text.color = color;
            text.horizontalOverflow = HorizontalWrapMode.Wrap;
            text.verticalOverflow = VerticalWrapMode.Truncate;
            text.supportRichText = true;
            text.raycastTarget = false;
            return text;
        }

        private static Button CreateButton(string label, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
            UnityEngine.Events.UnityAction action, out Text labelText)
        {
            var go = new GameObject(label + "Button", typeof(RectTransform), typeof(Image), typeof(Button));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            SetAnchors(rt, anchorMin, anchorMax, Vector2.zero, Vector2.zero);

            var image = go.GetComponent<Image>();
            image.color = new Color(0.16f, 0.19f, 0.22f, 1f);
            var button = go.GetComponent<Button>();
            button.targetGraphic = image;
            if (action != null) button.onClick.AddListener(action);
            var colors = button.colors;
            colors.highlightedColor = new Color(0.2f, 0.48f, 0.44f, 1f);
            colors.pressedColor = new Color(0.1f, 0.65f, 0.55f, 1f);
            colors.selectedColor = colors.highlightedColor;
            button.colors = colors;

            labelText = CreateText("Label", go.transform, Vector2.zero, Vector2.one,
                17, FontStyle.Bold, TextAnchor.MiddleCenter, Color.white);
            labelText.text = label;
            return button;
        }

        private static void SetAnchors(RectTransform rt, Vector2 min, Vector2 max,
            Vector2 offsetMin, Vector2 offsetMax)
        {
            rt.anchorMin = min;
            rt.anchorMax = max;
            rt.offsetMin = offsetMin;
            rt.offsetMax = offsetMax;
        }

        private static void EnsureEventSystem()
        {
            if (EventSystem.current != null) return;
            var go = new GameObject("XR EventSystem", typeof(EventSystem), typeof(XRUIInputModule));
            DontDestroyOnLoad(go);
        }
    }
}
