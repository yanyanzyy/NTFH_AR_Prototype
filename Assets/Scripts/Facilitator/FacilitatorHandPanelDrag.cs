using System;
using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection.Facilitator
{
    /// <summary>
    /// Uses Meta hand-pointer poses directly so either tracked hand can move the
    /// facilitator panel without controllers or Interaction SDK grab components.
    /// </summary>
    public class FacilitatorHandPanelDrag : MonoBehaviour
    {
        [SerializeField] private float _maximumGrabDistance = 4f;
        [SerializeField] private float _directGrabDistance = 0.12f;
        [SerializeField] private float _handSearchInterval = 3f;
        [SerializeField, Range(0.5f, 3f)] private float _holdDurationSeconds = 1f;

        private readonly List<OVRHand> _hands = new();
        private readonly Dictionary<OVRHand, Transform> _indexTips = new();
        private Transform _panel;
        private BoxCollider _grabCollider;
        private RectTransform _cursor;
        private UnityEngine.UI.Image _cursorImage;
        private OVRHand _activeHand;
        private float _grabDistance;
        private Vector3 _grabOffset;
        private bool _directGrab;
        private float _nextHandSearchTime;
        private float _nextStatusLogTime;
        private LineRenderer _pointerLine;
        private Material _pointerLineMaterial;
        private RectTransform _nextButton;
        private UnityEngine.UI.Image _nextHoldFill;
        private Action _onNextStep;
        private float _nextHoldTimer;
        private bool _nextRequiresRelease;
        private RectTransform _previousButton;
        private UnityEngine.UI.Image _previousHoldFill;
        private Action _onPreviousStep;
        private float _previousHoldTimer;
        private bool _previousRequiresRelease;

        public void Initialize(Transform panel, BoxCollider grabCollider,
            RectTransform cursor, UnityEngine.UI.Image cursorImage,
            RectTransform nextButton, UnityEngine.UI.Image nextHoldFill, Action onNextStep,
            RectTransform previousButton, UnityEngine.UI.Image previousHoldFill, Action onPreviousStep)
        {
            _panel = panel;
            _grabCollider = grabCollider;
            _cursor = cursor;
            _cursorImage = cursorImage;
            _nextButton = nextButton;
            _nextHoldFill = nextHoldFill;
            _onNextStep = onNextStep;
            _previousButton = previousButton;
            _previousHoldFill = previousHoldFill;
            _onPreviousStep = onPreviousStep;
            BuildPointerLine();
            FindTrackedHands();
        }

        private void OnDestroy()
        {
            if (_pointerLineMaterial != null) Destroy(_pointerLineMaterial);
        }

        private void Update()
        {
            if (_panel == null || _grabCollider == null) return;

            if (Time.unscaledTime >= _nextHandSearchTime)
                FindTrackedHands();

            bool cursorShown = false;
            int trackedCount = 0;
            int pointerCount = 0;
            int tipCount = 0;
            int hoverCount = 0;
            bool pointerLineShown = false;
            bool nextHoldActive = false;
            bool previousHoldActive = false;
            bool anyPinching = false;
            if (_activeHand != null)
            {
                bool stillPinching = IsTracked(_activeHand) && IsIndexPinching(_activeHand);
                if (!stillPinching)
                {
                    _activeHand = null;
                    _directGrab = false;
                }
                else if (_directGrab && TryGetIndexTip(_activeHand, out Vector3 tipPosition))
                {
                    _panel.position = tipPosition + _grabOffset;
                    FacePanelTowardViewer();
                    ShowCursor(_grabCollider.ClosestPoint(tipPosition), true);
                    cursorShown = true;
                }
                else if (!_directGrab && TryGetPointerRay(_activeHand, out Ray activeRay))
                {
                    Vector3 target = activeRay.GetPoint(_grabDistance);
                    _panel.position = target + _grabOffset;
                    FacePanelTowardViewer();
                    ShowCursor(target, true);
                    cursorShown = true;
                }
            }

            for (int i = 0; i < _hands.Count; i++)
            {
                OVRHand hand = _hands[i];
                if (hand == null) continue;

                bool pinching = IsTracked(hand) && IsIndexPinching(hand);
                anyPinching |= pinching;
                if (IsTracked(hand)) trackedCount++;

                bool hasDirectHover = TryGetDirectHover(hand, out Vector3 directPoint);
                Ray ray = default;
                bool hasRayHover = false;
                Vector3 rayPoint = default;
                float rayDistance = 0f;
                if (TryGetPointerRay(hand, out ray))
                {
                    pointerCount++;
                    hasRayHover = TryGetRayHover(ray, out rayPoint, out rayDistance);
                    if (!pointerLineShown)
                    {
                        ShowPointerLine(ray, hasRayHover ? rayPoint : ray.GetPoint(2f), hasRayHover);
                        pointerLineShown = true;
                    }
                }

                bool nextDirectHover = TryGetRectDirectHover(hand, _nextButton, out Vector3 nextDirectPoint);
                bool nextRayHover = TryGetRectRayHover(ray, _nextButton, out Vector3 nextRayPoint);
                if (_activeHand == null && pinching && (nextDirectHover || nextRayHover))
                {
                    nextHoldActive = true;
                    Vector3 nextPoint = nextDirectHover ? nextDirectPoint : nextRayPoint;
                    ShowCursor(nextPoint, true);
                    cursorShown = true;
                    if (nextRayHover)
                    {
                        ShowPointerLine(ray, nextRayPoint, true);
                        pointerLineShown = true;
                    }
                }

                bool previousDirectHover = TryGetRectDirectHover(hand, _previousButton, out Vector3 previousDirectPoint);
                bool previousRayHover = TryGetRectRayHover(ray, _previousButton, out Vector3 previousRayPoint);
                if (_activeHand == null && pinching && (previousDirectHover || previousRayHover))
                {
                    previousHoldActive = true;
                    Vector3 previousPoint = previousDirectHover ? previousDirectPoint : previousRayPoint;
                    ShowCursor(previousPoint, true);
                    cursorShown = true;
                    if (previousRayHover)
                    {
                        ShowPointerLine(ray, previousRayPoint, true);
                        pointerLineShown = true;
                    }
                }
                if (TryGetIndexTip(hand, out _)) tipCount++;
                if (hasDirectHover || hasRayHover) hoverCount++;

                if (_activeHand == null && !cursorShown)
                {
                    if (hasDirectHover)
                    {
                        ShowCursor(directPoint, pinching);
                        cursorShown = true;
                    }
                    else if (hasRayHover)
                    {
                        ShowCursor(rayPoint, pinching);
                        cursorShown = true;
                    }
                }

                if (_activeHand == null && pinching && hasDirectHover &&
                    TryGetIndexTip(hand, out Vector3 grabTip))
                {
                    _activeHand = hand;
                    _directGrab = true;
                    _grabOffset = _panel.position - grabTip;
                }
                else if (_activeHand == null && pinching && hasRayHover)
                {
                    _activeHand = hand;
                    _directGrab = false;
                    _grabDistance = rayDistance;
                    _grabOffset = _panel.position - rayPoint;
                }
            }

            UpdateNextStepHold(nextHoldActive, anyPinching);
            UpdatePreviousStepHold(previousHoldActive, anyPinching);

            if (!cursorShown && _cursor != null)
                _cursor.gameObject.SetActive(false);
            if (!pointerLineShown && _pointerLine != null)
                _pointerLine.enabled = false;

            if (Time.unscaledTime >= _nextStatusLogTime)
            {
                _nextStatusLogTime = Time.unscaledTime + 30f;
                Debug.Log($"[FacilitatorHandDrag] found={_hands.Count} tracked={trackedCount} " +
                          $"pointers={pointerCount} fingertips={tipCount} hovering={hoverCount} " +
                          $"grabbed={(_activeHand != null)}");
            }
        }

        private void FacePanelTowardViewer()
        {
            var cam = Camera.main;
            if (cam == null || _panel == null) return;

            Vector3 facingDirection = _panel.position - cam.transform.position;
            if (facingDirection.sqrMagnitude < 0.0001f) return;
            _panel.rotation = Quaternion.LookRotation(facingDirection.normalized, cam.transform.up);
        }

        private void UpdatePreviousStepHold(bool active, bool anyPinching)
        {
            if (!anyPinching) _previousRequiresRelease = false;

            if (!active || _previousRequiresRelease)
            {
                _previousHoldTimer = 0f;
                if (_previousHoldFill != null) _previousHoldFill.fillAmount = 0f;
                return;
            }

            _previousHoldTimer += Time.unscaledDeltaTime;
            if (_previousHoldFill != null)
                _previousHoldFill.fillAmount = Mathf.Clamp01(_previousHoldTimer / _holdDurationSeconds);

            if (_previousHoldTimer < _holdDurationSeconds) return;

            _previousHoldTimer = 0f;
            _previousRequiresRelease = true;
            if (_previousHoldFill != null) _previousHoldFill.fillAmount = 0f;
            _onPreviousStep?.Invoke();
        }

        private void UpdateNextStepHold(bool active, bool anyPinching)
        {
            if (!anyPinching) _nextRequiresRelease = false;

            if (!active || _nextRequiresRelease)
            {
                _nextHoldTimer = 0f;
                if (_nextHoldFill != null) _nextHoldFill.fillAmount = 0f;
                return;
            }

            _nextHoldTimer += Time.unscaledDeltaTime;
            if (_nextHoldFill != null)
                _nextHoldFill.fillAmount = Mathf.Clamp01(_nextHoldTimer / _holdDurationSeconds);

            if (_nextHoldTimer < _holdDurationSeconds) return;

            _nextHoldTimer = 0f;
            _nextRequiresRelease = true;
            if (_nextHoldFill != null) _nextHoldFill.fillAmount = 0f;
            _onNextStep?.Invoke();
        }

        private void FindTrackedHands()
        {
            _nextHandSearchTime = Time.unscaledTime + _handSearchInterval;
            var found = FindObjectsByType<OVRHand>(FindObjectsInactive.Include);

            _hands.Clear();
            _indexTips.Clear();
            for (int i = 0; i < found.Length; i++)
            {
                if (found[i] == null) continue;
                _hands.Add(found[i]);

                OVRSkeleton skeleton = found[i].GetComponent<OVRSkeleton>();
                if (skeleton == null) skeleton = found[i].GetComponentInParent<OVRSkeleton>();
                Transform tip = FindBone(skeleton, OVRSkeleton.BoneId.Hand_IndexTip);
                if (tip != null) _indexTips[found[i]] = tip;
            }
        }

        private bool TryGetDirectHover(OVRHand hand, out Vector3 closestPoint)
        {
            closestPoint = default;
            if (!TryGetIndexTip(hand, out Vector3 tipPosition)) return false;
            closestPoint = _grabCollider.ClosestPoint(tipPosition);
            return Vector3.Distance(tipPosition, closestPoint) <= _directGrabDistance;
        }

        private bool TryGetRayHover(Ray ray, out Vector3 point, out float distance)
        {
            point = default;
            distance = 0f;

            Vector3 planePoint = _grabCollider.transform.TransformPoint(_grabCollider.center);
            Vector3 planeNormal = _grabCollider.transform.forward;
            var plane = new Plane(planeNormal, planePoint);
            if (!plane.Raycast(ray, out distance) || distance < 0f || distance > _maximumGrabDistance)
                return false;

            point = ray.GetPoint(distance);
            Vector3 local = _grabCollider.transform.InverseTransformPoint(point);
            Vector3 min = _grabCollider.center - _grabCollider.size * 0.5f;
            Vector3 max = _grabCollider.center + _grabCollider.size * 0.5f;
            float padX = _grabCollider.size.x * 0.1f;
            float padY = _grabCollider.size.y * 0.1f;
            return local.x >= min.x - padX && local.x <= max.x + padX &&
                   local.y >= min.y - padY && local.y <= max.y + padY;
        }

        private bool TryGetRectRayHover(Ray ray, RectTransform rect, out Vector3 point)
        {
            point = default;
            if (rect == null || !rect.gameObject.activeInHierarchy || ray.direction.sqrMagnitude < 0.001f)
                return false;

            var plane = new Plane(rect.forward, rect.position);
            if (!plane.Raycast(ray, out float distance) || distance < 0f || distance > _maximumGrabDistance)
                return false;

            point = ray.GetPoint(distance);
            Vector3 local = rect.InverseTransformPoint(point);
            return rect.rect.Contains(new Vector2(local.x, local.y));
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

        private void BuildPointerLine()
        {
            var go = new GameObject("FacilitatorHandPointerLine");
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

        private void ShowCursor(Vector3 worldPoint, bool grabbing)
        {
            if (_cursor == null) return;
            Vector3 local = _grabCollider.transform.InverseTransformPoint(worldPoint);
            _cursor.localPosition = new Vector3(local.x, local.y, -8f);
            _cursor.gameObject.SetActive(true);
            if (_cursorImage != null)
                _cursorImage.color = grabbing
                    ? new Color(0.15f, 1f, 0.45f, 1f)
                    : new Color(0.2f, 0.9f, 0.82f, 0.95f);
        }

        private static Transform FindBone(OVRSkeleton skeleton, OVRSkeleton.BoneId id)
        {
            if (skeleton == null || !skeleton.IsInitialized || skeleton.Bones == null) return null;
            for (int i = 0; i < skeleton.Bones.Count; i++)
            {
                OVRBone bone = skeleton.Bones[i];
                if (bone != null && bone.Id == id) return bone.Transform;
            }
            return null;
        }

        private static bool IsTracked(OVRHand hand)
        {
            return hand != null && hand.IsTracked && hand.IsDataValid;
        }

        private static bool IsIndexPinching(OVRHand hand)
        {
            return hand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        }

        private static bool TryGetPointerRay(OVRHand hand, out Ray ray)
        {
            ray = default;
            if (!IsTracked(hand) || !hand.IsPointerPoseValid || hand.PointerPose == null) return false;
            Transform pointer = hand.PointerPose;
            ray = new Ray(pointer.position, pointer.rotation * Vector3.forward);
            return true;
        }
    }
}
