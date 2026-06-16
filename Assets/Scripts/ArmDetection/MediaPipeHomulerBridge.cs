using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using UnityEngine.Events;

namespace ARArmDetection
{
    /// <summary>
    /// Reflection bridge for homuler/MediaPipeUnityPlugin's HandLandmarkerRunner.
    /// It avoids a hard compile-time dependency, so this project still compiles
    /// before the MediaPipe package is imported.
    /// </summary>
    public class MediaPipeHomulerBridge : MonoBehaviour
    {
        [Tooltip("Drag the homuler HandLandmarkerRunner component here after importing MediaPipeUnityPlugin.")]
        [SerializeField] private MonoBehaviour _handLandmarkerRunner;
        [SerializeField] private MediaPipeHandArmDetector _target;
        [Tooltip("Flip MediaPipe normalized X coordinates if the camera feed is mirrored.")]
        [SerializeField] private bool _flipX;
        [Tooltip("Flip MediaPipe normalized Y coordinates if the source image is vertically inverted.")]
        [SerializeField] private bool _flipY;
        [Tooltip("If handedness cannot be read from MediaPipe, use this arm side for the generated COCO keypoints.")]
        [SerializeField] private Side _fallbackSide = Side.Right;

        private Delegate _callback;
        private FieldInfo _callbackField;
        private EventInfo _callbackEvent;

        private void Reset()
        {
            _target = GetComponent<MediaPipeHandArmDetector>()
                   ?? GetComponentInParent<MediaPipeHandArmDetector>()
                   ?? FindFirstObjectByType<MediaPipeHandArmDetector>();
        }

        private void OnEnable()
        {
            TrySubscribe();
        }

        private void OnDisable()
        {
            TryUnsubscribe();
        }

        public void TrySubscribe()
        {
            if (_handLandmarkerRunner == null || _target == null || _callback != null) return;

            var runnerType = _handLandmarkerRunner.GetType();
            _callbackField = runnerType.GetField("ProcessHandLandmark", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (_callbackField != null && TryCreateCallback(_callbackField.FieldType, out _callback))
            {
                var existing = _callbackField.GetValue(_handLandmarkerRunner) as Delegate;
                _callbackField.SetValue(_handLandmarkerRunner, Delegate.Combine(existing, _callback));
                Debug.Log($"[MediaPipeHomulerBridge] Subscribed to {runnerType.Name}.ProcessHandLandmark");
                return;
            }

            _callbackEvent = runnerType.GetEvent("OnLandmarksDetected", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                          ?? runnerType.GetEvent("OnHandLandmarksDetected", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (_callbackEvent != null && TryCreateCallback(_callbackEvent.EventHandlerType, out _callback))
            {
                _callbackEvent.AddEventHandler(_handLandmarkerRunner, _callback);
                Debug.Log($"[MediaPipeHomulerBridge] Subscribed to {runnerType.Name}.{_callbackEvent.Name}");
                return;
            }

            Debug.LogWarning("[MediaPipeHomulerBridge] Could not find a compatible HandLandmarkerRunner output. " +
                             "Expected a ProcessHandLandmark field or OnLandmarksDetected event.");
        }

        private void TryUnsubscribe()
        {
            if (_handLandmarkerRunner == null || _callback == null) return;

            if (_callbackField != null)
            {
                var existing = _callbackField.GetValue(_handLandmarkerRunner) as Delegate;
                _callbackField.SetValue(_handLandmarkerRunner, Delegate.Remove(existing, _callback));
            }
            else if (_callbackEvent != null)
            {
                _callbackEvent.RemoveEventHandler(_handLandmarkerRunner, _callback);
            }

            _callback = null;
            _callbackField = null;
            _callbackEvent = null;
        }

        private bool TryCreateCallback(Type delegateType, out Delegate callback)
        {
            callback = null;
            if (delegateType == null || !delegateType.IsGenericType) return false;

            var args = delegateType.GetGenericArguments();
            if (args.Length != 1) return false;

            var method = typeof(MediaPipeHomulerBridge)
                .GetMethod(nameof(OnHandResultGeneric), BindingFlags.Instance | BindingFlags.NonPublic)
                ?.MakeGenericMethod(args[0]);
            if (method == null) return false;

            callback = Delegate.CreateDelegate(delegateType, this, method);
            return callback != null;
        }

        private void OnHandResultGeneric<T>(T result)
        {
            ConsumeHandResult(result);
        }

        private void ConsumeHandResult(object result)
        {
            if (_target == null || result == null)
            {
                _target?.ClearLandmarks();
                return;
            }

            if (!TryExtractFirstHand(result, out var landmarks, out var side, out var confidence))
            {
                return;
            }

            _target.SetNormalizedLandmarks(landmarks, side, confidence);
        }

        private bool TryExtractFirstHand(object result, out List<Vector2> landmarks, out Side side, out float confidence)
        {
            landmarks = null;
            side = _fallbackSide;
            confidence = 1f;

            object handLandmarks = GetMember(result, "handLandmarks")
                                ?? GetMember(result, "landmarks");
            if (handLandmarks is not IEnumerable hands) return false;

            object firstHand = null;
            foreach (var hand in hands)
            {
                firstHand = hand;
                break;
            }
            if (firstHand == null) return false;

            object landmarkList = GetMember(firstHand, "landmarks")
                               ?? GetMember(firstHand, "Landmark")
                               ?? firstHand;
            if (landmarkList is not IEnumerable points) return false;

            landmarks = new List<Vector2>(21);
            foreach (var point in points)
            {
                if (!TryReadFloat(point, "x", out float x) && !TryReadFloat(point, "X", out x)) return false;
                if (!TryReadFloat(point, "y", out float y) && !TryReadFloat(point, "Y", out y)) return false;
                if (_flipX) x = 1f - x;
                if (_flipY) y = 1f - y;
                landmarks.Add(new Vector2(Mathf.Clamp01(x), Mathf.Clamp01(y)));
                if (landmarks.Count >= 21) break;
            }

            if (landmarks.Count < 21) return false;

            TryReadHandedness(result, out side, out confidence);
            return true;
        }

        private void TryReadHandedness(object result, out Side side, out float confidence)
        {
            side = _fallbackSide;
            confidence = 1f;

            object handedness = GetMember(result, "handedness");
            if (handedness is not IEnumerable handednessList) return;

            object firstList = null;
            foreach (var item in handednessList)
            {
                firstList = item;
                break;
            }
            if (firstList == null) return;

            object categories = GetMember(firstList, "categories")
                             ?? GetMember(firstList, "classification")
                             ?? firstList;
            if (categories is not IEnumerable categoryList) return;

            foreach (var category in categoryList)
            {
                string label = (GetMember(category, "categoryName")
                             ?? GetMember(category, "label")
                             ?? GetMember(category, "Label"))?.ToString();
                if (TryReadFloat(category, "score", out float score)
                 || TryReadFloat(category, "Score", out score))
                    confidence = Mathf.Clamp01(score);

                if (!string.IsNullOrEmpty(label))
                    side = label.IndexOf("left", StringComparison.OrdinalIgnoreCase) >= 0 ? Side.Left : Side.Right;
                return;
            }
        }

        private static object GetMember(object obj, string name)
        {
            if (obj == null) return null;
            var type = obj.GetType();
            var field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (field != null) return field.GetValue(obj);
            var prop = type.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            return prop != null ? prop.GetValue(obj) : null;
        }

        private static bool TryReadFloat(object obj, string name, out float value)
        {
            value = 0f;
            var member = GetMember(obj, name);
            if (member == null) return false;
            try
            {
                value = Convert.ToSingle(member);
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}
