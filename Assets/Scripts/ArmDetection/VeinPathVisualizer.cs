using System.Collections.Generic;
using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// Draws each VeinMap vein path as a LineRenderer IN THE HEADSET, so the authored
    /// waypoints can be verified against the physical mannequin (VeinMap's own gizmos
    /// only render in the editor Scene view — invisible in a Quest build, and passthrough
    /// can't run in the editor).
    ///
    /// Lines appear once the arm is locked, in each vein's debug colour. A vein that
    /// resolves to a single point (prefab path missing AND segment end left at 0,0) is
    /// drawn as a short tick so it's still findable. Point counts are logged once per
    /// vein so a name mismatch shows up in logcat. Disable or remove for final builds.
    /// </summary>
    public class VeinPathVisualizer : MonoBehaviour
    {
        [SerializeField] private VeinMap _veinMap;

        [Tooltip("Line width in metres (~4 mm reads well at arm's length).")]
        [SerializeField] private float _lineWidthMeters = 0.004f;

        [Tooltip("Optional — when set, the vein lines only draw while this overlay's mesh is shown " +
                 "(ArmOverlay.IsModelRevealed), so they stay part of the hidden 'answer key' during " +
                 "poking and only flash up with it. Leave empty to always draw (debug behaviour).")]
        [SerializeField] private ArmOverlay _revealGate;

        private readonly List<LineRenderer> _lines = new();
        private readonly List<Vector3> _points = new();
        private readonly Dictionary<int, int> _loggedCounts = new();

        private void Update()
        {
            if (_veinMap == null) return;

            // Keep the vein lines hidden while the answer-key overlay is hidden.
            if (_revealGate != null && !_revealGate.IsModelRevealed)
            {
                for (int i = 0; i < _lines.Count; i++)
                    if (_lines[i] != null) _lines[i].enabled = false;
                return;
            }

            var veins = _veinMap.Veins;
            while (_lines.Count < veins.Count) _lines.Add(CreateLine(_lines.Count));

            for (int i = 0; i < _lines.Count; i++)
            {
                bool used = i < veins.Count && _veinMap.HasArm;
                int count = used ? _veinMap.GetVeinPolyline(veins[i], _points) : 0;

                if (used && count != _loggedCounts.GetValueOrDefault(i, -1))
                {
                    _loggedCounts[i] = count;
                    Debug.Log($"[VeinPathVisualizer] '{veins[i].name}': {count} point(s) " +
                              $"(pathObjectName='{veins[i].pathObjectName}'). A prefab path with N " +
                              "waypoints should report N; 1-2 means it fell back to the cylinder segment.");
                }

                if (!used || count == 0)
                {
                    _lines[i].enabled = false;
                    continue;
                }

                // Single point: draw a 3 cm tick so the vein is still visible/locatable.
                if (count == 1) _points.Add(_points[0] + Vector3.up * 0.03f);

                var line = _lines[i];
                line.enabled = true;
                line.positionCount = _points.Count;
                for (int p = 0; p < _points.Count; p++) line.SetPosition(p, _points[p]);

                Color c = veins[i].debugColor;
                line.startColor = c;
                line.endColor = c;
            }
        }

        private LineRenderer CreateLine(int index)
        {
            var go = new GameObject($"VeinPath_{index}");
            go.transform.SetParent(transform, false);
            var line = go.AddComponent<LineRenderer>();
            line.useWorldSpace = true;
            line.startWidth = _lineWidthMeters;
            line.endWidth = _lineWidthMeters;
            line.material = new Material(Shader.Find("Sprites/Default"));
            line.enabled = false;
            return line;
        }
    }
}
