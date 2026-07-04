using UnityEngine;

namespace ARArmDetection
{
    /// <summary>
    /// One-Euro filter for a 6-DoF pose: low jitter when the target is still, low lag
    /// when it moves (cutoff frequency rises with speed). Position is filtered per
    /// axis; rotation is filtered on the quaternion manifold using the angular speed
    /// to adapt the cutoff.
    ///
    /// minCutoff: lower = smoother when still (more lag). beta: higher = snappier
    /// during motion. Standard starting points: minCutoff 1.0, beta 0.02–0.1.
    /// </summary>
    public class OneEuroPoseFilter
    {
        private readonly float _minCutoff;
        private readonly float _beta;
        private readonly float _dCutoff;

        private bool _hasPrev;
        private Vector3 _prevPos;
        private Vector3 _prevPosRate;   // filtered velocity (m/s)
        private Quaternion _prevRot;
        private float _prevAngRate;     // filtered angular speed (deg/s)

        public OneEuroPoseFilter(float minCutoff = 1f, float beta = 0.05f, float dCutoff = 1f)
        {
            _minCutoff = Mathf.Max(0.01f, minCutoff);
            _beta = Mathf.Max(0f, beta);
            _dCutoff = Mathf.Max(0.01f, dCutoff);
        }

        public void Reset()
        {
            _hasPrev = false;
        }

        private static float Alpha(float cutoff, float dt)
        {
            float tau = 1f / (2f * Mathf.PI * cutoff);
            return 1f / (1f + tau / Mathf.Max(1e-5f, dt));
        }

        public Pose Filter(Pose raw, float dt)
        {
            if (!_hasPrev || dt <= 0f)
            {
                _hasPrev = true;
                _prevPos = raw.position;
                _prevRot = raw.rotation;
                _prevPosRate = Vector3.zero;
                _prevAngRate = 0f;
                return raw;
            }

            // Position.
            Vector3 rate = (raw.position - _prevPos) / dt;
            float aRate = Alpha(_dCutoff, dt);
            _prevPosRate = Vector3.Lerp(_prevPosRate, rate, aRate);
            float cutoffPos = _minCutoff + _beta * _prevPosRate.magnitude;
            Vector3 pos = Vector3.Lerp(_prevPos, raw.position, Alpha(cutoffPos, dt));

            // Rotation. Keep quaternions in the same hemisphere before blending.
            Quaternion target = raw.rotation;
            if (Quaternion.Dot(target, _prevRot) < 0f)
                target = new Quaternion(-target.x, -target.y, -target.z, -target.w);
            float angSpeed = Quaternion.Angle(_prevRot, target) / dt; // deg/s
            _prevAngRate = Mathf.Lerp(_prevAngRate, angSpeed, aRate);
            // Angular beta scaled down: degrees are ~2 orders larger than metres.
            float cutoffRot = _minCutoff + _beta * 0.01f * _prevAngRate;
            Quaternion rot = Quaternion.Slerp(_prevRot, target, Alpha(cutoffRot, dt));

            _prevPos = pos;
            _prevRot = rot;
            return new Pose(pos, rot);
        }
    }
}
