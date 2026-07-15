using UnityEngine;

public class SyringeAngleEstimator : MonoBehaviour
{
    [Header("Keypoint Sphere Transforms")]
    [Tooltip("Assign your 4 tracking spheres in order: 0=NeedleTip, 1=BarrelTop, 2=BarrelBottom, 3=Plunger")]
    [SerializeField] private Transform[] keypointSpheres = new Transform[4];

    [Header("Facilitator Target Metrics")]
    [SerializeField] private float minAcceptableAngle = 15f;  // E.g., Intravenous/Venepuncture angle bounds
    [SerializeField] private float maxAcceptableAngle = 30f;

    // Public properties so your Facilitator scripts can easily read them
    public float CurrentInsertionAngle { get; private set; }
    public bool IsAngleAcceptable { get; private set; }

    /// <summary>True while the 4 keypoint spheres are assigned so an angle can be measured.
    /// Mirrors ARArmDetection.NeedleAngleEstimator so either estimator can drive the same
    /// feedback UI (VeinFeedbackController).</summary>
    public bool HasAngle { get; private set; }
    public float MinAcceptableAngle => minAcceptableAngle;
    public float MaxAcceptableAngle => maxAcceptableAngle;

    [Header("Debug")]
    [Tooltip("Logs raw sphere positions + the computed direction vector/angle to Logcat every " +
             "~1 second - grab an adb logcat capture and look for '[AngleDebug]' lines to see " +
             "whether the 4 spheres are actually at distinct positions or degenerate/overlapping " +
             "(which would explain a constant 90 degree reading - see CurrentInsertionAngle comments).")]
    [SerializeField] private bool logDebugInfo = true;
    private float _lastDebugLogTime;

    // Update is called once per frame
void Update()
    {
        if (!ValidateSpheres())
        {
            CurrentInsertionAngle = 0f;
            IsAngleAcceptable = false;
            HasAngle = false;
            return;
        }

        // 1. Calculate the Syringe Direction Vector (From Plunger pointing down to Needle Tip)
        Vector3 rawDirection = keypointSpheres[0].position - keypointSpheres[3].position;
        Vector3 syringeDirection = rawDirection.normalized;

        // 2. Calculate Angle relative to the horizon plane (World Ground)
        float angleToVertical = Vector3.Angle(syringeDirection, Vector3.up);

        // Convert from vertical offset to an insertion angle relative to a flat surface.
        // Vector3.Angle is unsigned (0-180), so "90 - angleToVertical" alone only comes out
        // positive when the tip points ABOVE horizontal. A real venepuncture insertion (arm
        // resting flat, needle entering the skin at a shallow angle) has the tip pointing
        // slightly BELOW horizontal instead - e.g. a textbook 20 degree insertion gives
        // angleToVertical ~110 degrees, so the un-abs'd formula produced -20, which could never
        // satisfy the positive 15-30 acceptable range. Taking the absolute deviation from
        // horizontal fixes this for both directions of tilt.
        CurrentInsertionAngle = Mathf.Abs(90f - angleToVertical);

        // 3. Evaluate Acceptability Rules
        IsAngleAcceptable = (CurrentInsertionAngle >= minAcceptableAngle && CurrentInsertionAngle <= maxAcceptableAngle);
        HasAngle = true;

        if (logDebugInfo && Time.time - _lastDebugLogTime >= 1f)
        {
            _lastDebugLogTime = Time.time;
            Debug.Log($"[AngleDebug] Tip={keypointSpheres[0].position:F3} BarrelTop={keypointSpheres[1].position:F3} " +
                       $"BarrelBottom={keypointSpheres[2].position:F3} Plunger={keypointSpheres[3].position:F3} " +
                       $"rawDirMagnitude={rawDirection.magnitude:F4} angleToVertical={angleToVertical:F2} " +
                       $"CurrentInsertionAngle={CurrentInsertionAngle:F2}");
        }
    }

    private bool ValidateSpheres()
    {
        if (keypointSpheres == null || keypointSpheres.Length < 4) return false;
        foreach (var sphere in keypointSpheres)
        {
            if (sphere == null) return false;
        }
        return true;
    }
}
