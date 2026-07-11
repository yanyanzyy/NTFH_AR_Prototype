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


    // Update is called once per frame
void Update()
    {
        if (!ValidateSpheres())
        {
            CurrentInsertionAngle = 0f;
            IsAngleAcceptable = false;
            return;
        }

        // 1. Calculate the Syringe Direction Vector (From Plunger pointing down to Needle Tip)
        Vector3 syringeDirection = (keypointSpheres[0].position - keypointSpheres[3].position).normalized;

        // 2. Calculate Angle relative to the horizon plane (World Ground)
        float angleToVertical = Vector3.Angle(syringeDirection, Vector3.up);
        
        // Convert from vertical offset to an insertion angle relative to a flat surface
        CurrentInsertionAngle = 90f - angleToVertical;

        // 3. Evaluate Acceptability Rules
        IsAngleAcceptable = (CurrentInsertionAngle >= minAcceptableAngle && CurrentInsertionAngle <= maxAcceptableAngle);
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
