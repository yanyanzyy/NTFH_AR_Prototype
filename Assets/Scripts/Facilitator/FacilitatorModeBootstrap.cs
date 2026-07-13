using UnityEngine;

namespace ARArmDetection.Facilitator
{
    public static class FacilitatorModeBootstrap
    {
        private const string DefaultProcedurePath = "Facilitator/VenepunctureProcedure";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateForArmDetectionScene()
        {
            // Check for EITHER the Arm Detection system OR your new Custom Syringe Detector.
            // This prevents the facilitator from crashing out silently when testing the syringe scene.
            bool hasArmDetection = UnityEngine.Object.FindAnyObjectByType<ArmDetectionManager>() != null;
            bool hasSyringeDetection = UnityEngine.Object.FindAnyObjectByType<CustomSyringeDetector>() != null;

            if (!hasArmDetection && !hasSyringeDetection) return;
            if (UnityEngine.Object.FindAnyObjectByType<FacilitatorModeController>() != null) return;

            var procedure = Resources.Load<FacilitatorProcedure>(DefaultProcedurePath);
            if (procedure == null)
            {
                Debug.LogError($"[Facilitator] Missing Resources/{DefaultProcedurePath}.asset");
                return;
            }

            var go = new GameObject("FacilitatorMode");
            var controller = go.AddComponent<FacilitatorModeController>();
            controller.Initialize(procedure);
        }
    }
}
