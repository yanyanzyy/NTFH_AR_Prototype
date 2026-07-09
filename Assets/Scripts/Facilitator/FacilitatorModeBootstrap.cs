using UnityEngine;

namespace ARArmDetection.Facilitator
{
    public static class FacilitatorModeBootstrap
    {
        private const string DefaultProcedurePath = "Facilitator/VenepunctureProcedure";

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void CreateForArmDetectionScene()
        {
            return; // Disable temporarily for pose debugging
            if (Object.FindFirstObjectByType<ArmDetectionManager>() == null) return;
            if (Object.FindFirstObjectByType<FacilitatorModeController>() != null) return;

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
