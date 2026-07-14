using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// One-off: adds a SyringeAngleEstimator component to SyringePoseDetectionScene_Improved.unity,
/// wired to the same 4 keypoint spheres as the original scene, with the same 15/30 degree
/// acceptable-angle bounds. The original scene got this component added directly (outside this
/// session); the Improved copy predates that and was missing it - SyringeVisualizer/
/// SyringeDebugHUD both null-check angleEstimator so this wasn't crashing, just showing
/// "unassigned" instead of the red/green line + angle readout.
/// </summary>
public static class AddAngleEstimatorToImprovedScene
{
    private const string ScenePath = "Assets/Scenes/SyringePoseDetectionScene_Improved.unity";

    [MenuItem("VPIC/Add SyringeAngleEstimator To Improved Scene")]
    public static void AddEstimator()
    {
        var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

        if (Object.FindFirstObjectByType<SyringeAngleEstimator>() != null)
        {
            Debug.Log("[AddAngleEstimatorToImprovedScene] Scene already has a SyringeAngleEstimator - skipping.");
            return;
        }

        Transform tip = FindSphere("Label_1_NeedleTip");
        Transform barrelTop = FindSphere("Label_2_BarrelTop");
        Transform barrelBottom = FindSphere("Label_3_BarrelBottom");
        Transform plunger = FindSphere("Label_4_Plunger");

        if (tip == null || barrelTop == null || barrelBottom == null || plunger == null)
        {
            Debug.LogError("[AddAngleEstimatorToImprovedScene] Could not find all 4 keypoint spheres by name - aborting.");
            return;
        }

        var go = new GameObject("SyringeAngleEstimator");
        var estimator = go.AddComponent<SyringeAngleEstimator>();

        var so = new SerializedObject(estimator);
        var spheresProp = so.FindProperty("keypointSpheres");
        spheresProp.arraySize = 4;
        spheresProp.GetArrayElementAtIndex(0).objectReferenceValue = tip;
        spheresProp.GetArrayElementAtIndex(1).objectReferenceValue = barrelTop;
        spheresProp.GetArrayElementAtIndex(2).objectReferenceValue = barrelBottom;
        spheresProp.GetArrayElementAtIndex(3).objectReferenceValue = plunger;
        so.FindProperty("minAcceptableAngle").floatValue = 15f;
        so.FindProperty("maxAcceptableAngle").floatValue = 30f;
        so.ApplyModifiedPropertiesWithoutUndo();

        // Wire it into SyringeVisualizer / SyringeDebugHUD explicitly too, rather than relying
        // solely on their FindAnyObjectByType() runtime fallback.
        var visualizer = Object.FindFirstObjectByType<SyringeVisualizer>();
        if (visualizer != null)
        {
            var visSO = new SerializedObject(visualizer);
            visSO.FindProperty("angleEstimator").objectReferenceValue = estimator;
            visSO.ApplyModifiedPropertiesWithoutUndo();
        }

        var hud = Object.FindFirstObjectByType<SyringeDebugHUD>();
        if (hud != null)
        {
            var hudSO = new SerializedObject(hud);
            hudSO.FindProperty("angleEstimator").objectReferenceValue = estimator;
            hudSO.ApplyModifiedPropertiesWithoutUndo();
        }

        EditorSceneManager.MarkSceneDirty(scene);
        bool saved = EditorSceneManager.SaveScene(scene);
        Debug.Log($"[AddAngleEstimatorToImprovedScene] Added SyringeAngleEstimator, wired to 4 spheres " +
                   $"+ visualizer ({visualizer != null}) + HUD ({hud != null}). Scene saved: {saved}.");
    }

    private static Transform FindSphere(string name)
    {
        GameObject go = GameObject.Find(name);
        return go != null ? go.transform : null;
    }
}
