using UnityEditor;
using UnityEngine;

/// <summary>
/// One-off: duplicates SyringePoseDetectionScene.unity into a new scene so the tuning
/// changes made to CustomSyringeDetector.cs / SyringeDebugHUD.cs (letterbox preprocessing,
/// throttled+async inference, per-keypoint confidence display) can be tested side-by-side
/// with the original scene rather than overwriting it. AssetDatabase.CopyAsset handles the
/// new scene's own GUID correctly (unlike a raw filesystem copy, which would need a manually
/// regenerated .meta to avoid a GUID collision with the source scene).
/// </summary>
public static class DuplicateSyringeScene
{
    private const string SourcePath = "Assets/Scenes/SyringePoseDetectionScene.unity";
    private const string DestPath = "Assets/Scenes/SyringePoseDetectionScene_Improved.unity";

    [MenuItem("VPIC/Duplicate SyringePoseDetectionScene (Improved copy)")]
    public static void Duplicate()
    {
        if (AssetDatabase.LoadAssetAtPath<Object>(DestPath) != null)
        {
            Debug.Log($"[DuplicateSyringeScene] {DestPath} already exists — skipping copy, leaving it as-is.");
            return;
        }

        bool ok = AssetDatabase.CopyAsset(SourcePath, DestPath);
        AssetDatabase.Refresh();
        Debug.Log(ok
            ? $"[DuplicateSyringeScene] Copied {SourcePath} -> {DestPath}"
            : $"[DuplicateSyringeScene] AssetDatabase.CopyAsset FAILED for {SourcePath} -> {DestPath}");
    }
}
